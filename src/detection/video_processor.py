from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Protocol

import cv2
import numpy as np

from .detector import Detection, draw_detections


class Detector(Protocol):
    def detect(self, image: np.ndarray) -> list[Detection]: ...


@dataclass(frozen=True)
class VideoProcessingSummary:
    input_path: Path
    output_path: Path
    processed_frames: int
    source_fps: float
    average_processing_fps: float


def process_video(
    detector: Detector,
    input_path: str | Path,
    output_path: str | Path,
) -> VideoProcessingSummary:
    """Detects people in each frame and writes an annotated MP4 video."""

    input_path = Path(input_path)
    output_path = Path(output_path)
    if not input_path.is_file():
        raise FileNotFoundError(f"Input video does not exist: {input_path}")

    capture = cv2.VideoCapture(str(input_path))
    if not capture.isOpened():
        raise ValueError(f"OpenCV could not open the video: {input_path}")

    try:
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        source_fps = capture.get(cv2.CAP_PROP_FPS)
        if width <= 0 or height <= 0:
            raise ValueError(f"Video has invalid dimensions: {input_path}")
        if source_fps <= 0:
            source_fps = 25.0

        output_path.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(
            str(output_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            source_fps,
            (width, height),
        )
        if not writer.isOpened():
            raise OSError(f"OpenCV could not create the output video: {output_path}")

        try:
            processed_frames = 0
            total_processing_seconds = 0.0
            while True:
                ok, frame = capture.read()
                if not ok:
                    break

                frame_started_at = perf_counter()
                detections = detector.detect(frame)
                annotated = draw_detections(frame, detections)
                elapsed_seconds = perf_counter() - frame_started_at
                total_processing_seconds += elapsed_seconds
                processed_frames += 1

                _draw_video_metrics(
                    annotated,
                    processing_fps=1.0 / elapsed_seconds if elapsed_seconds > 0 else 0.0,
                    source_fps=source_fps,
                )
                writer.write(annotated)
        finally:
            writer.release()
    finally:
        capture.release()

    average_processing_fps = (
        processed_frames / total_processing_seconds if total_processing_seconds > 0 else 0.0
    )
    return VideoProcessingSummary(
        input_path=input_path,
        output_path=output_path,
        processed_frames=processed_frames,
        source_fps=source_fps,
        average_processing_fps=average_processing_fps,
    )


def _draw_video_metrics(image: np.ndarray, processing_fps: float, source_fps: float) -> None:
    text = f"Processing FPS: {processing_fps:.1f} | Source FPS: {source_fps:.1f}"
    height, width = image.shape[:2]
    text_y = min(height - 12, 70)
    cv2.rectangle(image, (0, 44), (min(width, 430), 82), (20, 20, 20), -1)
    cv2.putText(
        image,
        text,
        (12, text_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
