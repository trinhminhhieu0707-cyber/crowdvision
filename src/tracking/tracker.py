from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import cv2
import numpy as np
from ultralytics import YOLO


@dataclass(frozen=True)
class TrackedPerson:
    track_id: int
    x1: int
    y1: int
    x2: int
    y2: int
    confidence: float


@dataclass(frozen=True)
class TrackingVideoSummary:
    input_path: Path
    output_path: Path
    processed_frames: int
    source_fps: float
    average_processing_fps: float


class PersonTracker:
    """Associates person detections across consecutive video frames with ByteTrack."""

    def __init__(
        self,
        weights: str | Path = "yolo11s.pt",
        confidence_threshold: float = 0.35,
        iou_threshold: float = 0.45,
        image_size: int = 640,
        device: str | None = None,
    ) -> None:
        if not 0.0 <= confidence_threshold <= 1.0:
            raise ValueError("confidence_threshold must be between 0 and 1.")
        if not 0.0 <= iou_threshold <= 1.0:
            raise ValueError("iou_threshold must be between 0 and 1.")
        if image_size <= 0:
            raise ValueError("image_size must be positive.")

        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.image_size = image_size
        self.device = device
        self.model = YOLO(str(weights))
        self.person_class_id = self._find_person_class_id()

    def track(self, frame: np.ndarray) -> list[TrackedPerson]:
        if frame is None or frame.size == 0:
            raise ValueError("frame must be a non-empty OpenCV image.")

        result = self.model.track(
            source=frame,
            persist=True,
            tracker="bytetrack.yaml",
            classes=[self.person_class_id],
            conf=self.confidence_threshold,
            iou=self.iou_threshold,
            imgsz=self.image_size,
            device=self.device,
            verbose=False,
        )[0]

        if result.boxes is None or result.boxes.id is None:
            return []

        boxes = result.boxes.xyxy.cpu().numpy()
        confidences = result.boxes.conf.cpu().numpy()
        track_ids = result.boxes.id.int().cpu().numpy()
        return [
            TrackedPerson(
                track_id=int(track_id),
                x1=int(round(x1)),
                y1=int(round(y1)),
                x2=int(round(x2)),
                y2=int(round(y2)),
                confidence=float(confidence),
            )
            for (x1, y1, x2, y2), confidence, track_id in zip(
                boxes, confidences, track_ids, strict=True
            )
        ]

    def _find_person_class_id(self) -> int:
        names: Any = self.model.names
        for class_id, class_name in names.items():
            if class_name == "person":
                return int(class_id)
        raise ValueError("The selected YOLO weights do not contain a 'person' class.")


def process_tracking_video(
    tracker: PersonTracker,
    input_path: str | Path,
    output_path: str | Path,
) -> TrackingVideoSummary:
    """Tracks people in a video and writes annotated frames to an MP4 file."""

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
                tracks = tracker.track(frame)
                annotated = draw_tracks(frame, tracks)
                elapsed_seconds = perf_counter() - frame_started_at
                total_processing_seconds += elapsed_seconds
                processed_frames += 1

                _draw_processing_fps(annotated, elapsed_seconds)
                writer.write(annotated)
        finally:
            writer.release()
    finally:
        capture.release()

    average_processing_fps = (
        processed_frames / total_processing_seconds if total_processing_seconds > 0 else 0.0
    )
    return TrackingVideoSummary(
        input_path=input_path,
        output_path=output_path,
        processed_frames=processed_frames,
        source_fps=source_fps,
        average_processing_fps=average_processing_fps,
    )


def draw_tracks(frame: np.ndarray, tracks: list[TrackedPerson]) -> np.ndarray:
    """Returns a copy of the frame with active ByteTrack IDs drawn on it."""

    annotated = frame.copy()
    height, width = annotated.shape[:2]
    for track in tracks:
        x1 = max(0, min(track.x1, width - 1))
        y1 = max(0, min(track.y1, height - 1))
        x2 = max(0, min(track.x2, width - 1))
        y2 = max(0, min(track.y2, height - 1))
        label = f"ID {track.track_id} | person {track.confidence:.2f}"
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (255, 160, 0), 2)
        cv2.putText(
            annotated,
            label,
            (x1, max(22, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 160, 0),
            2,
            cv2.LINE_AA,
        )

    cv2.rectangle(annotated, (0, 0), (340, 42), (20, 20, 20), -1)
    cv2.putText(
        annotated,
        f"Tracked people in frame: {len(tracks)}",
        (12, 29),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return annotated


def _draw_processing_fps(image: np.ndarray, elapsed_seconds: float) -> None:
    processing_fps = 1.0 / elapsed_seconds if elapsed_seconds > 0 else 0.0
    height, width = image.shape[:2]
    cv2.rectangle(image, (0, 44), (min(width, 300), 82), (20, 20, 20), -1)
    cv2.putText(
        image,
        f"Processing FPS: {processing_fps:.1f}",
        (12, min(height - 12, 70)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
