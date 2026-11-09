from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import cv2

from src.tracking.tracker import PersonTracker, draw_tracks

from .line_counter import CrossingCounts, LineCounter


@dataclass(frozen=True)
class FlowVideoSummary:
    input_path: Path
    output_path: Path
    processed_frames: int
    source_fps: float
    average_processing_fps: float
    counts: CrossingCounts


def process_flow_video(
    tracker: PersonTracker,
    input_path: str | Path,
    output_path: str | Path,
    line_y: int | None = None,
    crossing_margin: int = 12,
    *,
    line_x: int | None = None,
    start_second: float = 0.0,
    end_second: float | None = None,
) -> FlowVideoSummary:
    """Tracks people, counts crossings, and writes an annotated MP4 video."""

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
        source_fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if width <= 0 or height <= 0:
            raise ValueError(f"Video has invalid dimensions: {input_path}")
        if (line_y is None) == (line_x is None):
            raise ValueError("Provide exactly one of line_y or line_x.")
        if line_y is not None and not 0 <= line_y < height:
            raise ValueError(f"line_y must be between 0 and {height - 1} for this video.")
        if line_x is not None and not 0 <= line_x < width:
            raise ValueError(f"line_x must be between 0 and {width - 1} for this video.")
        if start_second < 0 or (end_second is not None and end_second <= start_second):
            raise ValueError("Use start_second >= 0 and end_second > start_second.")
        start_frame = int(round(start_second * source_fps))
        end_frame = int(round(end_second * source_fps)) if end_second is not None else frame_count
        if start_frame >= frame_count or end_frame <= start_frame:
            raise ValueError("Selected time range contains no video frames.")
        capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(
            str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), source_fps, (width, height)
        )
        if not writer.isOpened():
            raise OSError(f"OpenCV could not create the output video: {output_path}")

        counter = LineCounter(line_y=line_y, line_x=line_x, crossing_margin=crossing_margin)
        processed_frames = 0
        total_processing_seconds = 0.0
        try:
            while True:
                ok, frame = capture.read()
                if not ok or int(capture.get(cv2.CAP_PROP_POS_FRAMES)) > end_frame:
                    break
                started_at = perf_counter()
                tracks = tracker.track(frame)
                counts = counter.update(tracks)
                annotated = draw_tracks(frame, tracks)
                elapsed_seconds = perf_counter() - started_at
                total_processing_seconds += elapsed_seconds
                processed_frames += 1
                _draw_flow_overlay(annotated, line_y, line_x, counts, elapsed_seconds)
                writer.write(annotated)
        finally:
            writer.release()
    finally:
        capture.release()

    return FlowVideoSummary(
        input_path=input_path,
        output_path=output_path,
        processed_frames=processed_frames,
        source_fps=source_fps,
        average_processing_fps=(
            processed_frames / total_processing_seconds if total_processing_seconds > 0 else 0.0
        ),
        counts=counter.counts,
    )


def _draw_flow_overlay(
    image: object, line_y: int | None, line_x: int | None,
    counts: CrossingCounts, elapsed_seconds: float,
) -> None:
    frame = image
    height, width = frame.shape[:2]
    if line_y is not None:
        cv2.line(frame, (0, line_y), (width - 1, line_y), (0, 0, 255), 2)
        direction_label = "IN: downward | OUT: upward"
        label_position = (12, min(height - 94, max(20, line_y - 8)))
    else:
        cv2.line(frame, (line_x, 0), (line_x, height - 1), (0, 0, 255), 2)
        direction_label = "IN: rightward | OUT: leftward"
        label_position = (min(width - 260, max(12, line_x + 8)), 176)
    cv2.putText(frame, direction_label, label_position,
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.rectangle(frame, (0, 84), (min(width, 340), 154), (20, 20, 20), -1)
    cv2.putText(frame, f"IN: {counts.entered} | OUT: {counts.exited} | TOTAL: {counts.total}",
                (12, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
    fps = 1.0 / elapsed_seconds if elapsed_seconds > 0 else 0.0
    cv2.putText(frame, f"Processing FPS: {fps:.1f}", (12, 140),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
