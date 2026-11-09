from __future__ import annotations

import csv
from pathlib import Path

import cv2
import numpy as np

from .analysis import annotate_density_frame, find_dense_regions
from .csrnet import CSRNetCounter


def _save_count_trend(rows: list[dict[str, float]], destination: Path) -> None:
    width, height = 1000, 500
    canvas = np.full((height, width, 3), 248, dtype=np.uint8)
    left, right, top, bottom = 80, width - 35, 45, height - 65
    cv2.putText(canvas, "CSRNet estimated crowd count over video", (left, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (35, 35, 35), 2, cv2.LINE_AA)
    cv2.line(canvas, (left, top), (left, bottom), (60, 60, 60), 1)
    cv2.line(canvas, (left, bottom), (right, bottom), (60, 60, 60), 1)
    if not rows:
        cv2.putText(canvas, "No frames were analyzed", (left + 20, top + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (20, 20, 20), 2)
    else:
        counts = np.array([max(0.0, row["estimated_count"]) for row in rows], dtype=float)
        times = np.array([row["timestamp_seconds"] for row in rows], dtype=float)
        max_count = max(float(counts.max()), 1.0)
        duration = max(float(times.max()), 1.0)
        points = []
        for time_s, count in zip(times, counts, strict=True):
            x = int(left + (right - left) * time_s / duration)
            y = int(bottom - (bottom - top) * count / max_count)
            points.append((x, y))
        if len(points) == 1:
            cv2.circle(canvas, points[0], 5, (30, 90, 220), -1)
        else:
            cv2.polylines(canvas, [np.asarray(points, np.int32)], False, (30, 90, 220), 2, cv2.LINE_AA)
            for point in points:
                cv2.circle(canvas, point, 3, (30, 90, 220), -1)
        cv2.putText(canvas, f"0 - {max_count:.0f} people", (8, top + 8), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (45, 45, 45), 1, cv2.LINE_AA)
        cv2.putText(canvas, f"0 - {duration:.1f}s", (right - 100, bottom + 35), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (45, 45, 45), 1, cv2.LINE_AA)
    cv2.imwrite(str(destination), canvas)


def analyze_video_density(
    video_path: str | Path,
    output_dir: str | Path,
    counter: CSRNetCounter,
    sample_every: int = 15,
    grid_rows: int = 4,
    grid_cols: int = 4,
    top_regions: int = 3,
) -> dict[str, Path | int | float]:
    """Estimate crowd density on sampled frames; this does not track people."""
    video_path, output_dir = Path(video_path), Path(output_dir)
    if not video_path.is_file():
        raise FileNotFoundError(f"Input video does not exist: {video_path}")
    if sample_every < 1:
        raise ValueError("sample_every must be at least 1.")
    output_dir.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(f"OpenCV could not open the video: {video_path}")
    fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer_path = output_dir / f"{video_path.stem}_density.mp4"
    writer = cv2.VideoWriter(str(writer_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        capture.release()
        raise OSError(f"Could not create output video: {writer_path}")

    csv_path = output_dir / "density_counts.csv"
    chart_path = output_dir / "density_trend.png"
    rows: list[dict[str, float]] = []
    frame_index = 0
    last_prediction = None
    last_regions = []
    last_sample_time = 0.0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if frame_index % sample_every == 0:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                last_prediction = counter.predict_rgb(rgb)
                last_sample_time = frame_index / fps
                last_regions = find_dense_regions(last_prediction.density_map, (height, width), grid_rows, grid_cols, top_regions)
                rows.append({
                    "frame": float(frame_index),
                    "timestamp_seconds": float(last_sample_time),
                    "estimated_count": float(last_prediction.count),
                })
            if last_prediction is not None:
                writer.write(annotate_density_frame(frame, last_prediction, last_regions, last_sample_time))
            frame_index += 1
    finally:
        capture.release()
        writer.release()

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer_csv = csv.DictWriter(f, fieldnames=["frame", "timestamp_seconds", "estimated_count"])
        writer_csv.writeheader()
        writer_csv.writerows(rows)
    _save_count_trend(rows, chart_path)
    return {
        "frames": frame_index,
        "sampled_frames": len(rows),
        "source_fps": float(fps),
        "video": writer_path,
        "counts_csv": csv_path,
        "trend_chart": chart_path,
    }
