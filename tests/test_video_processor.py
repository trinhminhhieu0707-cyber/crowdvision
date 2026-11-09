from pathlib import Path

import cv2
import numpy as np

from src.detection.detector import Detection
from src.detection.video_processor import process_video


class StaticDetector:
    def detect(self, image: np.ndarray) -> list[Detection]:
        return [Detection(8, 8, 32, 40, 0.90)]


def test_process_video_writes_annotated_video(tmp_path: Path) -> None:
    input_path = tmp_path / "input.mp4"
    output_path = tmp_path / "output.mp4"
    writer = cv2.VideoWriter(
        str(input_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        10.0,
        (64, 48),
    )
    if not writer.isOpened():
        raise RuntimeError("OpenCV test video writer is unavailable.")
    for _ in range(3):
        writer.write(np.zeros((48, 64, 3), dtype=np.uint8))
    writer.release()

    summary = process_video(StaticDetector(), input_path, output_path)

    assert summary.processed_frames == 3
    assert summary.source_fps == 10.0
    assert summary.average_processing_fps > 0
    assert output_path.is_file()
    assert output_path.stat().st_size > 0
