from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import cv2

## [SỬA 6a | video_io.py | FILE MỚI] Helper dùng chung để mở video đọc/ghi.
## Trước đây đoạn "kiểm tra file -> mở VideoCapture -> lấy width/height/fps -> tạo VideoWriter -> release"
## bị copy ở 5 nơi (video_processor, tracker, flow_video, zone_monitor, crowd_counting/video).
## Giờ cả 5 nơi dùng 2 context manager dưới đây: tự kiểm tra lỗi và tự release, kể cả khi có exception.

DEFAULT_FPS = 25.0


@dataclass(frozen=True)
class VideoInfo:
    width: int
    height: int
    fps: float
    frame_count: int


@contextmanager
def open_video(path: str | Path) -> Iterator[tuple[cv2.VideoCapture, VideoInfo]]:
    """Open a video for reading; validates the file and releases the capture on exit."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Input video does not exist: {path}")
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise ValueError(f"OpenCV could not open the video: {path}")
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if width <= 0 or height <= 0:
            raise ValueError(f"Video has invalid dimensions: {path}")
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        if not fps > 0:  # xử lý cả 0, âm và NaN
            fps = DEFAULT_FPS
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        yield capture, VideoInfo(width=width, height=height, fps=fps, frame_count=frame_count)
    finally:
        capture.release()


@contextmanager
def open_writer(path: str | Path, fps: float, size: tuple[int, int]) -> Iterator[cv2.VideoWriter]:
    """Open an MP4 writer (creates parent folders); releases the writer on exit."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # noinspection PyUnresolvedReferences
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, size)
    try:
        if not writer.isOpened():
            raise OSError(f"OpenCV could not create the output video: {path}")
        yield writer
    finally:
        writer.release()
