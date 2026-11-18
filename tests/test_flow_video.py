from pathlib import Path

import numpy as np
import pytest

from src.counting.flow_video import process_flow_video
from src.tracking.tracker import TrackedPerson
from src.video_io import open_writer


class _WalkingTracker:
    """Một người đi từ trên xuống dưới qua y=48; đếm số lần reset()."""

    def __init__(self, frames: int) -> None:
        self.ys = np.linspace(10, 90, frames)
        self.frame = 0
        self.resets = 0

    def reset(self) -> None:
        self.resets += 1

    def track(self, frame: np.ndarray) -> list[TrackedPerson]:
        y2 = int(self.ys[self.frame])
        self.frame += 1
        return [TrackedPerson(1, 50, y2 - 30, 70, y2, 0.9)]


def _make_video(path: Path, frames: int) -> None:
    with open_writer(path, 10, (128, 96)) as writer:
        for _ in range(frames):
            writer.write(np.zeros((96, 128, 3), np.uint8))


def test_flow_video_counts_one_entry_and_resets_tracker(tmp_path: Path) -> None:
    video = tmp_path / "in.mp4"
    _make_video(video, frames=20)
    tracker = _WalkingTracker(frames=20)

    summary = process_flow_video(tracker, video, tmp_path / "out.mp4", line_y=48, crossing_margin=4)

    assert summary.counts.entered == 1 and summary.counts.exited == 0
    assert summary.counts.net == 1
    assert summary.processed_frames == 20
    assert tracker.resets == 1
    assert (tmp_path / "out.mp4").stat().st_size > 0


def test_flow_video_rejects_line_outside_frame(tmp_path: Path) -> None:
    video = tmp_path / "in.mp4"
    _make_video(video, frames=3)

    with pytest.raises(ValueError):
        process_flow_video(_WalkingTracker(3), video, tmp_path / "out.mp4", line_y=500)
    assert not (tmp_path / "out.mp4").exists()  # validate xong mới tạo file kết quả (giữ nguyên hành vi cũ)
