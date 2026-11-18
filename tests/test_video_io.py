from pathlib import Path

import numpy as np
import pytest

from src.video_io import open_video, open_writer


def _make_video(path: Path, frames: int = 5, fps: int = 10, size: tuple[int, int] = (64, 48)) -> None:
    with open_writer(path, fps, size) as writer:
        for _ in range(frames):
            writer.write(np.zeros((size[1], size[0], 3), np.uint8))


def test_open_video_reports_metadata(tmp_path: Path) -> None:
    video = tmp_path / "clip.mp4"
    _make_video(video, frames=5, fps=10, size=(64, 48))

    with open_video(video) as (capture, info):
        assert (info.width, info.height) == (64, 48)
        assert info.fps == pytest.approx(10.0)
        assert info.frame_count == 5
        ok, frame = capture.read()
        assert ok and frame.shape == (48, 64, 3)


def test_open_video_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        with open_video(tmp_path / "nope.mp4"):
            pass


def test_open_video_rejects_non_video_file(tmp_path: Path) -> None:
    fake = tmp_path / "fake.mp4"
    fake.write_bytes(b"this is not a video")

    with pytest.raises(ValueError):
        with open_video(fake):
            pass


def test_open_writer_creates_parent_folders(tmp_path: Path) -> None:
    target = tmp_path / "a" / "b" / "out.mp4"

    _make_video(target)

    assert target.is_file() and target.stat().st_size > 0
