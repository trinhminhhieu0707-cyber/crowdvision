import csv
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from src.crowd_counting.video import analyze_video_density
from src.video_io import open_writer


class _FakeCounter:
    def __init__(self) -> None:
        self.calls = 0

    def predict_rgb(self, rgb: np.ndarray):
        self.calls += 1
        return SimpleNamespace(count=7.0, density_map=np.full((12, 16), 7.0 / (12 * 16), np.float32))


def _make_video(path: Path, frames: int) -> None:
    with open_writer(path, 10, (128, 96)) as writer:
        for _ in range(frames):
            writer.write(np.zeros((96, 128, 3), np.uint8))


def test_analyze_video_density_samples_every_n_frames(tmp_path: Path) -> None:
    video = tmp_path / "crowd.mp4"
    _make_video(video, frames=10)
    counter = _FakeCounter()

    result = analyze_video_density(video, tmp_path / "out", counter, sample_every=4)

    assert result["frames"] == 10
    assert result["sampled_frames"] == 3 and counter.calls == 3  # frame 0, 4, 8
    with Path(result["counts_csv"]).open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert [r["frame"] for r in rows] == ["0.0", "4.0", "8.0"]
    assert Path(result["video"]).stat().st_size > 0 and Path(result["trend_chart"]).is_file()


def test_analyze_video_density_validates_inputs(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        analyze_video_density(tmp_path / "x.mp4", tmp_path / "out", _FakeCounter(), sample_every=0)
    with pytest.raises(FileNotFoundError):
        analyze_video_density(tmp_path / "missing.mp4", tmp_path / "out", _FakeCounter())
