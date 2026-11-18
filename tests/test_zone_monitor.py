
import csv
import json
from pathlib import Path

import numpy as np
import pytest

from src.counting.zone_monitor import Zone, assign_tracks, load_zones, process_zone_video
from src.tracking.tracker import TrackedPerson
from src.video_io import open_writer

FULL_FRAME = Zone("all", ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)), capacity=2)


def _person(track_id: int, x: int = 50, y2: int = 90) -> TrackedPerson:
    return TrackedPerson(track_id, x - 5, y2 - 40, x + 5, y2, 0.9)


class _ScriptedTracker:
    """Trả về số người theo kịch bản `counts[frame_index]`; đếm số lần reset()."""

    def __init__(self, counts: list[int]) -> None:
        self.counts = counts
        self.frame = 0
        self.resets = 0

    def reset(self) -> None:
        self.resets += 1

    def track(self, frame: np.ndarray) -> list[TrackedPerson]:
        count = self.counts[self.frame]
        self.frame += 1
        return [_person(i + 1, x=10 + i * 6) for i in range(count)]


def _write_video(path: Path, frames: int, fps: int = 10) -> None:
    with open_writer(path, fps, (128, 96)) as writer:
        for _ in range(frames):
            writer.write(np.zeros((96, 128, 3), np.uint8))


def _alerts(summary) -> list[dict[str, str]]:
    with summary.alerts_path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def test_assign_tracks_uses_footpoint_and_first_zone_wins() -> None:
    left = Zone("left", ((0.0, 0.0), (0.5, 0.0), (0.5, 1.0), (0.0, 1.0)), 5)
    overlap = Zone("overlap", ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)), 5)
    tracks = [_person(1, x=20), _person(2, x=100)]  # track 1 ở nửa trái, track 2 ở nửa phải

    result = assign_tracks(tracks, [left, overlap], width=128, height=96)

    assert result["left"] == {1}
    assert result["overlap"] == {2}  # track 1 đã thuộc "left" (zone khai báo trước)


def test_load_zones_rejects_invalid_config(tmp_path: Path) -> None:
    bad = {"zones": [{"name": "a", "polygon": [[0, 0], [1, 0]], "capacity": 3}]}
    path = tmp_path / "zones.json"
    path.write_text(json.dumps(bad), encoding="utf-8")

    with pytest.raises(ValueError):
        load_zones(path)


def test_alert_ignores_short_blip(tmp_path: Path) -> None:
    video = tmp_path / "in.mp4"
    _write_video(video, frames=12)
    counts = [1, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]  # 1 frame vượt rồi tụt
    tracker = _ScriptedTracker(counts)

    summary = process_zone_video(
        tracker, video, tmp_path / "out.mp4", [FULL_FRAME], alert_min_seconds=0.3, alert_clear_seconds=0.3
    )

    assert _alerts(summary) == []
    assert tracker.resets == 1


def test_alert_fires_once_for_sustained_overcapacity_and_not_on_flicker(tmp_path: Path) -> None:
    video = tmp_path / "in.mp4"
    _write_video(video, frames=14)
    counts = [1, 3, 3, 3, 3, 1, 3, 3, 3, 3, 1, 1, 1, 1]
    tracker = _ScriptedTracker(counts)

    summary = process_zone_video(
        tracker, video, tmp_path / "out.mp4", [FULL_FRAME], alert_min_seconds=0.3, alert_clear_seconds=0.3
    )

    rows = _alerts(summary)
    assert len(rows) == 1                    # 1 cảnh báo cho cả đợt, không bắn lại sau khi dao động
    assert rows[0]["frame"] == "3"           # xác nhận ở frame thứ 3 liên tiếp (0.3 giây @10 FPS)
    assert rows[0]["first_exceeded_frame"] == "1"


def test_alert_rearms_after_clear_period(tmp_path: Path) -> None:
    video = tmp_path / "in.mp4"
    _write_video(video, frames=16)
    counts = [3, 3, 3, 1, 1, 1, 1, 1, 3, 3, 3, 3, 1, 1, 1, 1]  # hai đợt vượt cách nhau 5 frame
    tracker = _ScriptedTracker(counts)

    summary = process_zone_video(
        tracker, video, tmp_path / "out.mp4", [FULL_FRAME], alert_min_seconds=0.3, alert_clear_seconds=0.3
    )

    assert len(_alerts(summary)) == 2


def test_zero_seconds_restores_immediate_alert(tmp_path: Path) -> None:
    video = tmp_path / "in.mp4"
    _write_video(video, frames=4)
    tracker = _ScriptedTracker([1, 3, 1, 3])

    summary = process_zone_video(
        tracker, video, tmp_path / "out.mp4", [FULL_FRAME], alert_min_seconds=0, alert_clear_seconds=0
    )

    assert len(_alerts(summary)) == 2  # hành vi cũ: mỗi lần chuyển sang over_capacity là một cảnh báo
