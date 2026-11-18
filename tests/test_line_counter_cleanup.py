import pytest

from src.counting.line_counter import CrossingCounts, LineCounter
from src.tracking.tracker import TrackedPerson


def _at(track_id: int, y2: int) -> TrackedPerson:
    return TrackedPerson(track_id, 40, y2 - 40, 60, y2, 0.9)


def test_net_is_entered_minus_exited() -> None:
    assert CrossingCounts(entered=5, exited=2).net == 3
    assert CrossingCounts(entered=1, exited=4).net == -3


def test_idle_tracks_are_forgotten() -> None:
    counter = LineCounter(line_y=100, crossing_margin=5, max_idle_frames=3)
    counter.update([_at(1, 50)])  # track 1 ở phía trên vạch
    assert 1 in counter._last_stable_side

    for _ in range(5):  # track 1 biến mất 5 frame > max_idle_frames
        counter.update([_at(2, 50)])

    assert 1 not in counter._last_stable_side
    assert 1 not in counter._last_seen
    assert 2 in counter._last_stable_side  # track còn hoạt động thì giữ lại


def test_active_track_in_margin_is_not_forgotten() -> None:
    counter = LineCounter(line_y=100, crossing_margin=10, max_idle_frames=2)
    counter.update([_at(1, 50)])
    for _ in range(6):  # đứng trong vùng đệm (y2=100) nhiều frame: vẫn "được thấy"
        counter.update([_at(1, 100)])

    counts = counter.update([_at(1, 150)])  # rồi đi xuống phía dưới vạch

    assert counts.entered == 1  # vẫn nhớ phía trước đó là "trên" nên đếm được 1 lượt vào


def test_forgotten_track_cannot_be_counted_after_long_absence() -> None:
    counter = LineCounter(line_y=100, crossing_margin=5, max_idle_frames=2)
    counter.update([_at(1, 50)])
    for _ in range(4):
        counter.update([])

    counts = counter.update([_at(1, 150)])  # cùng ID nhưng đã bị quên -> coi như track mới, không đếm

    assert counts.entered == 0


def test_max_idle_frames_must_be_positive() -> None:
    with pytest.raises(ValueError):
        LineCounter(line_y=10, max_idle_frames=0)
