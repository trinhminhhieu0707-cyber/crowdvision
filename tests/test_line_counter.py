from src.counting import LineCounter
from src.tracking.tracker import TrackedPerson


def person(track_id: int, bottom_y: int) -> TrackedPerson:
    return TrackedPerson(track_id, 10, bottom_y - 30, 40, bottom_y, 0.9)


def test_counts_downward_crossing_as_in() -> None:
    counter = LineCounter(line_y=100, crossing_margin=5)
    counter.update([person(1, 80)])
    counts = counter.update([person(1, 120)])
    assert (counts.entered, counts.exited) == (1, 0)


def test_counts_upward_crossing_as_out() -> None:
    counter = LineCounter(line_y=100, crossing_margin=5)
    counter.update([person(1, 120)])
    counts = counter.update([person(1, 80)])
    assert (counts.entered, counts.exited) == (0, 1)


def test_does_not_count_jitter_inside_line_margin() -> None:
    counter = LineCounter(line_y=100, crossing_margin=5)
    counter.update([person(1, 80)])
    counter.update([person(1, 99)])
    counter.update([person(1, 102)])
    counts = counter.update([person(1, 104)])
    assert (counts.entered, counts.exited) == (0, 0)


def test_counts_a_true_return_crossing_once_per_direction() -> None:
    counter = LineCounter(line_y=100, crossing_margin=5)
    counter.update([person(1, 80)])
    counter.update([person(1, 120)])
    counts = counter.update([person(1, 80)])
    assert (counts.entered, counts.exited) == (1, 1)
