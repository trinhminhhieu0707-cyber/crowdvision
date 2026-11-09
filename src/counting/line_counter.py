from __future__ import annotations

from dataclasses import dataclass

from src.tracking.tracker import TrackedPerson


@dataclass(frozen=True)
class CrossingCounts:
    entered: int
    exited: int

    @property
    def total(self) -> int:
        return self.entered + self.exited


class LineCounter:
    """Counts tracked people crossing a horizontal or vertical line."""

    def __init__(
        self,
        line_y: int | None = None,
        crossing_margin: int = 12,
        *,
        line_x: int | None = None,
    ) -> None:
        if (line_y is None) == (line_x is None):
            raise ValueError("Provide exactly one of line_y or line_x.")
        coordinate = line_y if line_y is not None else line_x
        if coordinate is None or coordinate < 0:
            raise ValueError("Line coordinate must be non-negative.")
        if crossing_margin < 0:
            raise ValueError("crossing_margin must be non-negative.")

        self.orientation = "horizontal" if line_y is not None else "vertical"
        self.line_y = line_y
        self.line_x = line_x
        self.crossing_margin = crossing_margin
        self._last_stable_side: dict[int, int] = {}
        self._entered = 0
        self._exited = 0

    @property
    def counts(self) -> CrossingCounts:
        return CrossingCounts(entered=self._entered, exited=self._exited)

    def update(self, tracks: list[TrackedPerson]) -> CrossingCounts:
        """Updates counts from active tracks; downward is IN and upward is OUT."""

        for track in tracks:
            coordinate = _bottom_center_y(track) if self.orientation == "horizontal" else _center_x(track)
            side = self._side_of_line(coordinate)
            if side == 0:
                continue

            previous_side = self._last_stable_side.get(track.track_id)
            if previous_side is not None and previous_side != side:
                if previous_side < side:
                    self._entered += 1
                else:
                    self._exited += 1
            self._last_stable_side[track.track_id] = side
        return self.counts

    def _side_of_line(self, coordinate: int) -> int:
        line_coordinate = self.line_y if self.orientation == "horizontal" else self.line_x
        assert line_coordinate is not None
        if coordinate < line_coordinate - self.crossing_margin:
            return -1
        if coordinate > line_coordinate + self.crossing_margin:
            return 1
        return 0


def _bottom_center_y(track: TrackedPerson) -> int:
    return track.y2


def _center_x(track: TrackedPerson) -> int:
    return round((track.x1 + track.x2) / 2)
