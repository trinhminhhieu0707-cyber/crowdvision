import numpy as np

from src.tracking.tracker import TrackedPerson, draw_tracks


def test_draw_tracks_returns_annotated_frame() -> None:
    frame = np.zeros((100, 120, 3), dtype=np.uint8)
    tracks = [TrackedPerson(7, 10, 20, 70, 90, 0.92)]

    annotated = draw_tracks(frame, tracks)

    assert annotated.shape == frame.shape
    assert not np.array_equal(annotated, frame)
