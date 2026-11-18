
from types import SimpleNamespace

import numpy as np

from src.tracking.tracker import PersonTracker


class _FakeTensor:
    def __init__(self, array: np.ndarray) -> None:
        self._array = array

    def cpu(self) -> "_FakeTensor":
        return self

    def numpy(self) -> np.ndarray:
        return self._array

    def int(self) -> "_FakeTensor":
        return _FakeTensor(self._array.astype(int))


class _FakeModel:
    """Giả lập ultralytics.YOLO: ghi lại tham số `conf` và trả về 3 box với điểm 0.9 / 0.2 / 0.05."""

    def __init__(self) -> None:
        self.predictor = object()
        self.received_conf: float | None = None

    def track(self, **kwargs):
        self.received_conf = kwargs["conf"]
        boxes = SimpleNamespace(
            xyxy=_FakeTensor(np.array([[0, 0, 10, 10], [20, 20, 30, 30], [40, 40, 50, 50]], float)),
            conf=_FakeTensor(np.array([0.9, 0.2, 0.05])),
            id=_FakeTensor(np.array([1, 2, 3])),
        )
        return [SimpleNamespace(boxes=boxes)]


def _make_tracker(confidence: float = 0.35, low: float = 0.1) -> tuple[PersonTracker, _FakeModel]:
    tracker = object.__new__(PersonTracker)
    model = _FakeModel()
    tracker.model = model
    tracker.person_class_id = 0
    tracker.confidence_threshold = confidence
    tracker.iou_threshold = 0.45
    tracker.image_size = 640
    tracker.device = None
    tracker.low_confidence_threshold = low
    tracker._tracker_conf = min(low, confidence)
    return tracker, model


def test_track_passes_low_conf_to_model_but_filters_output() -> None:
    tracker, model = _make_tracker(confidence=0.35, low=0.1)

    people = tracker.track(np.zeros((64, 64, 3), np.uint8))

    assert model.received_conf == 0.1          # ByteTrack nhìn thấy cả box điểm thấp
    assert [p.track_id for p in people] == [1]  # nhưng chỉ box >= 0.35 được trả ra


def test_tracker_conf_never_exceeds_display_threshold() -> None:
    tracker, model = _make_tracker(confidence=0.05, low=0.1)

    tracker.track(np.zeros((8, 8, 3), np.uint8))

    assert model.received_conf == 0.05


def test_reset_clears_predictor_so_trackers_are_rebuilt() -> None:
    tracker, model = _make_tracker()
    assert model.predictor is not None

    tracker.reset()

    assert model.predictor is None
