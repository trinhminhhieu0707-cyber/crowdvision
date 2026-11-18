import numpy as np
import pytest

from src.detection.detector import Detection, PersonDetector, _merge_detections, _tile_starts


def _tiled_detector(per_crop) -> PersonDetector:
    """Tạo PersonDetector mà không tải YOLO: detect() được thay bằng hàm giả `per_crop(crop)`."""
    detector = object.__new__(PersonDetector)
    detector.detect = per_crop  # type: ignore[method-assign]
    detector.iou_threshold = 0.45
    return detector


def test_tile_starts_cover_whole_length_and_touch_edge() -> None:
    starts = _tile_starts(1000, 320, 0.25)

    assert starts[0] == 0 and starts[-1] == 1000 - 320
    assert all(b > a for a, b in zip(starts, starts[1:]))
    assert _tile_starts(300, 320, 0.25) == [0]


def test_merge_removes_duplicate_and_cut_in_half_boxes() -> None:
    full = Detection(100, 100, 140, 200, 0.9)
    duplicate = Detection(101, 100, 141, 201, 0.6)
    half = Detection(100, 100, 140, 150, 0.4)  # người bị cắt đôi ở mép ô (IoU thấp nhưng nằm gọn trong `full`)
    other = Detection(300, 100, 340, 200, 0.5)

    kept = _merge_detections([half, duplicate, other, full], threshold=0.5)

    assert full in kept and other in kept
    assert len(kept) == 2


def test_detect_tiled_shifts_boxes_back_to_full_image_coordinates() -> None:
    calls: list[tuple[int, int]] = []

    def per_crop(crop: np.ndarray) -> list[Detection]:
        calls.append(crop.shape[:2])
        if crop.shape[:2] == (200, 400):  # lượt chạy ảnh gốc: không thấy ai
            return []
        return [Detection(10, 10, 30, 50, 0.9)]  # mỗi ô thấy 1 người ở góc trên trái của ô

    detector = _tiled_detector(per_crop)
    image = np.zeros((200, 400, 3), np.uint8)

    result = detector.detect_tiled(image, tile_size=200, overlap=0.0)

    assert len(calls) == 3  # 1 ảnh gốc + 2 ô (400/200 theo chiều ngang, 1 theo chiều dọc)
    assert sorted((d.x1, d.y1) for d in result) == [(10, 10), (210, 10)]


def test_detect_tiled_small_image_falls_back_to_plain_detect() -> None:
    calls: list[int] = []

    def per_crop(crop: np.ndarray) -> list[Detection]:
        calls.append(1)
        return [Detection(1, 1, 5, 9, 0.8)]

    detector = _tiled_detector(per_crop)

    result = detector.detect_tiled(np.zeros((100, 120, 3), np.uint8), tile_size=320)

    assert len(calls) == 1 and len(result) == 1


def test_detect_tiled_validates_arguments() -> None:
    detector = _tiled_detector(lambda crop: [])
    image = np.zeros((50, 50, 3), np.uint8)

    with pytest.raises(ValueError):
        detector.detect_tiled(image, tile_size=0)
    with pytest.raises(ValueError):
        detector.detect_tiled(image, overlap=1.0)
