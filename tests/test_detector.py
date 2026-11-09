import numpy as np

from src.detection.detector import Detection, draw_detections


def test_draw_detections_returns_annotated_image() -> None:
    image = np.zeros((100, 120, 3), dtype=np.uint8)
    detections = [Detection(10, 20, 70, 90, 0.92)]

    annotated = draw_detections(image, detections)

    assert annotated.shape == image.shape
    assert not np.array_equal(annotated, image)


def test_draw_detections_handles_no_people() -> None:
    image = np.zeros((100, 120, 3), dtype=np.uint8)

    annotated = draw_detections(image, [])

    assert annotated.shape == image.shape
    assert not np.array_equal(annotated, image)
