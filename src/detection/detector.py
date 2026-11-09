from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from ultralytics import YOLO


PERSON_CLASS_ID = 0


@dataclass(frozen=True)
class Detection:
    """One detected person in pixel coordinates."""

    x1: int
    y1: int
    x2: int
    y2: int
    confidence: float


class PersonDetector:
    """Runs a COCO-pretrained YOLO model and keeps person detections only."""

    def __init__(
        self,
        weights: str | Path = "yolo11s.pt",
        confidence_threshold: float = 0.35,
        iou_threshold: float = 0.45,
        image_size: int = 640,
        device: str | None = None,
    ) -> None:
        if not 0.0 <= confidence_threshold <= 1.0:
            raise ValueError("confidence_threshold must be between 0 and 1.")
        if not 0.0 <= iou_threshold <= 1.0:
            raise ValueError("iou_threshold must be between 0 and 1.")
        if image_size <= 0:
            raise ValueError("image_size must be positive.")

        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.image_size = image_size
        self.device = device
        self.model = YOLO(str(weights))
        self.person_class_id = self._find_person_class_id()

    def detect(self, image: np.ndarray) -> list[Detection]:
        if image is None or image.size == 0:
            raise ValueError("image must be a non-empty OpenCV image.")

        result = self.model.predict(
            source=image,
            classes=[self.person_class_id],
            conf=self.confidence_threshold,
            iou=self.iou_threshold,
            imgsz=self.image_size,
            device=self.device,
            verbose=False,
        )[0]

        if result.boxes is None:
            return []

        boxes = result.boxes.xyxy.cpu().numpy()
        confidences = result.boxes.conf.cpu().numpy()
        return [
            Detection(
                x1=int(round(x1)),
                y1=int(round(y1)),
                x2=int(round(x2)),
                y2=int(round(y2)),
                confidence=float(confidence),
            )
            for (x1, y1, x2, y2), confidence in zip(boxes, confidences, strict=True)
        ]

    def _find_person_class_id(self) -> int:
        names: Any = self.model.names
        for class_id, class_name in names.items():
            if class_name == "person":
                return int(class_id)
        raise ValueError("The selected YOLO weights do not contain a 'person' class.")


def draw_detections(image: np.ndarray, detections: list[Detection]) -> np.ndarray:
    """Returns an annotated copy of the input image."""

    annotated = image.copy()
    height, width = annotated.shape[:2]

    for detection in detections:
        x1 = max(0, min(detection.x1, width - 1))
        y1 = max(0, min(detection.y1, height - 1))
        x2 = max(0, min(detection.x2, width - 1))
        y2 = max(0, min(detection.y2, height - 1))
        label = f"person {detection.confidence:.2f}"
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 180, 0), 2)
        cv2.putText(
            annotated,
            label,
            (x1, max(22, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 180, 0),
            2,
            cv2.LINE_AA,
        )

    banner = f"People in frame: {len(detections)}"
    cv2.rectangle(annotated, (0, 0), (300, 42), (20, 20, 20), -1)
    cv2.putText(
        annotated,
        banner,
        (12, 29),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return annotated
