from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Protocol

import cv2
import numpy as np
from ultralytics import YOLO

from src.video_io import open_video, open_writer


@dataclass(frozen=True)
class TrackedPerson:
    track_id: int
    x1: int
    y1: int
    x2: int
    y2: int
    confidence: float


@dataclass(frozen=True)
class TrackingVideoSummary:
    input_path: Path
    output_path: Path
    processed_frames: int
    source_fps: float
    average_processing_fps: float


## [SỬA 8a | tracker.py | class mới Tracker] Protocol mô tả "thứ gì có track() và reset()".
## Giúp các hàm process_* nhận cả PersonTracker thật lẫn tracker giả khi test, và hết cảnh báo
## "Expected type 'PersonTracker', got '_ScriptedTracker'" trong PyCharm.
class Tracker(Protocol):
    def track(self, frame: np.ndarray) -> list[TrackedPerson]: ...

    def reset(self) -> None: ...


class PersonTracker:
    """Associates person detections across consecutive video frames with ByteTrack."""

    def __init__(
        self,
        weights: str | Path = "yolo11s.pt",
        confidence_threshold: float = 0.35,
        iou_threshold: float = 0.45,
        image_size: int = 640,
        device: str | None = None,
        low_confidence_threshold: float = 0.1,
    ) -> None:
        if not 0.0 <= confidence_threshold <= 1.0:
            raise ValueError("confidence_threshold must be between 0 and 1.")
        if not 0.0 <= iou_threshold <= 1.0:
            raise ValueError("iou_threshold must be between 0 and 1.")
        if image_size <= 0:
            raise ValueError("image_size must be positive.")
        ## [SỬA 1b | tracker.py | __init__] Validate tham số mới.
        if not 0.0 <= low_confidence_threshold <= 1.0:
            raise ValueError("low_confidence_threshold must be between 0 and 1.")

        self.confidence_threshold = confidence_threshold
        ## [SỬA 1c | tracker.py | __init__] Lưu 2 ngưỡng riêng biệt:
        ##   - confidence_threshold : ngưỡng HIỂN THỊ/ĐẾM (giữ nguyên ý nghĩa cũ, mặc định 0.35)
        ##   - _tracker_conf        : ngưỡng đưa vào YOLO -> ByteTrack (thấp hơn, mặc định 0.1)
        ## min(...) đảm bảo ngưỡng cho tracker không bao giờ cao hơn ngưỡng hiển thị.
        self.low_confidence_threshold = low_confidence_threshold
        self._tracker_conf = min(low_confidence_threshold, confidence_threshold)
        self.iou_threshold = iou_threshold
        self.image_size = image_size
        self.device = device
        self.model = YOLO(str(weights))
        self.person_class_id = self._find_person_class_id()

    def track(self, frame: np.ndarray) -> list[TrackedPerson]:
        if frame is None or frame.size == 0:
            raise ValueError("frame must be a non-empty OpenCV image.")

        result = self.model.track(
            source=frame,
            persist=True,
            tracker="bytetrack.yaml",
            classes=[self.person_class_id],
            ## [SỬA 1d | tracker.py | track()] TRƯỚC: conf=self.confidence_threshold (0.35).
            ## Lọc 0.35 ngay tại YOLO khiến ByteTrack không còn thấy các box điểm thấp (0.1-0.35),
            ## trong khi bước "ghép lần 2" của ByteTrack cần chính các box đó để giữ ID khi bị che.
            ## SAU: đưa ngưỡng thấp vào tracker, rồi lọc lại để hiển thị ở bước 1e bên dưới.
            conf=self._tracker_conf,
            iou=self.iou_threshold,
            imgsz=self.image_size,
            device=self.device,
            verbose=False,
        )[0]

        if result.boxes is None or result.boxes.id is None:
            return []

        boxes = result.boxes.xyxy.cpu().numpy()
        confidences = result.boxes.conf.cpu().numpy()
        track_ids = result.boxes.id.int().cpu().numpy()
        return [
            TrackedPerson(
                track_id=int(track_id),
                x1=int(round(x1)),
                y1=int(round(y1)),
                x2=int(round(x2)),
                y2=int(round(y2)),
                confidence=float(confidence),
            )
            for (x1, y1, x2, y2), confidence, track_id in zip(
                boxes, confidences, track_ids, strict=True
            )
            ## [SỬA 1e | tracker.py | track()] Lọc đầu ra theo ngưỡng hiển thị cũ (0.35):
            ## video/đếm người trông giống trước, nhưng ID bên trong tracker ổn định hơn.
            ## (Box điểm thấp vẫn giúp giữ ID nhưng không bị vẽ/đếm.)
            if float(confidence) >= self.confidence_threshold
        ]

    ## [SỬA 2a | tracker.py | method mới reset()] Xoá trạng thái ByteTrack giữa 2 video.
    ## Vì track(persist=True), nếu dùng lại 1 PersonTracker cho video khác thì ID/track cũ bị mang sang.
    ## Đặt predictor=None để Ultralytics tạo tracker mới ở lần track() kế tiếp
    ## (cách này không phụ thuộc cấu trúc nội bộ của từng phiên bản ultralytics).
    def reset(self) -> None:
        self.model.predictor = None

    def _find_person_class_id(self) -> int:
        names: Any = self.model.names
        for class_id, class_name in names.items():
            if class_name == "person":
                return int(class_id)
        raise ValueError("The selected YOLO weights do not contain a 'person' class.")


## [SỬA 6c | tracker.py | process_tracking_video] Viết lại bằng open_video/open_writer (SỬA 6a):
## bỏ ~30 dòng mở/kiểm tra/release lặp lại; logic xử lý từng frame GIỮ NGUYÊN.
def process_tracking_video(
    tracker: Tracker,
    input_path: str | Path,
    output_path: str | Path,
) -> TrackingVideoSummary:
    """Tracks people in a video and writes annotated frames to an MP4 file."""

    input_path = Path(input_path)
    output_path = Path(output_path)

    with open_video(input_path) as (capture, info), open_writer(
        output_path, info.fps, (info.width, info.height)
    ) as writer:
        ## [SỬA 2b | tracker.py | process_tracking_video] Reset tracker trước mỗi video mới.
        tracker.reset()

        processed_frames = 0
        total_processing_seconds = 0.0
        while True:
            ok, frame = capture.read()
            if not ok:
                break

            frame_started_at = perf_counter()
            tracks = tracker.track(frame)
            annotated = draw_tracks(frame, tracks)
            elapsed_seconds = perf_counter() - frame_started_at
            total_processing_seconds += elapsed_seconds
            processed_frames += 1

            _draw_processing_fps(annotated, elapsed_seconds)
            writer.write(annotated)

    average_processing_fps = (
        processed_frames / total_processing_seconds if total_processing_seconds > 0 else 0.0
    )
    return TrackingVideoSummary(
        input_path=input_path,
        output_path=output_path,
        processed_frames=processed_frames,
        source_fps=info.fps,
        average_processing_fps=average_processing_fps,
    )


def draw_tracks(frame: np.ndarray, tracks: list[TrackedPerson]) -> np.ndarray:
    """Returns a copy of the frame with active ByteTrack IDs drawn on it."""

    annotated = frame.copy()
    height, width = annotated.shape[:2]
    for track in tracks:
        x1 = max(0, min(track.x1, width - 1))
        y1 = max(0, min(track.y1, height - 1))
        x2 = max(0, min(track.x2, width - 1))
        y2 = max(0, min(track.y2, height - 1))
        label = f"ID {track.track_id} | person {track.confidence:.2f}"
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (255, 160, 0), 2)
        cv2.putText(
            annotated,
            label,
            (x1, max(22, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 160, 0),
            2,
            cv2.LINE_AA,
        )

    cv2.rectangle(annotated, (0, 0), (340, 42), (20, 20, 20), -1)
    cv2.putText(
        annotated,
        f"Tracked people in frame: {len(tracks)}",
        (12, 29),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return annotated


def _draw_processing_fps(image: np.ndarray, elapsed_seconds: float) -> None:
    processing_fps = 1.0 / elapsed_seconds if elapsed_seconds > 0 else 0.0
    height, width = image.shape[:2]
    cv2.rectangle(image, (0, 44), (min(width, 300), 82), (20, 20, 20), -1)
    cv2.putText(
        image,
        f"Processing FPS: {processing_fps:.1f}",
        (12, min(height - 12, 70)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
