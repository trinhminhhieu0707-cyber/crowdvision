from __future__ import annotations

import json
import logging
import tempfile
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
import torch

try:
    from streamlit_image_coordinates import streamlit_image_coordinates
except ModuleNotFoundError:
    streamlit_image_coordinates = None

from src.crowd_counting import CSRNetCounter
from src.counting import load_zones
from src.detection import PersonDetector
from src.ui.paths import ROOT, OUTPUTS, ZONE_CONFIG

LOG = logging.getLogger("crowdvision.app")
@st.cache_resource(show_spinner=False)
def get_detector(weights: str, confidence: float, iou: float, image_size: int, device: str) -> PersonDetector:
    return PersonDetector(weights, confidence, iou, image_size, device)

@st.cache_resource(show_spinner=False)
def get_csrnet(checkpoint: str, device: str) -> CSRNetCounter:
    return CSRNetCounter(checkpoint, device=device)

def decode_image(upload) -> np.ndarray:
    image = cv2.imdecode(np.frombuffer(upload.getvalue(), np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Không đọc được ảnh. Hãy chọn file PNG, JPG hoặc JPEG hợp lệ.")
    return image

def save_upload(upload, directory: Path, filename: str = "input") -> Path:
    suffix = Path(upload.name).suffix.lower() or ".mp4"
    destination = directory / f"{filename}{suffix}"
    destination.write_bytes(upload.getvalue())
    return destination

def first_video_frame(upload) -> tuple[np.ndarray, int, int, float, int]:
    with tempfile.TemporaryDirectory(prefix="crowdvision_preview_") as temporary:
        path = save_upload(upload, Path(temporary))
        capture = cv2.VideoCapture(str(path))
        if not capture.isOpened():
            raise ValueError("Không mở được video. Hãy chọn một video khác.")
        try:
            width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = float(capture.get(cv2.CAP_PROP_FPS) or 25.0)
            frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            ok, frame = capture.read()
            if not ok or width <= 0 or height <= 0:
                raise ValueError("Video không có frame hợp lệ.")
            return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), width, height, fps, frame_count
        finally:
            capture.release()

def present_error(error: Exception) -> None:
    LOG.exception("CrowdVision operation failed", exc_info=error)
    st.error(f"Không xử lý được đầu vào: {error}")

def draw_roi_preview(image_rgb: np.ndarray, zones: list[dict], pending: list[list[float]]) -> np.ndarray:
    image = cv2.cvtColor(image_rgb.copy(), cv2.COLOR_RGB2BGR)
    height, width = image.shape[:2]
    colors = [(0, 220, 255), (255, 170, 0), (255, 80, 180), (80, 230, 80), (200, 100, 255)]
    for index, zone in enumerate(zones):
        polygon = np.asarray(
            [(round(x * (width - 1)), round(y * (height - 1))) for x, y in zone["polygon"]], np.int32
        )
        color = colors[index % len(colors)]
        cv2.polylines(image, [polygon], True, color, 2, cv2.LINE_AA)
        x, y = map(int, polygon[0])
        cv2.putText(image, str(zone["name"]), (x, max(24, y - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2, cv2.LINE_AA)
    if pending:
        points = np.asarray(
            [(round(x * (width - 1)), round(y * (height - 1))) for x, y in pending], np.int32
        )
        for x, y in points:
            cv2.circle(image, (int(x), int(y)), 5, (0, 255, 0), -1)
        if len(points) > 1:
            cv2.polylines(image, [points], False, (0, 255, 0), 2, cv2.LINE_AA)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

def configure_sidebar() -> tuple[str, float, float, int, str, str]:
    st.sidebar.title("CrowdVision")
    st.session_state.setdefault("navigation_page", "Dashboard")
    page = st.sidebar.selectbox(
        "Khu vực",
        [
            "Dashboard",
            "Image Analysis",
            "Video Analysis",
            "Tracking & Line Crossing",
            "Crowd Density",
            "Model Comparison",
            "Zone Monitoring",
            "Evaluation & Reports",
        ],
        key="navigation_page",
    )
    st.sidebar.divider()
    weights = str(ROOT / "yolo11s.pt")
    confidence = st.sidebar.slider("Confidence", 0.05, 0.95, 0.35, 0.05)
    iou = st.sidebar.slider("IoU", 0.10, 0.90, 0.45, 0.05)
    image_size = st.sidebar.select_slider("Image size", options=[320, 480, 640, 800, 960], value=640)
    devices = ["CPU"] + (["GPU 0"] if torch.cuda.is_available() else [])
    selected_device = st.sidebar.selectbox("Thiết bị", devices)
    device = "cpu" if selected_device == "CPU" else "0"
    st.sidebar.caption("CSRNet là module density riêng, không cộng kết quả với YOLO/ByteTrack.")
    return page, confidence, iou, image_size, device, weights


def record_run_report(module: str, input_name: str, metrics: dict) -> None:
    reports = st.session_state.setdefault("run_reports", [])
    reports.insert(0, {
        "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
        "module": module,
        "input": input_name,
        "metrics": metrics,
    })

def load_zones_from_dict(config: dict):
    with tempfile.NamedTemporaryFile("w", suffix=".json", encoding="utf-8", delete=False) as handle:
        temporary = Path(handle.name)
        json.dump(config, handle)
    try:
        return load_zones(temporary)
    finally:
        temporary.unlink(missing_ok=True)

