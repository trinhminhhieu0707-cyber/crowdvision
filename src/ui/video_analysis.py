from __future__ import annotations

import tempfile
from pathlib import Path

import streamlit as st

from src.detection import process_video
from src.ui.common import get_detector, present_error, record_run_report, save_upload

def page_video_analysis(device: str, weights: str, confidence: float, iou: float, image_size: int) -> None:
    st.title("Video Analysis")
    upload = st.file_uploader("Tải video lên", type=["mp4", "mov", "avi", "mkv"], key="video_analysis_upload")
    if upload is None:
        st.info("Chọn video để chạy YOLO trên từng frame.")
        return
    if st.button("Phát hiện người trong video", type="primary"):
        try:
            with st.spinner("Đang xử lý video. Thời gian phụ thuộc độ dài video và thiết bị..."):
                detector = get_detector(weights, confidence, iou, image_size, device)
                with tempfile.TemporaryDirectory(prefix="crowdvision_detection_") as temporary:
                    folder = Path(temporary)
                    source = save_upload(upload, folder)
                    destination = folder / "people_detected.mp4"
                    summary = process_video(detector, source, destination)
                    result = {
                        "video": destination.read_bytes(), "frames": summary.processed_frames,
                        "fps": summary.average_processing_fps, "source_fps": summary.source_fps,
                    }
                    st.session_state["video_detection"] = result
                    record_run_report("Video Detection / YOLO", upload.name, {
                        "frames": result["frames"], "processing_fps": round(result["fps"], 2),
                        "source_fps": round(result["source_fps"], 2),
                    })
        except (ValueError, OSError, RuntimeError) as error:
            present_error(error)
    result = st.session_state.get("video_detection")
    if result:
        st.video(result["video"])
        a, b, c = st.columns(3)
        a.metric("Frames", result["frames"])
        b.metric("Processing FPS", f"{result['fps']:.1f}")
        c.metric("Source FPS", f"{result['source_fps']:.1f}")
        st.download_button("Tải video kết quả", result["video"], "people_detected.mp4", "video/mp4")

