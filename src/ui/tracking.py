from __future__ import annotations

import tempfile
from pathlib import Path

import cv2
import streamlit as st

from src.counting.flow_video import process_flow_video
from src.tracking import PersonTracker, process_tracking_video
from src.ui.common import first_video_frame, present_error, record_run_report, save_upload

def page_tracking(device: str, weights: str, confidence: float, iou: float, image_size: int) -> None:
    st.title("Tracking & Line Crossing")
    mode = st.radio("Chức năng", ["Tracking IDs", "Line Crossing IN/OUT"], horizontal=True)
    upload = st.file_uploader("Tải video lên", type=["mp4", "mov", "avi", "mkv"], key="tracking_upload")
    if upload is None:
        st.info("Chọn video để bắt đầu.")
        return
    orientation, line_x, line_y = None, None, None
    try:
        preview, width, height, _fps, _frames = first_video_frame(upload)
    except (ValueError, OSError) as error:
        st.error(str(error))
        return
    if mode.startswith("Line"):
        orientation = st.radio("Hướng đường đếm", ["Dọc", "Ngang"], horizontal=True)
        if orientation == "Dọc":
            line_x = st.slider("Vị trí line theo X", 0, width - 1, width // 2)
            preview_bgr = cv2.cvtColor(preview, cv2.COLOR_RGB2BGR)
            cv2.line(preview_bgr, (line_x, 0), (line_x, height - 1), (0, 0, 255), 3)
        else:
            line_y = st.slider("Vị trí line theo Y", 0, height - 1, height // 2)
            preview_bgr = cv2.cvtColor(preview, cv2.COLOR_RGB2BGR)
            cv2.line(preview_bgr, (0, line_y), (width - 1, line_y), (0, 0, 255), 3)
        st.image(cv2.cvtColor(preview_bgr, cv2.COLOR_BGR2RGB), caption="Xem trước vị trí line", width="stretch")
        compare_truth = st.checkbox("Tôi có ground truth IN/OUT để đối chiếu")
        if compare_truth:
            truth_in_col, truth_out_col = st.columns(2)
            truth_in = int(truth_in_col.number_input("Ground truth IN", min_value=0, step=1, key="line_truth_in"))
            truth_out = int(truth_out_col.number_input("Ground truth OUT", min_value=0, step=1, key="line_truth_out"))
        else:
            truth_in = truth_out = None
    if st.button("Chạy tracking", type="primary"):
        try:
            tracker = PersonTracker(weights, confidence, iou, image_size, device)
            with st.spinner("Đang tracking video..."):
                with tempfile.TemporaryDirectory(prefix="crowdvision_tracking_") as temporary:
                    folder = Path(temporary)
                    source = save_upload(upload, folder)
                    destination = folder / "tracking_result.mp4"
                    if mode == "Tracking IDs":
                        summary = process_tracking_video(tracker, source, destination)
                        result = {"video": destination.read_bytes(), "frames": summary.processed_frames,
                                  "fps": summary.average_processing_fps, "mode": "tracking"}
                    else:
                        summary = process_flow_video(tracker, source, destination, line_y=line_y, line_x=line_x)
                        result = {"video": destination.read_bytes(), "frames": summary.processed_frames,
                                  "fps": summary.average_processing_fps, "mode": "line",
                                  "in": summary.counts.entered, "out": summary.counts.exited,
                                  "total": summary.counts.total}
                    st.session_state["tracking_result"] = result
                    metrics = {"frames": result["frames"], "processing_fps": round(result["fps"], 2)}
                    if result["mode"] == "line":
                        metrics.update({"IN": result["in"], "OUT": result["out"], "total_crossings": result["total"],
                                        "line_orientation": orientation, "line_position_px": line_x if line_x is not None else line_y})
                        if truth_in is not None and truth_out is not None:
                            metrics.update({
                                "ground_truth_IN": truth_in,
                                "ground_truth_OUT": truth_out,
                                "IN_absolute_error": abs(result["in"] - truth_in),
                                "OUT_absolute_error": abs(result["out"] - truth_out),
                                "total_absolute_error": abs(result["total"] - truth_in - truth_out),
                            })
                    record_run_report("Line Crossing" if result["mode"] == "line" else "Tracking", upload.name, metrics)
        except (ValueError, OSError, RuntimeError) as error:
            present_error(error)
    result = st.session_state.get("tracking_result")
    if result:
        st.video(result["video"])
        if result["mode"] == "line":
            cols = st.columns(3)
            cols[0].metric("IN", result["in"])
            cols[1].metric("OUT", result["out"])
            cols[2].metric("TOTAL lượt crossing", result["total"])
        else:
            a, b = st.columns(2)
            a.metric("Frames", result["frames"])
            b.metric("Processing FPS", f"{result['fps']:.1f}")
        st.download_button("Tải video kết quả", result["video"], "tracking_result.mp4", "video/mp4")

