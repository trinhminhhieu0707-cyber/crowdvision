from __future__ import annotations

import cv2
import pandas as pd
import streamlit as st

from src.crowd_counting.analysis import overlay_density
from src.detection import draw_detections
from src.ui.common import decode_image, get_csrnet, get_detector, ROOT, present_error, record_run_report

def page_model_comparison(device: str, weights: str, confidence: float, iou: float, image_size: int) -> None:
    st.title("Model Comparison")
    st.caption("YOLO detection count và CSRNet density count là hai đầu ra riêng; không cộng chúng với nhau.")
    upload = st.file_uploader("Tải ảnh lên", type=["jpg", "jpeg", "png"], key="comparison_upload")
    ground_truth = st.number_input("Ground truth (tùy chọn)", min_value=0, step=1, value=0)
    has_ground_truth = st.checkbox("Tôi đã nhập ground truth")
    if upload is None:
        st.info("Ảnh test Ground Truth có ích để tính sai số cho chính ảnh này.")
        return
    try:
        image = decode_image(upload)
        if st.button("So sánh hai mô hình", type="primary"):
            with st.spinner("Đang chạy YOLO và CSRNet..."):
                detector = get_detector(weights, confidence, iou, image_size, device)
                detections = detector.detect(image)
                checkpoint = ROOT / "models" / "csrnet" / "PartAmodel_best.pth.tar"
                counter = get_csrnet(str(checkpoint), device)
                prediction = counter.predict_rgb(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                annotated = draw_detections(image, detections)
                overlay = overlay_density(image, prediction.density_map)
                truth = int(ground_truth) if has_ground_truth else None
                st.session_state["comparison_result"] = {
                    "yolo": len(detections), "csrnet": prediction.count, "truth": truth,
                    "yolo_image": cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
                    "csrnet_image": cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB),
                }
                metrics = {"yolo_count": len(detections), "csrnet_estimated_count": round(float(prediction.count), 2)}
                if truth is not None:
                    metrics.update({"ground_truth": truth, "yolo_absolute_error": abs(len(detections) - truth),
                                    "csrnet_absolute_error": round(abs(float(prediction.count) - truth), 2),
                                    "yolo_signed_error": len(detections) - truth,
                                    "csrnet_signed_error": round(float(prediction.count) - truth, 2)})
                record_run_report("Model Comparison", upload.name, metrics)
    except (ValueError, OSError, RuntimeError) as error:
        present_error(error)
    result = st.session_state.get("comparison_result")
    if result:
        columns = st.columns(3 if result["truth"] is not None else 2)
        columns[0].image(result["yolo_image"], caption=f"YOLO | count={result['yolo']}", width="stretch")
        columns[1].image(result["csrnet_image"], caption=f"CSRNet | estimate={result['csrnet']:.1f}", width="stretch")
        if result["truth"] is not None:
            columns[2].metric("Ground truth", result["truth"])
            st.dataframe(pd.DataFrame([
                {"Model": "YOLO", "Prediction": result["yolo"], "Absolute error": abs(result["yolo"] - result["truth"])},
                {"Model": "CSRNet", "Prediction": round(result["csrnet"], 1), "Absolute error": round(abs(result["csrnet"] - result["truth"]), 1)},
            ]), hide_index=True, width="stretch")

