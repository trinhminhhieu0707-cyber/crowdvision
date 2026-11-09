from __future__ import annotations

import cv2
import numpy as np
import pandas as pd
import streamlit as st

from src.crowd_counting.analysis import density_heatmap, find_dense_regions, overlay_density
from src.detection import draw_detections
from src.ui.common import decode_image, get_csrnet, get_detector, ROOT, present_error, record_run_report

def page_image_analysis(device: str, weights: str, confidence: float, iou: float, image_size: int) -> None:
    st.title("Image Analysis")
    upload = st.file_uploader("Tải ảnh lên", type=["jpg", "jpeg", "png"], key="image_analysis_upload")
    method = st.radio("Mô hình", ["YOLO — phát hiện từng người", "CSRNet — ước lượng mật độ"], horizontal=True)
    if upload is None:
        st.info("Chọn một ảnh để bắt đầu.")
        return
    try:
        image = decode_image(upload)
        st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), caption=upload.name, width="stretch")
        if st.button("Phân tích ảnh", type="primary"):
            with st.spinner("Đang phân tích ảnh..."):
                if method.startswith("YOLO"):
                    detector = get_detector(weights, confidence, iou, image_size, device)
                    detections = detector.detect(image)
                    result = draw_detections(image, detections)
                    st.session_state["image_result"] = {
                        "kind": "yolo", "image": cv2.cvtColor(result, cv2.COLOR_BGR2RGB), "count": len(detections)
                    }
                    record_run_report("Image / YOLO", upload.name, {
                        "predicted_count": len(detections), "confidence_threshold": confidence,
                        "iou_threshold": iou, "image_size": image_size,
                    })
                else:
                    checkpoint = ROOT / "models" / "csrnet" / "PartAmodel_best.pth.tar"
                    counter = get_csrnet(str(checkpoint), device)
                    prediction = counter.predict_rgb(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                    overlay = overlay_density(image, prediction.density_map)
                    heat = density_heatmap(prediction.density_map, image.shape[:2])
                    regions = find_dense_regions(prediction.density_map, image.shape[:2], top_k=3)
                    st.session_state["image_result"] = {
                        "kind": "csrnet", "overlay": cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB),
                        "heat": cv2.cvtColor(heat, cv2.COLOR_BGR2RGB), "count": prediction.count,
                        "regions": [region.to_dict() for region in regions],
                    }
                    record_run_report("Image / CSRNet", upload.name, {
                        "estimated_count": round(float(prediction.count), 2),
                        "dense_regions": len(regions), "device": str(counter.device),
                    })
    except (ValueError, OSError, RuntimeError) as error:
        present_error(error)

    result = st.session_state.get("image_result")
    if result:
        st.metric("Số người dự đoán", f"{result['count']:.1f}" if result["kind"] == "csrnet" else result["count"])
        if result["kind"] == "yolo":
            st.image(result["image"], caption="YOLO detections", width="stretch")
            st.download_button("Tải ảnh kết quả", cv2.imencode(".jpg", cv2.cvtColor(result["image"], cv2.COLOR_RGB2BGR))[1].tobytes(), "people_detected.jpg", "image/jpeg")
        else:
            a, b = st.columns(2)
            a.image(result["overlay"], caption="Density overlay", width="stretch")
            b.image(result["heat"], caption="Density heatmap", width="stretch")
            st.dataframe(pd.DataFrame(result["regions"]), hide_index=True, width="stretch")

