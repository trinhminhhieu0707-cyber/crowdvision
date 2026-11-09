from __future__ import annotations

import tempfile
from pathlib import Path

import cv2
import pandas as pd
import streamlit as st

from src.crowd_counting.analysis import density_heatmap, find_dense_regions, overlay_density
from src.crowd_counting.video import analyze_video_density
from src.ui.common import decode_image, get_csrnet, present_error, record_run_report, save_upload, ROOT

def page_crowd_density(device: str) -> None:
    st.title("Crowd Density — CSRNet")
    mode = st.radio("Đầu vào", ["Ảnh", "Video"], horizontal=True)
    types = ["jpg", "jpeg", "png"] if mode == "Ảnh" else ["mp4", "mov", "avi", "mkv"]
    upload = st.file_uploader("Tải đầu vào lên", type=types, key=f"density_{mode}")
    if upload is None:
        st.info("CSRNet ước lượng count và phân bố density, không tracking ID hoặc đếm IN/OUT.")
        return
    checkpoint = ROOT / "models" / "csrnet" / "PartAmodel_best.pth.tar"
    if mode == "Ảnh":
        try:
            image = decode_image(upload)
            if st.button("Phân tích density", type="primary"):
                with st.spinner("Đang chạy CSRNet..."):
                    counter = get_csrnet(str(checkpoint), device)
                    prediction = counter.predict_rgb(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                    overlay = overlay_density(image, prediction.density_map)
                    heat = density_heatmap(prediction.density_map, image.shape[:2])
                    regions = find_dense_regions(prediction.density_map, image.shape[:2], top_k=3)
                    st.session_state["density_image_result"] = {
                        "count": prediction.count, "overlay": cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB),
                        "heat": cv2.cvtColor(heat, cv2.COLOR_BGR2RGB),
                        "regions": [region.to_dict() for region in regions],
                    }
                    record_run_report("Crowd Density / Image", upload.name, {
                        "estimated_count": round(float(prediction.count), 2),
                        "dense_regions": len(regions), "device": str(counter.device),
                    })
        except (ValueError, OSError, RuntimeError) as error:
            present_error(error)
        result = st.session_state.get("density_image_result")
        if result:
            st.metric("CSRNet estimated count", f"{result['count']:.1f}")
            left, right = st.columns(2)
            left.image(result["overlay"], caption="Heatmap overlay", width="stretch")
            right.image(result["heat"], caption="Density heatmap", width="stretch")
            st.dataframe(pd.DataFrame(result["regions"]), hide_index=True, width="stretch")
    else:
        sample_every = st.slider("Phân tích mỗi N frame", 1, 60, 15)
        if st.button("Phân tích density video", type="primary"):
            try:
                counter = get_csrnet(str(checkpoint), device)
                with st.spinner("CSRNet đang phân tích các frame đã chọn..."):
                    with tempfile.TemporaryDirectory(prefix="crowdvision_density_") as temporary:
                        folder = Path(temporary)
                        source = save_upload(upload, folder)
                        output_dir = folder / "density_output"
                        summary = analyze_video_density(source, output_dir, counter, sample_every=sample_every)
                        result = {
                            "video": Path(summary["video"]).read_bytes(),
                            "trend": Path(summary["trend_chart"]).read_bytes(),
                            "csv": Path(summary["counts_csv"]).read_bytes(),
                            "frames": summary["frames"], "sampled": summary["sampled_frames"],
                        }
                        st.session_state["density_video_result"] = result
                        record_run_report("Crowd Density / Video", upload.name, {
                            "frames": result["frames"], "sampled_frames": result["sampled"],
                            "sample_every": sample_every,
                        })
            except (ValueError, OSError, RuntimeError) as error:
                present_error(error)
        result = st.session_state.get("density_video_result")
        if result:
            st.video(result["video"])
            st.image(result["trend"], caption="Estimated count over time", width="stretch")
            st.caption(f"Đã xử lý {result['sampled']} frame mẫu trên tổng {result['frames']} frame.")
            st.download_button("Tải CSV density", result["csv"], "density_counts.csv", "text/csv")
            st.download_button("Tải video density", result["video"], "density_video.mp4", "video/mp4")

