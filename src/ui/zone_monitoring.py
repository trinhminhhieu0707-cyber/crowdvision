from __future__ import annotations

import json
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st

from src.counting import load_zones, process_zone_video
from src.tracking import PersonTracker
from src.ui.common import ROOT, ZONE_CONFIG, draw_roi_preview, first_video_frame, load_zones_from_dict, present_error, record_run_report, save_upload, streamlit_image_coordinates

def page_zone_monitoring(device: str, weights: str, confidence: float, iou: float, image_size: int) -> None:
    st.title("Zone Monitoring")
    st.caption("Mỗi Track ID chỉ thuộc tối đa một ROI trong mỗi frame. Occupancy theo vùng không phải tổng người qua video.")
    upload = st.file_uploader("Tải video camera lên", type=["mp4", "mov", "avi", "mkv"], key="zone_upload")
    config_path = ZONE_CONFIG
    if "zone_config" not in st.session_state:
        path = config_path if config_path.is_file() else ROOT / "configs" / "zones.example.json"
        st.session_state["zone_config"] = json.loads(path.read_text(encoding="utf-8"))
    config = st.session_state["zone_config"]
    zones = config.get("zones", [])
    preview = None
    if upload is not None:
        try:
            preview, width, height, _fps, frame_count = first_video_frame(upload)
            st.caption(f"Video: {width}×{height} | {frame_count} frames")
        except (ValueError, OSError) as error:
            st.error(str(error))
            return

    if zones:
        zone_options = [str(zone["name"]) for zone in zones]
    else:
        zone_options = []
    selected = st.selectbox("Vùng đang chỉnh sửa", ["+ Thêm vùng mới", *zone_options], key="zone_editor_select")
    existing = next((zone for zone in zones if zone["name"] == selected), None)
    target_key = selected
    if st.session_state.get("active_zone_editor") != target_key:
        st.session_state["active_zone_editor"] = target_key
        st.session_state["roi_pending_points"] = [list(point) for point in existing["polygon"]] if existing else []
        st.session_state["roi_draft_name"] = existing["name"] if existing else ""
        st.session_state["roi_draft_capacity"] = int(existing["capacity"]) if existing else 10

    name_col, capacity_col = st.columns(2)
    name_col.text_input("Tên vùng", key="roi_draft_name")
    capacity_col.number_input("Sức chứa tối đa", min_value=1, step=1, key="roi_draft_capacity")
    if preview is not None:
        image = draw_roi_preview(preview, zones, st.session_state["roi_pending_points"])
        preview_width = min(width, 900)
        preview_height = max(1, round(height * preview_width / width))
        clickable_image = cv2.resize(image, (preview_width, preview_height))
        if streamlit_image_coordinates is None:
            st.warning("Thiếu gói streamlit-image-coordinates. Cài bằng `pip install -r requirements.txt` để bấm chọn đỉnh; bạn vẫn có thể sửa JSON ROI bên dưới.")
            config_text = st.text_area("ROI configuration JSON", json.dumps(config, indent=2), height=240, key="zone_json_fallback")
            if st.button("Áp dụng JSON ROI"):
                try:
                    edited_config = json.loads(config_text)
                    load_zones_from_dict(edited_config)
                    st.session_state["zone_config"] = edited_config
                    st.success("Đã áp dụng cấu hình ROI.")
                    st.rerun()
                except (ValueError, TypeError, KeyError) as error:
                    st.error(f"Cấu hình ROI chưa hợp lệ: {error}")
        else:
            click = streamlit_image_coordinates(
                clickable_image,
                width=preview_width,
                key=f"roi_canvas_{upload.name}_{upload.size}",
                cursor="crosshair",
            )
            if click and click.get("unix_time") != st.session_state.get("roi_last_click"):
                x = float(click["x"]) / max(preview_width - 1, 1)
                y = float(click["y"]) / max(preview_height - 1, 1)
                st.session_state["roi_pending_points"].append([round(float(np.clip(x, 0, 1)), 6), round(float(np.clip(y, 0, 1)), 6)])
                st.session_state["roi_last_click"] = click.get("unix_time")
                st.rerun()
            st.caption("Bấm lần lượt các đỉnh polygon trên ảnh. Với zone hiện có, các điểm đã lưu được nạp để chỉnh sửa; dùng nút xóa điểm nếu muốn vẽ lại.")
    else:
        st.info("Tải video lên để xem frame và bấm tạo polygon ROI.")

    points = st.session_state["roi_pending_points"]
    st.write(f"Số đỉnh đang có: {len(points)}")
    undo_col, clear_col, save_poly_col = st.columns([1, 1, 1.15], vertical_alignment="bottom")
    if undo_col.button("Hoàn tác điểm cuối", disabled=not points):
        points.pop()
        st.rerun()
    if clear_col.button("Xóa các điểm để vẽ lại", disabled=not points):
        st.session_state["roi_pending_points"] = []
        st.rerun()
    if save_poly_col.button("Lưu polygon vào cấu hình", type="primary"):
        name = st.session_state["roi_draft_name"].strip()
        capacity = int(st.session_state["roi_draft_capacity"])
        if not name or len(points) < 3:
            st.error("Nhập tên vùng và tạo polygon có ít nhất 3 đỉnh.")
        elif any(zone["name"].casefold() == name.casefold() and zone["name"] != selected for zone in zones):
            st.error("Tên vùng này đã tồn tại.")
        else:
            zone_data = {"name": name, "capacity": capacity, "polygon": [list(point) for point in points]}
            if existing:
                zones[zones.index(existing)] = zone_data
            else:
                zones.append(zone_data)
            config["zones"] = zones
            st.session_state["zone_config"] = config
            st.session_state["active_zone_editor"] = None
            st.success(f"Đã cập nhật polygon {name}. Nhấn 'Lưu file cấu hình' để ghi xuống máy.")
            st.rerun()

    delete_col, persist_col, download_col = st.columns([1, 1, 1.2], vertical_alignment="bottom")
    if existing and delete_col.button("Xóa vùng đang chọn"):
        config["zones"] = [zone for zone in zones if zone["name"] != selected]
        st.session_state["zone_config"] = config
        st.session_state["active_zone_editor"] = None
        st.rerun()
    if persist_col.button("Lưu file cấu hình"):
        try:
            parsed = load_zones_from_dict(config)
            config_path.parent.mkdir(parents=True, exist_ok=True)
            config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
            st.success(f"Đã lưu {len(parsed)} vùng vào configs/zones.json.")
        except (ValueError, TypeError, KeyError) as error:
            st.error(f"Cấu hình ROI chưa hợp lệ: {error}")
    download_col.download_button("Tải cấu hình ROI JSON", json.dumps(config, indent=2), "zones.json", "application/json", width="stretch")

    if preview is not None:
        st.subheader("Xem trước các ROI")
        st.image(draw_roi_preview(preview, zones, st.session_state["roi_pending_points"]), width="stretch")

    if upload is not None and zones:
        if st.button("Chạy Zone Monitoring", type="primary"):
            try:
                parsed = load_zones_from_dict(config)
                tracker = PersonTracker(weights, confidence, iou, image_size, device)
                with st.spinner("Đang tracking và tính occupancy theo vùng..."):
                    with tempfile.TemporaryDirectory(prefix="crowdvision_zones_") as temporary:
                        folder = Path(temporary)
                        source = save_upload(upload, folder)
                        destination = folder / "zone_monitoring.mp4"
                        summary = process_zone_video(tracker, source, destination, parsed)
                        st.session_state["zone_result"] = {
                            "video": destination.read_bytes(),
                            "occupancy": summary.csv_path.read_bytes(),
                            "alerts": summary.alerts_path.read_bytes(),
                            "frames": summary.processed_frames,
                            "fps": summary.average_processing_fps,
                            "final": summary.final_occupancy,
                        }
                        occupancy = pd.read_csv(pd.io.common.BytesIO(summary.csv_path.read_bytes()))
                        zone_summary = occupancy.groupby("zone").agg(
                            peak_occupancy=("occupancy", "max"),
                            capacity=("capacity", "last"),
                            peak_occupancy_rate_percent=("occupancy_rate_percent", "max"),
                        ).to_dict(orient="index")
                        record_run_report("Zone Monitoring", upload.name, {
                            "frames": summary.processed_frames,
                            "processing_fps": round(summary.average_processing_fps, 2),
                            "peak_by_zone": zone_summary,
                            "capacity_alert_events": len(pd.read_csv(pd.io.common.BytesIO(summary.alerts_path.read_bytes()))),
                        })
            except (ValueError, OSError, RuntimeError) as error:
                present_error(error)
    result = st.session_state.get("zone_result")
    if result:
        st.subheader("Kết quả giám sát")
        occupancy = pd.read_csv(pd.io.common.BytesIO(result["occupancy"]))
        alerts = pd.read_csv(pd.io.common.BytesIO(result["alerts"]))
        summary_cols = st.columns(max(1, len(zones)))
        latest = occupancy.sort_values("frame").groupby("zone", as_index=False).tail(1)
        for column, (_, row) in zip(summary_cols, latest.iterrows()):
            with column:
                st.metric(str(row["zone"]), f"{int(row['occupancy'])}/{int(row['capacity'])}", f"{row['occupancy_rate_percent']:.0f}%")
                st.caption(str(row["status"]).replace("_", " ").title())
        st.video(result["video"])
        if not occupancy.empty:
            chart = occupancy.pivot_table(index="timestamp_seconds", columns="zone", values="occupancy", aggfunc="last")
            st.line_chart(chart)
            st.dataframe(latest[["zone", "occupancy", "capacity", "occupancy_rate_percent", "status"]], hide_index=True, width="stretch")
        if alerts.empty:
            st.success("Không có vùng nào vượt sức chứa trong video.")
        else:
            st.warning("Có cảnh báo vượt sức chứa.")
            st.dataframe(alerts, hide_index=True, width="stretch")
        a, b, c = st.columns(3)
        a.download_button("Tải video kết quả", result["video"], "zone_monitoring.mp4", "video/mp4")
        b.download_button("Tải occupancy CSV", result["occupancy"], "zone_monitoring_occupancy.csv", "text/csv")
        c.download_button("Tải alerts CSV", result["alerts"], "zone_monitoring_alerts.csv", "text/csv")

