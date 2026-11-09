from __future__ import annotations

import streamlit as st


def _open_page(page: str) -> None:
    st.session_state["navigation_page"] = page


def _feature_card(column, eyebrow: str, title: str, description: str, page: str, key: str, color: str) -> None:
    with column:
        with st.container(border=True):
            st.markdown(
                f"""
                <div style="border-radius:12px;padding:15px 16px 12px;background:linear-gradient(135deg,{color}18,#ffffff 82%);min-height:126px">
                  <div style="font-size:11px;letter-spacing:1.3px;text-transform:uppercase;color:{color};font-weight:750">{eyebrow}</div>
                  <div style="font-size:21px;font-weight:750;color:#172033;margin-top:8px">{title}</div>
                  <div style="font-size:13px;color:#536174;line-height:1.45;margin-top:6px">{description}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.button("Mở chức năng", key=key, on_click=_open_page, args=(page,), width="stretch")


def page_dashboard() -> None:
    st.markdown(
        """
        <div style="padding:30px 34px;border-radius:22px;background:linear-gradient(115deg,#111827 0%,#312e81 52%,#0e7490 100%);color:white;margin-bottom:20px">
          <div style="font-size:12px;letter-spacing:2px;text-transform:uppercase;color:#a5f3fc;font-weight:700">AI people counting & crowd analysis</div>
          <div style="font-size:38px;font-weight:800;line-height:1.15;margin-top:10px">CrowdVision</div>
          <div style="font-size:16px;color:#e0e7ff;max-width:760px;margin-top:10px">Phân tích con người bằng hai hướng AI độc lập: YOLO + ByteTrack cho detection/tracking, CSRNet cho mật độ đám đông.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    reports = st.session_state.get("run_reports", [])
    st.metric("Lượt phân tích phiên này", str(len(reports)))

    st.markdown("### Chọn chức năng")
    st.caption("Mở từ các thẻ bên dưới hoặc chuyển trang bằng menu ở thanh bên.")
    features = [
        ("PEOPLE COUNTING", "Image Analysis", "Phát hiện từng người trên ảnh bằng YOLO hoặc ước lượng density bằng CSRNet.", "Image Analysis", "#2563eb"),
        ("VIDEO", "Video Analysis", "Chạy YOLO theo frame, xem FPS và tải video có bounding box.", "Video Analysis", "#0891b2"),
        ("TRACKING", "Tracking & Line Crossing", "Theo dõi Track ID hoặc đếm lượt IN/OUT qua đường ảo.", "Tracking & Line Crossing", "#7c3aed"),
        ("CROWD DENSITY", "Crowd Density", "Xem count, density heatmap và vùng tập trung bằng CSRNet.", "Crowd Density", "#db2777"),
        ("EXPERIMENTS", "Model Comparison", "Chạy YOLO và CSRNet trên cùng ảnh, nhập ground truth nếu có.", "Model Comparison", "#d97706"),
        ("OCCUPANCY", "Zone Monitoring", "Thiết lập polygon ROI, sức chứa và theo dõi occupancy theo thời gian.", "Zone Monitoring", "#059669"),
        ("REPORTS", "Evaluation & Reports", "Xem và xuất báo cáo được tạo từ các lượt phân tích trong phiên này.", "Evaluation & Reports", "#475569"),
    ]
    for start in range(0, len(features), 3):
        columns = st.columns(3)
        for column, (eyebrow, title, description, page, color) in zip(columns, features[start:start + 3]):
            key = "open_" + page.lower().replace(" ", "_").replace("&", "and")
            _feature_card(column, eyebrow, title, description, page, key, color)

    if not reports:
        st.markdown("### Hai nhóm AI")
        left, right = st.columns(2)
        left.info("**People Counting:** YOLO phát hiện người; ByteTrack giữ ID và hỗ trợ line crossing.")
        right.info("**Crowd Density:** CSRNet ước lượng count và phân bố mật độ; không tạo box hay tracking ID.")

