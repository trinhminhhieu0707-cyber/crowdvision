from __future__ import annotations

import json
import math

import pandas as pd
import streamlit as st


def page_evaluation_reports() -> None:
    st.title("Báo cáo phiên phân tích")
    st.caption("Báo cáo được tạo trực tiếp từ ảnh/video bạn vừa xử lý trong phiên Streamlit này.")
    reports = st.session_state.get("run_reports", [])
    if not reports:
        st.info("Chưa có báo cáo trong phiên này. Chạy một chức năng như Image Analysis, Video Analysis hoặc Zone Monitoring trước.")
        return

    flat_rows = []
    for report in reports:
        flat_rows.append({
            "Thời gian": report["timestamp"],
            "Chức năng": report["module"],
            "Đầu vào": report["input"],
            **report["metrics"],
        })
    table = pd.json_normalize(flat_rows, sep=".").fillna("")
    st.metric("Lượt phân tích", len(reports))
    st.dataframe(table, hide_index=True, width="stretch")

    comparisons = [
        report["metrics"] for report in reports
        if report["module"] == "Model Comparison" and "ground_truth" in report["metrics"]
    ]
    if comparisons:
        st.subheader("Sai số trên các ảnh có ground truth nhập trong phiên này")
        rows = []
        for model, error_key in (("YOLO", "yolo_signed_error"), ("CSRNet", "csrnet_signed_error")):
            errors = [float(item[error_key]) for item in comparisons]
            rows.append({
                "Model": model,
                "Images": len(errors),
                "MAE": sum(abs(error) for error in errors) / len(errors),
                "RMSE": math.sqrt(sum(error * error for error in errors) / len(errors)),
            })
        st.dataframe(pd.DataFrame(rows).round(2), hide_index=True, width="stretch")

    line_runs = [
        report["metrics"] for report in reports
        if report["module"] == "Line Crossing" and "ground_truth_IN" in report["metrics"]
    ]
    if line_runs:
        st.subheader("Đối chiếu line crossing với ground truth thủ công")
        line_table = pd.DataFrame([
            {
                "Videos": len(line_runs),
                "Mean absolute IN error": sum(item["IN_absolute_error"] for item in line_runs) / len(line_runs),
                "Mean absolute OUT error": sum(item["OUT_absolute_error"] for item in line_runs) / len(line_runs),
                "Mean absolute TOTAL error": sum(item["total_absolute_error"] for item in line_runs) / len(line_runs),
            }
        ])
        st.dataframe(line_table.round(2), hide_index=True, width="stretch")

    selected_index = st.selectbox(
        "Chi tiết lượt chạy",
        range(len(reports)),
        format_func=lambda index: f"{reports[index]['module']} · {reports[index]['input']} · {reports[index]['timestamp']}",
    )
    st.json(reports[selected_index])
    report_json = json.dumps(reports, ensure_ascii=False, indent=2)
    csv_bytes = table.to_csv(index=False).encode("utf-8-sig")
    left, right = st.columns(2)
    left.download_button("Tải báo cáo JSON", report_json, "crowdvision_session_report.json", "application/json", width="stretch")
    right.download_button("Tải báo cáo CSV", csv_bytes, "crowdvision_session_report.csv", "text/csv", width="stretch")
    if st.button("Xóa báo cáo phiên này"):
        st.session_state["run_reports"] = []
        st.rerun()

