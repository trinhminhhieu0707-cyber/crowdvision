from __future__ import annotations

import streamlit as st

st.set_page_config(page_title="CrowdVision", page_icon="CV", layout="wide")
st.markdown(
    """
    <style>
      .block-container {padding-top: 1.35rem; padding-bottom: 2.5rem; max-width: 1500px;}
      [data-testid="stAppViewContainer"] {background:linear-gradient(180deg,#f7f9ff 0%,#f3f6fb 100%);}
      [data-testid="stSidebar"] {background:linear-gradient(180deg,#111827 0%,#1e1b4b 100%);}
      [data-testid="stSidebar"] * {color:#e5e7eb;}
      [data-testid="stSidebar"] [data-baseweb="select"] div {color:#111827;}
      [data-testid="stMetric"] {background:#ffffff;border:1px solid #e5eaf3;padding:12px 16px;border-radius:14px;box-shadow:0 4px 16px #0f172a0a;}
      [data-testid="stMetricLabel"] p {font-size:0.9rem !important;}
      [data-testid="stMetricValue"] {font-size:1.65rem !important;}
      div.stButton > button[kind="primary"] {border:0;border-radius:10px;background:linear-gradient(100deg,#4f46e5,#0891b2);font-weight:650;}
      div.stButton > button {border-radius:10px;}
      [data-testid="stContainer"] {border-radius:16px;}
    </style>
    """,
    unsafe_allow_html=True,
)

from src.ui.common import configure_sidebar
from src.ui.dashboard import page_dashboard
from src.ui.image_analysis import page_image_analysis
from src.ui.video_analysis import page_video_analysis
from src.ui.tracking import page_tracking
from src.ui.crowd_density import page_crowd_density
from src.ui.model_comparison import page_model_comparison
from src.ui.zone_monitoring import page_zone_monitoring
from src.ui.reports import page_evaluation_reports


page, confidence, iou, image_size, device, weights = configure_sidebar()
if page == "Dashboard":
    page_dashboard()
elif page == "Image Analysis":
    page_image_analysis(device, weights, confidence, iou, image_size)
elif page == "Video Analysis":
    page_video_analysis(device, weights, confidence, iou, image_size)
elif page == "Tracking & Line Crossing":
    page_tracking(device, weights, confidence, iou, image_size)
elif page == "Crowd Density":
    page_crowd_density(device)
elif page == "Model Comparison":
    page_model_comparison(device, weights, confidence, iou, image_size)
elif page == "Zone Monitoring":
    page_zone_monitoring(device, weights, confidence, iou, image_size)
else:
    page_evaluation_reports()
