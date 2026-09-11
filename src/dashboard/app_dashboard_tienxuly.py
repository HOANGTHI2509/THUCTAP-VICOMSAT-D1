"""Topic 1 dashboard: raw fuel versus the causal purple CleanFuel curve."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.dashboard.dashboard_data import (  # noqa: E402
    available_vehicle_sources,
    estimate_capacity_liters,
    load_telemetry_csv,
    run_topic1_filter,
)


DATASET_CHOICES = {
    "fulltt (Dữ liệu thô chưa xử lý - 9 xe)": PROJECT_ROOT / "fulltt",
    "CarFuelHistory (Bộ 5 xe - Thunghiem5)": PROJECT_ROOT / "Thunghiem5" / "CarFuelHistory.xlsx",
    "Toàn bộ dữ liệu thô (14 xe: fulltt + CarFuelHistory)": "ALL_RAW",
}
DATA_DIRECTORY = Path(os.getenv("FUEL_DATA_DIRECTORY", PROJECT_ROOT / "fulltt"))
MODEL_DIRECTORY = str(PROJECT_ROOT / "models" / "fuel_state_classifier")
FILTER_PIPELINE_VERSION = "smooth-tracking-origin-dev-1cbc895-v4"

st.set_page_config(
    page_title="VCOMSAT Fuel Denoising",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("📈 Dashboard lọc nhiễu nhiên liệu — Đề tài 1")
st.caption(
    "Đường tím là CleanFuel causal theo thời gian thực. Dashboard không kết luận "
    "nạp/rút, đỗ hay tắt máy."
)


@st.cache_data(show_spinner=False)
def _load_source(file_path: str, modified_at_ns: int):
    del modified_at_ns
    return load_telemetry_csv(file_path)


@st.cache_data(show_spinner=False)
def _filter_source(
    frame,
    vehicle_id: str,
    capacity_est_liters: float,
    pipeline_version: str,
):
    del pipeline_version
    return run_topic1_filter(
        frame,
        vehicle_id=vehicle_id,
        capacity_est_liters=capacity_est_liters,
        model_dir=MODEL_DIRECTORY,
    )


with st.sidebar:
    st.header("⚙️ Dữ liệu")
    selected_dataset = st.radio(
        "📁 Nguồn dữ liệu",
        list(DATASET_CHOICES.keys()),
        index=0,  # Mặc định fulltt
    )
    DATA_DIRECTORY = DATASET_CHOICES[selected_dataset]

sources = available_vehicle_sources(DATA_DIRECTORY)
if not sources:
    st.warning(f"Chưa có dữ liệu trong `{DATA_DIRECTORY}`.")
    st.stop()

with st.sidebar:
    vehicle_id = st.selectbox("🚚 Chọn xe", list(sources))

source_path = sources[vehicle_id]
if isinstance(source_path, str) and "::" in source_path:
    real_path = Path(source_path.split("::")[0])
    mtime = real_path.stat().st_mtime_ns
else:
    real_path = Path(source_path)
    mtime = real_path.stat().st_mtime_ns
frame = _load_source(str(source_path), mtime)
if frame.empty:
    st.warning("File telemetry không có điểm hợp lệ.")
    st.stop()

min_date = frame["FuelTime"].min().date()
max_date = frame["FuelTime"].max().date()
with st.sidebar:
    date_range = st.date_input(
        "📅 Khoảng hiển thị",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date,
    )
    segment_options = ["Tất cả phân đoạn"] + [
        str(value) for value in frame["SegmentID"].drop_duplicates().tolist()
    ]
    selected_segment = st.selectbox("📏 Phân đoạn", segment_options)

scope = frame.copy()
if selected_segment != "Tất cả phân đoạn":
    scope = scope[scope["SegmentID"].astype(str) == selected_segment].copy()

capacity_est_liters = estimate_capacity_liters(scope, vehicle_id=vehicle_id)
with st.sidebar:
    st.markdown("---")
    st.caption(
        f"Capacity: {capacity_est_liters:.1f} L. "
        "Q/R dùng cấu hình versioned của Smooth-Tracking."
    )
    st.caption("Muốn đổi Q/R phải cập nhật config và chạy golden/KPI, không chỉnh trên dashboard.")

with st.spinner("Đang chạy Smooth-Tracking causal…"):
    filtered = _filter_source(
        scope,
        vehicle_id,
        capacity_est_liters,
        FILTER_PIPELINE_VERSION,
    )


if isinstance(date_range, (tuple, list)) and len(date_range) == 2:
    start_date, end_date = date_range
else:
    start_date = end_date = date_range
visible = filtered[
    (filtered["FuelTime"].dt.date >= start_date)
    & (filtered["FuelTime"].dt.date <= end_date)
].copy()
if visible.empty:
    st.warning("Không có dữ liệu trong khoảng đã chọn.")
    st.stop()

latest = visible.iloc[-1]
metric_columns = st.columns(6)
metric_columns[0].metric("Bản ghi", f"{len(visible):,}")
metric_columns[1].metric("RawFuel", f"{latest['FuelLevel']:.1f} L")
metric_columns[2].metric("CleanFuel (tím)", f"{latest['CleanFuel']:.1f} L")
if "Kalman_Adaptive" in latest and not pd.isna(latest["Kalman_Adaptive"]):
    metric_columns[3].metric("Kalman Adaptive", f"{latest['Kalman_Adaptive']:.1f} L")
else:
    metric_columns[3].metric("Kalman Adaptive", "N/A")
metric_columns[4].metric("SignalState", str(latest["SignalState"]))
metric_columns[5].metric("QualityFlag", str(latest["QualityFlag"]))

fig = make_subplots(
    rows=3,
    cols=1,
    shared_xaxes=True,
    vertical_spacing=0.06,
    subplot_titles=(
        "1. RawFuel, CleanFuel (Smooth-Tracking) và Kalman Adaptive",
        "2. Tốc độ di chuyển",
        "3. Độ nhiễu cục bộ (Rolling Std)",
    ),
    row_heights=[0.62, 0.19, 0.19],
)
for segment_number, (_, plot_segment) in enumerate(
    visible.groupby("SegmentID", sort=False, dropna=False)
):
    show_legend = segment_number == 0
    fig.add_trace(
        go.Scatter(
            x=plot_segment["FuelTime"],
            y=plot_segment["FuelLevel"],
            mode="lines+markers",
            name="RawFuel",
            legendgroup="raw-fuel",
            showlegend=show_legend,
            line={"color": "#EF553B", "width": 1.2},
            marker={"size": 3},
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=plot_segment["FuelTime"],
            y=plot_segment["CleanFuel"],
            mode="lines",
            name="CleanFuel (tím)",
            legendgroup="clean-fuel",
            showlegend=show_legend,
            line={"color": "#AB63FA", "width": 2.8},
            customdata=plot_segment[["SignalState", "QualityFlag", "MotionState"]],
            hovertemplate=(
                "%{x}<br>CleanFuel: %{y:.2f} L<br>SignalState: %{customdata[0]}"
                "<br>QualityFlag: %{customdata[1]}<br>MotionState: %{customdata[2]}<extra></extra>"
            ),
        ),
        row=1,
        col=1,
    )
    if "Kalman_Adaptive" in plot_segment.columns:
        fig.add_trace(
            go.Scatter(
                x=plot_segment["FuelTime"],
                y=plot_segment["Kalman_Adaptive"],
                mode="lines",
                name="Kalman Adaptive (xanh)",
                legendgroup="kalman-adaptive",
                showlegend=show_legend,
                line={"color": "#00CC96", "width": 2.2, "dash": "solid"},
                hovertemplate=(
                    "%{x}<br>Kalman Adaptive: %{y:.2f} L<extra></extra>"
                ),
            ),
            row=1,
            col=1,
        )
    fig.add_trace(
        go.Scatter(
            x=plot_segment["FuelTime"],
            y=plot_segment["Speed"],
            mode="lines",
            name="Speed (km/h)",
            legendgroup="speed",
            showlegend=show_legend,
            line={"color": "#636EFA", "width": 1.5},
            fill="tozeroy",
            fillcolor="rgba(99, 110, 250, 0.12)",
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=plot_segment["FuelTime"],
            y=plot_segment["RollingStd"],
            mode="lines",
            name="Rolling Std (L)",
            legendgroup="rolling-std",
            showlegend=show_legend,
            line={"color": "#EF553B", "width": 1.5},
            fill="tozeroy",
            fillcolor="rgba(239, 85, 59, 0.12)",
        ),
        row=3,
        col=1,
    )
fig.update_layout(
    height=860,
    template="plotly_white",
    hovermode="x unified",
    legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "right", "x": 1},
    margin={"l": 30, "r": 30, "t": 65, "b": 25},
)
fig.update_yaxes(title_text="Lít", row=1, col=1)
fig.update_yaxes(title_text="km/h", row=2, col=1)
fig.update_yaxes(title_text="Std (L)", row=3, col=1)
st.plotly_chart(fig, width="stretch")

with st.expander("🔎 Data Inspector — trạng thái lọc từng điểm", expanded=True):
    columns = [
        "FuelTime",
        "FuelLevel",
        "CleanFuel",
        "Kalman_Adaptive",
        "Speed",
        "SignalState",
        "QualityFlag",
        "MotionState",
        "MotionConfidence",
        "GpsDisplacementMeters",
        "RollingStd",
    ]
    st.dataframe(visible[[column for column in columns if column in visible]], width="stretch", height=360)

st.info(
    "SignalState mô tả tín hiệu; QualityFlag mô tả hành động lọc; MotionState chỉ là "
    "bằng chứng từ GPS/vận tốc. Không dùng ba trường này để kết luận nạp/rút hoặc đỗ xe."
)
