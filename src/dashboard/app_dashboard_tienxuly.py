# Trình điều khiển Dashboard (Đã khôi phục Segment cũ)
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import sys
import os

# Ensure the root directory is in the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

# Custom Imports
from src.core.filters import kalman_traditional as kalman
from src.core.filters import kalman_adaptive
from src.core.filters.median_filter import apply_median_filter
BoLocKalmanThichNghi1D = kalman_adaptive.BoLocKalmanThichNghi1D
from src.core.filters.ai_enhanced_adaptive import filter_ai_enhanced_adaptive
from src.core.filters.ai_enhanced_adaptive_realtime import filter_ai_enhanced_adaptive_realtime
is_valid_measurement = kalman_adaptive.is_valid_measurement
classify_signal_modes = getattr(kalman_adaptive, "classify_signal_modes", None)

if classify_signal_modes is None:
    def classify_signal_modes(
        group,
        vehicle_profile=None,
        source_col="FuelLevel",
        output_col="ProfileCleanFuel",
        mode_col="SignalMode",
        **_,
    ):
        group = group.copy()
        group[output_col] = group[source_col]
        group[mode_col] = "NORMAL"
        return group


def clean_transient_shapes(
    group,
    vehicle_profile=None,
    source_col="FuelLevel",
    output_col="ShapeCleanFuel",
    flag_col="ShapeCleanFlag",
    max_points=12,
):
    group = group.copy()
    raw = pd.to_numeric(group[source_col], errors="coerce").to_numpy(dtype=float)
    speed_values = pd.to_numeric(
        group.get("Speed", group.get("MotionSpeed", pd.Series(0.0, index=group.index))),
        errors="coerce",
    ).fillna(0.0).to_numpy(dtype=float)
    rolling_values = pd.to_numeric(
        group.get("RollingStd", pd.Series(raw, index=group.index).rolling(window=12, min_periods=1).std()),
        errors="coerce",
    ).fillna(0.0).to_numpy(dtype=float)
    clean = raw.copy()
    flags = np.array(["RAW"] * len(group), dtype=object)

    if len(group) < 3:
        group[output_col] = clean
        group[flag_col] = flags
        return group

    capacity = float(getattr(vehicle_profile, "capacity_est", np.nan))
    profile = vehicle_profile
    capacity = float(getattr(profile, "capacity_est", np.nan))
    if pd.isna(capacity) or capacity <= 0:
        valid = raw[(~np.isnan(raw)) & (raw > 0)]
        capacity = float(np.nanquantile(valid, 0.995)) if len(valid) else 200.0
    flat = float(profile.flat_jitter_threshold) if profile is not None else 0.8
    spike_gate = float(profile.spike_threshold) if profile is not None else 3.0
    return_gate = max(spike_gate * 1.0, flat * 2.5, 3.5)
    max_points = max(3, int(max_points))

    # One-point spike: neighbors agree, center is far away.
    for i in range(1, len(raw) - 1):
        if np.isnan(raw[i - 1]) or np.isnan(raw[i]) or np.isnan(raw[i + 1]):
            continue
        neighbor_level = 0.5 * (clean[i - 1] + raw[i + 1])
        neighbors_match = abs(clean[i - 1] - raw[i + 1]) <= return_gate
        center_far = abs(raw[i] - neighbor_level) >= spike_gate
        if neighbors_match and center_far:
            clean[i] = neighbor_level
            flags[i] = "SINGLE_SPIKE"

    i = 1
    while i < len(raw) - 2:
        if np.isnan(clean[i]) or np.isnan(clean[i - 1]):
            i += 1
            continue

        prev_window = clean[max(0, i - 5) : i]
        prev_window = prev_window[(~np.isnan(prev_window)) & (prev_window > 0)]
        if len(prev_window) < 2:
            i += 1
            continue

        baseline = float(np.median(prev_window))
        first_delta = clean[i] - baseline
        # CHỈ xử lý HỐ SỤT CHỮ U (first_delta < -return_gate), TUYỆT ĐỐI KHÔNG can thiệp vào nạp nhiên liệu
        if first_delta > -return_gate:
            i += 1
            continue

        best_return = None
        search_end = min(len(raw), i + max_points + 1)
        for j in range(i + 1, search_end):
            if np.isnan(clean[j]):
                continue
            if abs(clean[j] - baseline) <= return_gate:
                best_return = j
                break

        if best_return is None:
            i += 1
            continue

        span = clean[i : best_return + 1]
        valid_span = span[~np.isnan(span)]
        if len(valid_span) < 2:
            i += 1
            continue

        amplitude = float(np.max(np.abs(valid_span - baseline)))
        span_speed = speed_values[i : best_return + 1]
        span_rolling = rolling_values[i : best_return + 1]
        max_speed = float(np.nanmax(span_speed)) if len(span_speed) else 0.0
        max_rolling = float(np.nanmax(span_rolling)) if len(span_rolling) else 0.0
        span_direction_changes = 0
        if len(valid_span) >= 3:
            span_diff_sign = np.sign(np.diff(valid_span))
            span_diff_sign = span_diff_sign[span_diff_sign != 0]
            if len(span_diff_sign) >= 2:
                span_direction_changes = int(np.sum(span_diff_sign[1:] != span_diff_sign[:-1]))
        after_window = clean[best_return : min(len(clean), best_return + 4)]
        after_window = after_window[(~np.isnan(after_window)) & (after_window > 0)]
        returns_and_stays = len(after_window) == 0 or abs(float(np.median(after_window)) - baseline) <= return_gate
        noise_context = (
            max_speed > 8.0
            or max_rolling >= max(flat * 1.6, spike_gate * 0.45)
            or span_direction_changes >= 2
            or max_speed <= 1.0
        )

        # Do not erase real refuel/drain: only short excursions returning to old level.
        if amplitude >= spike_gate and returns_and_stays and noise_context:
            end_level = float(clean[best_return])
            replacement = np.linspace(baseline, end_level, best_return - i + 1)
            clean[i : best_return + 1] = replacement
            flags[i : best_return + 1] = "TRANSIENT_SHAPE"
            i = best_return + 1
        else:
            i += 1

    group[output_col] = clean
    group[flag_col] = flags
    return group

from src.core.filters.fuel_state_filter import filter_fuel_series
from src.core.filters.ai_state_filter import filter_with_ai_state, load_fuel_state_classifier
from src.core.filters.tcn_state_classifier import load_fuel_state_tcn
from src.core.filters.anomaly_detector import FuelAnomalyDetector

@st.cache_resource
def load_ai_state_model():
    return load_fuel_state_classifier("models/fuel_state_classifier")

ai_state_model, ai_state_metadata = load_ai_state_model()

@st.cache_resource
def load_tcn_state_model():
    model, metadata = load_fuel_state_tcn("models/fuel_state_tcn")
    if metadata and float(metadata.get("test_accuracy", 0.0) or 0.0) < 0.90:
        return None, None
    return model, metadata

tcn_state_model, tcn_state_metadata = load_tcn_state_model()



try:
    from src.utils.calculate_metrics import calculate_metrics
except ImportError:
    pass

# --- CONFIGURATION ---
st.set_page_config(page_title="Vcomsat Fuel Dashboard", layout="wide", initial_sidebar_state="expanded")

# Custom CSS for light premium look
st.markdown("""
<style>
    .main {background-color: #F8F9FA;}
    h1, h2, h3 {color: #212529; font-family: 'Inter', sans-serif;}
</style>
""", unsafe_allow_html=True)

st.title("🔬 Dashboard Dữ Liệu Tiền Xử Lý (26 Xe)")
st.markdown("**Trực quan hóa và so sánh hiệu năng của các thuật toán: Raw, Kalman, và CNN-GA**")

with st.expander("🏆 PERFORMANCE MATRIX / METRIC HEATMAP", expanded=False):
    st.markdown("Đánh giá định lượng toàn diện trên **Synthetic Test Set** (15 segments, 4500 samples chưa từng được huấn luyện).")
    try:
        import plotly.figure_factory as ff
        df_bench = pd.read_csv('artifacts/benchmark_summary.csv')
        models_str = [m.replace('1D-CNN + Gated Attention', 'CNN-GA').replace('Standard Kalman', 'Standard').replace('Adaptive Kalman', 'Adaptive') for m in df_bench['Model'].values]
        metrics_info = {
            'RMSE (L) ↓': -1, 'SNR (dB) ↑': 1, 'Smoothness (L/step)': -1,
            'Refuel F1 ↑': 1, 'Theft F1 ↑': 1, 'Delay (steps) ↓': -1, 'Latency (ms) ↓': -1
        }
        
        matrix = []
        labels_text = []
        for metric, direction in metrics_info.items():
            vals = pd.to_numeric(df_bench[metric].astype(str).str.replace('~', ''), errors='coerce').fillna(0).values
            v_min, v_max = np.min(vals), np.max(vals)
            norm = np.ones_like(vals) if v_max == v_min else ((vals - v_min) / (v_max - v_min) if direction == 1 else (v_max - vals) / (v_max - v_min))
            matrix.append(norm)
            fmt_labels = [f"{val:.3f}" if 0 < val < 0.01 else (f"{val:.2f}" if val != 0 else "0.0") for val in vals]
            labels_text.append(fmt_labels)
            
        y_labels = [m.replace(' (L)', '').replace(' (dB)', '').replace(' (L/step)', '').replace(' (steps)', '').replace(' (ms)', '') for m in metrics_info.keys()]
        
        fig_heat = ff.create_annotated_heatmap(
            z=matrix,
            x=models_str,
            y=y_labels,
            annotation_text=labels_text,
            colorscale='RdYlGn',
            showscale=False
        )
        fig_heat.update_layout(height=450, margin=dict(t=50, l=100, r=20, b=20), font=dict(size=14))
        # Add colorbar manually since create_annotated_heatmap hides it by default
        fig_heat['data'][0]['showscale'] = True
        fig_heat['data'][0]['colorbar'] = dict(title='Scale (0-1)')
        
        st.plotly_chart(fig_heat, width='stretch')
        st.markdown("*Lưu ý: 🟩 Xanh = Tốt nhất, 🟥 Đỏ = Kém nhất. Mỗi hàng được chuẩn hóa độc lập theo thang Min-Max.*")
    except Exception as e:
        st.warning(f"Chưa tìm thấy dữ liệu Benchmark. Vui lòng chạy script `benchmark_cnn.py` trước. Lỗi: {e}")

@st.cache_data
def load_data(file_path, car_id=None):
    if car_id is None and "source_by_label" in globals() and file_path in source_by_label:
        source = source_by_label[file_path]
        file_path = source["path"]
        car_id = source["car_id"]
    elif car_id is None:
        car_id = file_path
        file_path = f"TienXuLy/{car_id}_processed.csv"

    if not os.path.exists(file_path):
        st.error(f"Không tìm thấy file dữ liệu cho xe: {car_id}")
        return pd.DataFrame()
        
    df = pd.read_csv(file_path)
    df['FuelTime'] = pd.to_datetime(df['FuelTime'], errors="coerce")
    df['FuelLevel'] = pd.to_numeric(df['FuelLevel'], errors="coerce")
    if 'SegmentID' in df.columns:
        df['SegmentID'] = pd.to_numeric(df['SegmentID'], errors="coerce")
    else:
        df['SegmentID'] = 0
    if 'Speed' not in df.columns and 'MotionSpeed' in df.columns:
        df['Speed'] = pd.to_numeric(df['MotionSpeed'], errors="coerce")
    
    return df


def load_vehicle_profile(car_id, df):
    fuel = pd.to_numeric(df.get("FuelLevel"), errors="coerce")
    fuel = fuel[(fuel > 0) & fuel.notna()]
    capacity = float(fuel.quantile(0.995)) if not fuel.empty else 200.0
    if pd.isna(capacity) or capacity < 50.0:
        capacity = 200.0
    return type(
        "VehicleProfile",
        (),
        {
            "capacity_est": capacity,
            "noise_sigma_liters": max(0.5, 0.002 * capacity),
            "flat_jitter_threshold": max(0.8, 0.003 * capacity),
            "spike_threshold": max(3.0, 0.012 * capacity),
            "event_threshold": max(6.0, 0.035 * capacity),
        },
    )()

import glob

def build_data_sources():
    sources = []

    for file_path in glob.glob("TienXuLy/*_processed.csv"):
        car_id = os.path.basename(file_path).replace("_processed.csv", "")
        sources.append(
            {
                "label": car_id,
                "car_id": car_id,
                "path": file_path,
                "source": "TienXuLy",
            }
        )

    for file_path in glob.glob("data/processed/CarFuelHistory_Processed_Car*.csv"):
        if file_path.endswith("_CNN_Realtime.csv"):
            continue
        car_id = (
            os.path.basename(file_path)
            .replace("CarFuelHistory_Processed_", "")
            .replace(".csv", "")
        )
        sources.append(
            {
                "label": f"{car_id} (5-car GPS)",
                "car_id": car_id,
                "path": file_path,
                "source": "5-car GPS",
            }
        )

    return sorted(sources, key=lambda item: (item["source"], item["car_id"]))

data_sources = build_data_sources()
source_labels = [item["label"] for item in data_sources]
source_by_label = {item["label"]: item for item in data_sources}
cars = source_labels

if not data_sources:
    st.error("Không tìm thấy dữ liệu đã xử lý.")
    st.stop()

with st.sidebar:
    st.header("⚙️ Cấu hình Dữ liệu")
    selected_car = st.selectbox("🚗 Chọn Xe (Vehicle ID)", cars)
    
df = load_data(selected_car)
vehicle_profile = load_vehicle_profile(selected_car, df)

min_date = df['FuelTime'].min().date()
max_date = df['FuelTime'].max().date()

default_start_date = max(min_date, max_date - pd.Timedelta(days=7))

with st.sidebar:
    st.markdown("---")
    st.header("📅 Bộ lọc Thời gian")
    
    if "last_car" not in st.session_state or st.session_state.last_car != selected_car:
        st.session_state.date_range_key = (default_start_date, max_date)
        st.session_state.last_car = selected_car
        
    def reset_date_range():
        st.session_state.date_range_key = (min_date, max_date)
        
    st.button("🔄 Trở về Toàn Bộ Thời Gian", on_click=reset_date_range)
    
    date_range = st.date_input(
        "Chọn ngày (Một hoặc nhiều ngày)",
        min_value=min_date,
        max_value=max_date,
        key="date_range_key"
    )

if len(date_range) == 2:
    start_date, end_date = date_range
    df = df[(df['FuelTime'].dt.date >= start_date) & (df['FuelTime'].dt.date <= end_date)]
elif len(date_range) == 1:
    start_date = date_range[0]
    df = df[df['FuelTime'].dt.date == start_date]

if df.empty:
    st.warning("Không có dữ liệu trong khoảng thời gian này. Vui lòng chọn ngày khác.")
    st.stop()

with st.sidebar:
    st.markdown("---")
    st.header("🧠 Chế độ AI Filter")
    ai_filter_mode = st.radio("Chế độ:", ["offline", "realtime"], index=0, help="offline: dùng tương lai. realtime: xử lý nhân quả (causal)")
    st.markdown("---")
    st.header("📏 Chọn Phân Đoạn")
    segment_options = ["Toàn bộ dữ liệu trong khoảng thời gian (All)"] + list(df['SegmentID'].dropna().unique())
    selected_segment = st.selectbox("Hiển thị theo phân đoạn", segment_options)

if selected_segment == "Toàn bộ dữ liệu trong khoảng thời gian (All)":
    df_seg = df.copy()
else:
    df_seg = df[df['SegmentID'] == selected_segment].copy()

df_seg["_OriginalOrder"] = np.arange(len(df_seg))
df_seg = df_seg.sort_values(["SegmentID", "FuelTime", "_OriginalOrder"], kind="stable")

estimated_capacity = vehicle_profile.capacity_est
if pd.isna(estimated_capacity) or estimated_capacity < 50.0:
    estimated_capacity = df_seg['FuelLevel'].quantile(0.99)
if pd.isna(estimated_capacity) or estimated_capacity < 50.0:
    estimated_capacity = 200.0

with st.sidebar:
    st.markdown("---")
    st.header("🎛️ Tinh chỉnh Thuật toán")
    kalman_r_default = int(max(64.0, (0.04 * estimated_capacity)**2))
    kalman_r = st.slider("Nhiễu đo lường Kalman (R)", min_value=1, max_value=2000, value=kalman_r_default, step=1)
    
    st.subheader("🤖 Cấu hình Adaptive Kalman")

    adapt_threshold_default = float(vehicle_profile.event_threshold)
    adapt_threshold = st.slider("Ngưỡng bắt nhảy (Threshold) - Lít", min_value=1.0, max_value=max(100.0, adapt_threshold_default*2), value=adapt_threshold_default, step=0.5, help="Chênh lệch tối thiểu để bắt đầu nghi ngờ có bơm/rút xăng")
    adapt_persistence = st.slider("Số nhịp chờ xác nhận (Persistence)", min_value=1, max_value=15, value=3, step=1, help="Số chu kỳ tín hiệu phải duy trì ở mức cao/thấp để xác nhận là đổ/rút thật (lọc đỉnh nhiễu)")
    st.caption(
        f"Profile xe: capacity={vehicle_profile.capacity_est:.1f}L, "
        f"noise={vehicle_profile.noise_sigma_liters:.2f}L, "
        f"jitter={vehicle_profile.flat_jitter_threshold:.1f}L, "
        f"spike={vehicle_profile.spike_threshold:.1f}L"
    )

    st.subheader("🧠 Cấu hình AI-Enhanced Kalman")
    with st.expander("Tùy chỉnh R & Q theo Nhãn AI", expanded=False):
        st.markdown("**1. UNKNOWN (Mặc định)**")
        col1, col2 = st.columns(2)
        cfg_unk_r = col1.number_input("R (UNKNOWN)", value=16.0, step=1.0, min_value=0.1)
        cfg_unk_q = col2.number_input("Q (UNKNOWN)", value=0.25, step=0.01, min_value=0.0001, format="%.4f")

        st.markdown("**2. REFUEL (Đang chờ)**")
        col1, col2 = st.columns(2)
        cfg_ref_r = col1.number_input("R (REFUEL)", value=1.0, step=0.1, min_value=0.1)
        cfg_ref_q = col2.number_input("Q (REFUEL)", value=5.0, step=0.1, min_value=0.0001, format="%.4f")

        st.markdown("**3. SLOSHING_NOISE**")
        col1, col2 = st.columns(2)
        cfg_slosh_r = col1.number_input("R (SLOSHING)", value=100.0, step=10.0, min_value=1.0)
        cfg_slosh_q = col2.number_input("Q (SLOSHING)", value=0.005, step=0.001, min_value=0.0001, format="%.4f")

        st.markdown("**4. CONSUMPTION**")
        col1, col2 = st.columns(2)
        cfg_cons_r = col1.number_input("R (CONSUMPTION)", value=30.0, step=5.0, min_value=0.1)
        cfg_cons_q = col2.number_input("Q (CONSUMPTION)", value=0.20, step=0.01, min_value=0.0001, format="%.4f")

        st.markdown("**5. DRAIN (Sụt giảm)**")
        col1, col2 = st.columns(2)
        cfg_drain_r = col1.number_input("R (DRAIN)", value=5.0, step=1.0, min_value=0.1)
        cfg_drain_q = col2.number_input("Q (DRAIN)", value=2.0, step=0.1, min_value=0.0001, format="%.4f")

        st.markdown("**6. STABLE_JITTER**")
        col1, col2 = st.columns(2)
        cfg_stable_r = col1.number_input("R (STABLE)", value=15.0, step=1.0, min_value=0.1)
        cfg_stable_q = col2.number_input("Q (STABLE)", value=0.2, step=0.01, min_value=0.0001, format="%.4f")
        cfg_stable_r_very = col1.number_input("R (STABLE RẤT ÊM)", value=5.0, step=1.0, min_value=0.1)

        st.markdown("**7. SPIKE (Nhiễu gai)**")
        col1, col2 = st.columns(2)
        cfg_spike_r = col1.number_input("R (SPIKE)", value=10000.0, step=100.0, min_value=1.0)
        cfg_spike_q = col2.number_input("Q (SPIKE)", value=0.0001, step=0.0001, min_value=0.00001, format="%.5f")
        
    ai_kalman_config = {
        "UNKNOWN": (cfg_unk_r, cfg_unk_q),
        "REFUEL": (cfg_ref_r, cfg_ref_q),
        "SLOSHING": (cfg_slosh_r, cfg_slosh_q),
        "CONSUMPTION": (cfg_cons_r, cfg_cons_q),
        "DRAIN": (cfg_drain_r, cfg_drain_q),
        "STABLE_JITTER": (cfg_stable_r, cfg_stable_q, cfg_stable_r_very),
        "SPIKE": (cfg_spike_r, cfg_spike_q)
    }

with st.spinner("Đang chạy thuật toán lọc nhiễu..."):
    df_seg['Custom_Kalman'] = np.nan
    df_seg['Custom_Adaptive_Kalman'] = np.nan
    df_seg['AI_Enhanced_Kalman'] = np.nan
    df_seg['AI_Enhanced_Kalman_Realtime'] = np.nan
    df_seg['Median_Filter'] = np.nan
    df_seg['FuelLevel_Filtered_EMA'] = np.nan
    df_seg['FuelLevel_Filtered_Edge'] = np.nan
    df_seg['FuelLevel_Filtered_AlphaBeta'] = np.nan
    
    anomaly_detector = FuelAnomalyDetector(capacity=estimated_capacity)
    df_seg = (
        df_seg.groupby("SegmentID", sort=False, dropna=False, group_keys=False)
        .apply(lambda g: anomaly_detector.detect_and_clean(g))
    )
    
    if "Acceleration" not in df_seg.columns:
        if "Speed" in df_seg.columns:
            time_gap_sec = df_seg["TimeGapMinutes"].fillna(5.0) * 60.0
            safe_time_gap = time_gap_sec.replace(0, 1.0)
            df_seg["Acceleration"] = df_seg.groupby("SegmentID", dropna=False)["Speed"].diff().fillna(0.0) / safe_time_gap
        else:
            df_seg["Acceleration"] = 0.0

    df_seg = (
        df_seg.groupby("SegmentID", sort=False, dropna=False, group_keys=False)
        .apply(
            lambda group: classify_signal_modes(
                group,
                vehicle_profile,
                source_col="FuelLevel",
                output_col="ProfileCleanFuel",
                mode_col="SignalMode",
                lookback=7,
                lookahead=5,
            )
        )
    )

    df_seg = (
        df_seg.groupby("SegmentID", sort=False, dropna=False, group_keys=False)
        .apply(
            lambda group: filter_fuel_series(
                group,
                vehicle_profile,
                source_col="FuelLevel",
                lookback=7,
                lookahead=5,
            )
        )
    )

    df_seg = (
        df_seg.groupby("SegmentID", sort=False, dropna=False, group_keys=False)
        .apply(
            lambda group: clean_transient_shapes(
                group,
                vehicle_profile,
                source_col="FuelLevel",
                output_col="ShapeCleanFuel",
                flag_col="ShapeCleanFlag",
                max_points=50,
            )
        )
    )

    if ai_state_model is not None and ai_state_metadata is not None:
        df_seg = (
            df_seg.groupby("SegmentID", sort=False, dropna=False, group_keys=False)
            .apply(
                lambda group: filter_with_ai_state(
                    group,
                    model=ai_state_model,
                    metadata=ai_state_metadata,
                    tcn_model=tcn_state_model,
                    tcn_metadata=tcn_state_metadata,
                    profile=vehicle_profile,
                    mode=ai_filter_mode,
                )
            )
        )
    else:
        df_seg["AI_State"] = "MODEL_NOT_FOUND"
        df_seg["AI_State_Raw"] = "MODEL_NOT_FOUND"
        df_seg["AI_State_Confidence"] = np.nan
        df_seg["AI_CleanInput"] = np.nan
        df_seg["AI_State_Filtered"] = np.nan

    for seg_id, group in df_seg.groupby('SegmentID', sort=False, dropna=False):
        if "CleanedFuel" in group.columns:
            adaptive_source_col = "CleanedFuel"
        elif "ShapeCleanFuel" in group.columns:
            adaptive_source_col = "ShapeCleanFuel"
        elif "ProfileCleanFuel" in group.columns:
            adaptive_source_col = "ProfileCleanFuel"
        else:
            adaptive_source_col = "FuelLevel"
        fuels = group[adaptive_source_col].tolist()
        
        group_rolling_std = group[adaptive_source_col].rolling(window=12, min_periods=1).std().fillna(0.0).tolist()
        group_raw_rolling_std = group["FuelLevel"].rolling(window=12, min_periods=1).std().fillna(0.0).tolist()
        kf_std = None
        kf_adapt = None
        
        kalman_std_vals = []
        kalman_adapt_vals = []
        x_f, P_f, x_p, P_p = [], [], [], []
        
        valid_gaps = group['TimeGapMinutes'].dropna()
        reference_gap = valid_gaps[valid_gaps > 0].median()
        if pd.isna(reference_gap) or reference_gap <= 0:
            reference_gap = 5.0
        khoang_thoi_gian_tich_luy_std = 0.0
        khoang_thoi_gian_tich_luy_adapt = 0.0
        
        for i_loc, dong in enumerate(group.itertuples()):
            idx = dong.Index
            gap = getattr(dong, "TimeGapMinutes", reference_gap)
            if pd.isna(gap) or gap <= 0: gap = reference_gap
            
            speed_value = float(getattr(dong, 'Speed', 0.0) or 0.0)
            movement_raw = str(getattr(dong, "MovementState", "")).strip().upper()
            if movement_raw in {"STOPPED", "STOP", "IDLE"}:
                movement_state = 0
            elif movement_raw:
                movement_state = 1
            else:
                movement_state = 0 if speed_value <= 3.0 else 1
            acceleration = float(getattr(dong, "Acceleration", 0.0))
            
            if not is_valid_measurement(getattr(dong, adaptive_source_col, None), getattr(dong, 'FeatureStatus', '')):
                khoang_thoi_gian_tich_luy_std += gap
                khoang_thoi_gian_tich_luy_adapt += gap
                kalman_std_vals.append(np.nan)
                x_f.append(np.nan)
                P_f.append(0.0)
                x_p.append(np.nan)
                P_p.append(0.0)
                continue
            
            measurement = float(getattr(dong, adaptive_source_col, dong.FuelLevel))
            future_values = pd.to_numeric(
                group[adaptive_source_col].iloc[i_loc + 1 : i_loc + 3],
                errors="coerce",
            ).dropna().to_numpy(dtype=float)
            previous_adaptive = x_f[-1] if x_f and not pd.isna(x_f[-1]) else measurement
            refuel_gate = max(vehicle_profile.event_threshold * 0.45, vehicle_profile.spike_threshold, 3.0)
            moderate_up_gate = max(
                vehicle_profile.flat_jitter_threshold * 1.2,
                vehicle_profile.noise_sigma_liters * 1.6,
                0.8,
            )
            future_median = float(np.median(future_values)) if len(future_values) > 0 else measurement
            moderate_up_confirm = (
                len(future_values) > 0
                and measurement - previous_adaptive >= moderate_up_gate
                and measurement - previous_adaptive < refuel_gate
                and future_median >= previous_adaptive + max(
                    vehicle_profile.flat_jitter_threshold * 0.8,
                    vehicle_profile.noise_sigma_liters * 1.1,
                    0.5,
                )
                and abs(future_median - measurement) <= max(
                    vehicle_profile.flat_jitter_threshold * 5.0,
                    vehicle_profile.noise_sigma_liters * 5.0,
                    abs(measurement - previous_adaptive) * 1.20,
                )
            )
            slow_or_stopped_refuel = (
                (movement_state == 0 or speed_value <= 3.0)
                and measurement - previous_adaptive >= refuel_gate
            )
            future_confirms_refuel = moderate_up_confirm or slow_or_stopped_refuel or (
                len(future_values) > 0
                and measurement - previous_adaptive >= refuel_gate
                and future_median >= previous_adaptive + refuel_gate * 0.60
                and abs(future_median - measurement) <= max(refuel_gate, abs(measurement - previous_adaptive) * 0.45)
            )
            
            khoang_thoi_gian_std = gap + khoang_thoi_gian_tich_luy_std
            khoang_thoi_gian_tich_luy_std = 0.0
            dt_ratio = khoang_thoi_gian_std / reference_gap
            
            if kf_std is None:
                kf_std = kalman.BoLocKalmanTieuChuan1D(trang_thai_ban_dau=measurement, nhieu_qua_trinh=1.0, nhieu_do_luong=kalman_r)
                kalman_std_vals.append(measurement)
            else:
                kalman_std_vals.append(kf_std.cap_nhat(measurement, ty_le_dt=dt_ratio))
                
            khoang_thoi_gian_adapt = gap + khoang_thoi_gian_tich_luy_adapt
            khoang_thoi_gian_tich_luy_adapt = 0.0
            if kf_adapt is None:
                kf_adapt = BoLocKalmanThichNghi1D(
                    trang_thai_ban_dau=measurement, 
                    capacity=estimated_capacity,
                    sai_so_uoc_luong_ban_dau=4.0, 
                    nhieu_qua_trinh=0.25,
                    r_co_ban=max(
                        6.0,
                        min(
                            36.0,
                            vehicle_profile.noise_sigma_liters ** 2
                            + vehicle_profile.flat_jitter_threshold ** 2,
                        ),
                    ),
                    nhip_cho_xac_nhan=adapt_persistence,
                )
                x_f.append(measurement)
            else:
                x_forward = kf_adapt.cap_nhat(
                    measurement, 
                    ty_le_dt=dt_ratio, 
                    trang_thai_chuyen_dong=movement_state, 
                    gia_toc=acceleration,
                    rolling_std=float(group_rolling_std[i_loc]),
                    van_toc=speed_value,
                    xac_nhan_nap_nhanh=bool(future_confirms_refuel),
                )
                x_f.append(x_forward)
        
        kalman_adapt_vals = x_f
        
        df_seg.loc[group.index, 'Custom_Kalman'] = kalman_std_vals
        df_seg.loc[group.index, 'Custom_Adaptive_Kalman'] = kalman_adapt_vals
        
        df_seg_subset = df_seg.loc[group.index]
        df_seg.loc[group.index, 'AI_Enhanced_Kalman'] = filter_ai_enhanced_adaptive(df_seg_subset, config=ai_kalman_config)
        df_seg.loc[group.index, 'AI_Enhanced_Kalman_Realtime'] = filter_ai_enhanced_adaptive_realtime(df_seg_subset, config=ai_kalman_config)
        df_seg.loc[group.index, 'Median_Filter'] = apply_median_filter(df_seg_subset, window_size=10)
        
        if 'AI_State_Filtered' in df_seg.columns:
            pass  # Giữ nguyên bản gốc để backup
# Restore original order just in case
df_seg = df_seg.sort_values("_OriginalOrder", kind="stable").drop(columns="_OriginalOrder")

# Metric cards
st.markdown("### 📊 Thông số phân đoạn hiện tại")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Tổng số bản ghi", f"{len(df_seg):,}")
col2.metric("Thời gian bắt đầu", df_seg['FuelTime'].min().strftime("%H:%M:%S %d/%m") if not df_seg.empty else "N/A")
col3.metric("Tham số R (Kalman)", f"{kalman_r}")
col4.metric("Bản ghi có lỗi (QualityFlag=1)", f"{df_seg['QualityFlag'].sum()}")

st.markdown("---")
st.markdown("### 📈 Biểu đồ Đấu trường Thuật toán (Đã Fix Index & Burn-in)")

# Create Plotly Subplots
fig = make_subplots(rows=3, cols=1, shared_xaxes=True, 
                    vertical_spacing=0.05,
                    specs=[[{"secondary_y": True}], [{}], [{}]],
                    subplot_titles=("1. Đối chứng: Mức Nhiên Liệu Gốc vs 3 Bộ Lọc", "2. Tốc độ di chuyển", "3. Độ nhiễu cục bộ (Rolling Std)"),
                    row_heights=[0.6, 0.2, 0.2])

# 1. Fuel Plot & Custom Algorithm Traces
show_legend = True
show_experimental_debug_traces = False
for seg_id, group in df_seg.groupby('SegmentID', sort=False):
    # Raw Fuel
    fig.add_trace(go.Scatter(x=group['FuelTime'], y=group['FuelLevel'], 
                             mode='lines+markers', name=f'Raw FuelLevel', legendgroup='raw', showlegend=show_legend,
                             line=dict(color='red', width=1), marker=dict(size=4), connectgaps=True), row=1, col=1, secondary_y=False)
                             
    # Kalman Standard
    fig.add_trace(go.Scatter(x=group['FuelTime'], y=group['Custom_Kalman'], 
                             mode='lines', name=f'Kalman Filter (R={kalman_r})', legendgroup='kalman', showlegend=show_legend,
                             line=dict(color='#00CC96', width=2), connectgaps=True, visible='legendonly'), row=1, col=1, secondary_y=False)
                             
    if show_experimental_debug_traces and 'ProfileCleanFuel' in group.columns:
        fig.add_trace(go.Scatter(x=group['FuelTime'], y=group['ProfileCleanFuel'],
                                 mode='lines', name='Profile-clean Fuel', legendgroup='profile_clean', showlegend=show_legend,
                                 line=dict(color='#444444', width=1.5, dash='dot'), connectgaps=True, visible='legendonly'), row=1, col=1, secondary_y=False)

    if show_experimental_debug_traces and 'SignalMode' in group.columns:
        signal_points = group[group['SignalMode'].isin(['SPIKE_UP', 'SPIKE_DOWN', 'REFUEL', 'DROP_EVENT', 'TREND_DOWN'])]
        if not signal_points.empty:
            fig.add_trace(go.Scatter(x=signal_points['FuelTime'], y=signal_points['FuelLevel'],
                                     mode='markers', name='SignalMode markers', legendgroup='signal_mode', showlegend=show_legend,
                                     text=signal_points['SignalMode'],
                                     marker=dict(size=7, color='black', symbol='circle-open'),
                                     visible='legendonly'), row=1, col=1, secondary_y=False)

    # Kalman Adaptive (Mặc định hiển thị)
    if show_experimental_debug_traces and 'FuelLevel_CleanInput' in group.columns:
        fig.add_trace(go.Scatter(x=group['FuelTime'], y=group['FuelLevel_CleanInput'],
                                 mode='lines', name='State clean input', legendgroup='state_clean', showlegend=show_legend,
                                 line=dict(color='#777777', width=1.5, dash='dot'), connectgaps=True,
                                 visible='legendonly'), row=1, col=1, secondary_y=False)

    if show_experimental_debug_traces and 'ShapeCleanFuel' in group.columns:
        fig.add_trace(go.Scatter(x=group['FuelTime'], y=group['ShapeCleanFuel'],
                                 mode='lines', name='Shape-clean Fuel', legendgroup='shape_clean', showlegend=show_legend,
                                 line=dict(color='#111111', width=1.5, dash='dot'), connectgaps=True,
                                 visible='legendonly'), row=1, col=1, secondary_y=False)

    if show_experimental_debug_traces and 'FuelLevel_Filtered_EMA' in group.columns:
        fig.add_trace(go.Scatter(x=group['FuelTime'], y=group['FuelLevel_Filtered_EMA'],
                                 mode='lines', name='State Robust EMA', legendgroup='state_ema', showlegend=show_legend,
                                 line=dict(color='#2CA02C', width=2), connectgaps=True,
                                 visible='legendonly'), row=1, col=1, secondary_y=False)

    state_edge_col = 'FuelLevel_Filtered_Edge' if 'FuelLevel_Filtered_Edge' in group.columns else 'FuelLevel_Filtered_AlphaBeta'
    if show_experimental_debug_traces and state_edge_col in group.columns:
        fig.add_trace(go.Scatter(x=group['FuelTime'], y=group[state_edge_col],
                                 mode='lines', name='State Edge Filter', legendgroup='state_edge', showlegend=show_legend,
                                 line=dict(color='#006400', width=2.5), connectgaps=True), row=1, col=1, secondary_y=False)

    if show_experimental_debug_traces and 'FuelState' in group.columns:
        fuel_state_points = group[group['FuelState'].isin(['SPIKE_UP', 'SPIKE_DOWN', 'DROPOUT', 'REFUEL_CONFIRMED', 'DROP_CONFIRMED', 'JITTER'])]
        if not fuel_state_points.empty:
            fig.add_trace(go.Scatter(x=fuel_state_points['FuelTime'], y=fuel_state_points['FuelLevel'],
                                     mode='markers', name='Fuel state markers', legendgroup='fuel_state', showlegend=show_legend,
                                     text=fuel_state_points['FuelState'],
                                     marker=dict(size=8, color='#006400', symbol='diamond-open'),
                                     visible='legendonly'), row=1, col=1, secondary_y=False)

    fig.add_trace(go.Scatter(x=group['FuelTime'], y=group['Custom_Adaptive_Kalman'], 
                             mode='lines', name=f'Adaptive Kalman (Dynamic R)', legendgroup='adapt', showlegend=show_legend,
                             line=dict(color='blue', width=2, dash='dash'), connectgaps=True), row=1, col=1, secondary_y=False)

    if 'AI_State_Filtered' in group.columns:
        fig.add_trace(go.Scatter(x=group['FuelTime'], y=group['AI_State_Filtered'],
                                 mode='lines', name='AI State Filter', legendgroup='ai_state_filter', showlegend=show_legend,
                                 line=dict(color='#111111', width=2.5), connectgaps=True,
                                 visible='legendonly'), row=1, col=1, secondary_y=False)

    if 'AI_Enhanced_Kalman' in group.columns:
        fig.add_trace(go.Scatter(x=group['FuelTime'], y=group['AI_Enhanced_Kalman'],
                                 mode='lines', name='AI-Enhanced Adaptive Kalman (Offline)',
                                 legendgroup='ai_enhanced_kalman', showlegend=show_legend,
                                 line=dict(color='#8A2BE2', width=2.5), connectgaps=True),
                      row=1, col=1, secondary_y=False)

    if 'AI_Enhanced_Kalman_Realtime' in group.columns:
        fig.add_trace(go.Scatter(x=group['FuelTime'], y=group['AI_Enhanced_Kalman_Realtime'],
                                 mode='lines', name='AI-Enhanced Kalman (Realtime / Causal)',
                                 legendgroup='ai_enhanced_kalman_realtime', showlegend=show_legend,
                                 line=dict(color='#FF7F0E', width=2.5), connectgaps=True),
                      row=1, col=1, secondary_y=False)

    if 'Median_Filter' in group.columns:
        fig.add_trace(go.Scatter(x=group['FuelTime'], y=group['Median_Filter'],
                                 mode='lines', name='Median Filter (N=10)',
                                 legendgroup='median_filter', showlegend=show_legend,
                                 line=dict(color='#FF1493', width=2.0), connectgaps=True,
                                 visible='legendonly'), row=1, col=1, secondary_y=False)
    if 'CleanedFuel' in group.columns:
        fig.add_trace(go.Scatter(x=group['FuelTime'], y=group['CleanedFuel'],
                                 mode='lines', name='Cleaned & Interpolated (Nối phẳng mượt Chữ U)',
                                 legendgroup='cleaned_fuel', showlegend=show_legend,
                                 line=dict(color='#008080', width=2.5, dash='solid'), connectgaps=True,
                                 visible='legendonly'),
                      row=1, col=1, secondary_y=False)

    if show_experimental_debug_traces and 'AI_State' in group.columns:
        ai_state_points = group[group['AI_State'].isin(['REFUEL', 'DRAIN', 'SPIKE', 'TRANSIENT_NOISE', 'TRANSIENT_UP_NOISE', 'TRANSIENT_DOWN_NOISE', 'TRANSIENT_CLUSTER_NOISE', 'SLOSHING_NOISE'])]
        if not ai_state_points.empty:
            fig.add_trace(go.Scatter(x=ai_state_points['FuelTime'], y=ai_state_points['FuelLevel'],
                                     mode='markers', name='AI state markers', legendgroup='ai_state_markers',
                                     showlegend=show_legend, text=ai_state_points['AI_State'],
                                     marker=dict(size=8, color='#111111', symbol='x'),
                                     visible='legendonly'), row=1, col=1, secondary_y=False)
    if show_experimental_debug_traces and 'TCN_State' in group.columns:
        tcn_points = group[group['TCN_State'].isin(['REFUEL', 'DRAIN', 'SPIKE', 'TRANSIENT_NOISE', 'TRANSIENT_UP_NOISE', 'TRANSIENT_DOWN_NOISE', 'TRANSIENT_CLUSTER_NOISE', 'SLOSHING_NOISE', 'CONSUMPTION'])]
        if not tcn_points.empty:
            fig.add_trace(go.Scatter(x=tcn_points['FuelTime'], y=tcn_points['FuelLevel'],
                                     mode='markers', name='TCN state markers', legendgroup='tcn_state_markers',
                                     showlegend=show_legend,
                                     text=tcn_points['TCN_State'] + ' conf=' + tcn_points['TCN_Confidence'].round(2).astype(str),
                                     marker=dict(size=7, color='#8A2BE2', symbol='diamond-open'),
                                     visible='legendonly'), row=1, col=1, secondary_y=False)

    show_legend = False

# Highlight anomalous Deltas
anomalies = df_seg[df_seg['FlagLargeDelta'] == 1]
if not anomalies.empty:
    fig.add_trace(go.Scatter(x=anomalies['FuelTime'], y=anomalies['FuelLevel'],
                             mode='markers', name='Sự kiện Sụt giảm mạnh',
                             visible='legendonly', # Ẩn mặc định, ấn vào legend mới hiện
                             marker=dict(color='red', size=10, symbol='x', line=dict(width=2, color='black'))), row=1, col=1, secondary_y=False)

if 'Speed' in df_seg.columns:
    # Overlay speed for TOOLTIP ONLY (transparent line, no fill, no legend)
    fig.add_trace(go.Scatter(x=df_seg['FuelTime'], y=df_seg['Speed'], 
                             mode='lines', name='Speed (km/h)', 
                             line=dict(color='rgba(0,0,0,0)', width=0),
                             showlegend=False), row=1, col=1, secondary_y=True)

# 2. Speed Plot (Dedicated)
if 'Speed' in df_seg.columns:
    fig.add_trace(go.Scatter(x=df_seg['FuelTime'], y=df_seg['Speed'], 
                             mode='lines', name='Speed (km/h)', 
                             line=dict(color='#636EFA', width=1.5),
                             fill='tozeroy', fillcolor='rgba(99, 110, 250, 0.1)'), row=2, col=1)

# 3. Rolling Std Plot
if 'RollingStd' in df_seg.columns:
    fig.add_trace(go.Scattergl(x=df_seg['FuelTime'], y=df_seg['RollingStd'], 
                             mode='lines', name='Rolling Std (Noise)', 
                             line=dict(color='#EF553B', width=1.5),
                             fill='tozeroy', fillcolor='rgba(239, 85, 59, 0.1)'), row=3, col=1)

# Update layout for premium light aesthetic
fig.update_layout(
    height=850,
    template="plotly_white",
    hovermode="x unified",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    margin=dict(l=20, r=20, t=60, b=20)
)

fig.update_yaxes(title_text="Lít", row=1, col=1)
fig.update_yaxes(title_text="km/h", row=2, col=1)
fig.update_yaxes(title_text="Std (Lít)", row=3, col=1)

fig.update_traces(
    hovertemplate="%{x}<br>%{fullData.name}: %{y:.1f}<extra></extra>",
    selector=dict(type="scatter"),
)
fig.update_traces(
    hovertemplate="%{x}<br>%{fullData.name}: %{y:.1f}<extra></extra>",
    selector=dict(type="scattergl"),
)

st.plotly_chart(fig, width='stretch')

with st.expander("🔍 Soi Chi tiết Nhãn AI từng mốc thời gian (Data Inspector)", expanded=True):
    st.markdown("Bảng tra cứu nhãn do Random Forest dự đoán trên từng điểm dữ liệu của phân đoạn đang chọn:")
    cols_to_show = ['FuelTime', 'FuelLevel', 'Speed', 'AI_State', 'AI_State_Raw', 'AI_State_Confidence', 'QualityReason']
    cols_to_show = [c for c in cols_to_show if c in df_seg.columns]
    st.dataframe(df_seg[cols_to_show], height=350, use_container_width=True)

st.info("💡 **Hướng dẫn:** Các thuật toán lọc thích nghi và AI hiện đã được đồng bộ nguồn dữ liệu làm phẳng sóng nhiễu transient. Sử dụng bảng Data Inspector ở trên để kiểm tra chính xác nhãn AI do mô hình Random Forest phân loại trên từng mốc thời gian!")
