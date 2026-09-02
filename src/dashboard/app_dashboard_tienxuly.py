# Trình điều khiển Dashboard Trực quan hóa Dữ liệu và Thuật toán
import os
import sys
import glob
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

# Ensure the root directory is in the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

# Custom Imports (Chỉ 2 thuật toán: Kalman truyền thống và AI-Enhanced Kalman Realtime)
from src.core.filters import kalman_traditional as kalman
from src.core.filters.ai_enhanced_adaptive_realtime import filter_ai_enhanced_adaptive_realtime
from src.core.filters.ai_state_filter import filter_with_ai_state, load_fuel_state_classifier
from src.core.filters.anomaly_detector import FuelAnomalyDetector


def is_valid_measurement(val, feature_status=""):
    """Kiểm tra tính hợp lệ của điểm đo nhiên liệu."""
    if val is None or pd.isna(val):
        return False
    if str(feature_status).strip().upper() == "INVALID":
        return False
    return True


@st.cache_resource
def load_ai_state_model():
    return load_fuel_state_classifier("models/fuel_state_classifier")


ai_state_model, ai_state_metadata = load_ai_state_model()

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
st.markdown("**Trực quan hóa đối chứng: Mức thô (Raw), Kalman truyền thống, và AI-Enhanced Kalman (Realtime / Causal)**")

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

def build_data_sources():
    sources = []
    for file_path in sorted(glob.glob("TienXuLy/*_processed.csv")):
        car_id = os.path.basename(file_path).replace("_processed.csv", "")
        sources.append(
            {
                "label": car_id,
                "car_id": car_id,
                "path": file_path,
                "source": "TienXuLy",
            }
        )
    return sources

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
# Keep the unfiltered rows for causal-filter burn-in.  The date picker below
# controls what is displayed, not the state history needed to process it.
df_all = df.copy()

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
    ai_filter_mode = st.radio("Chế độ:", ["realtime", "offline"], index=0, help="realtime: thuần nhân quả (causal) không nhìn tương lai. offline: tham chiếu nhìn tương lai.")
    st.markdown("---")
    st.header("📏 Chọn Phân Đoạn")
    segment_options = ["Toàn bộ dữ liệu trong khoảng thời gian (All)"] + list(df['SegmentID'].dropna().unique())
    selected_segment = st.selectbox("Hiển thị theo phân đoạn", segment_options)

if selected_segment == "Toàn bộ dữ liệu trong khoảng thời gian (All)":
    df_visible = df.copy()
else:
    df_visible = df[df['SegmentID'] == selected_segment].copy()

# A realtime filter must be warmed up with points before the selected range.
# Keep a bounded history per segment: processing the vehicle's *entire*
# history here makes every dashboard interaction unnecessarily expensive.
# The final valid point is also retained when a dropout began more than the
# burn-in window ago, so a 0L window still has a level to hold.
display_indices = df_visible.index
visible_segment_ids = df_visible['SegmentID'].dropna().unique()
display_start_time = df_visible['FuelTime'].min()
df_history = df_all[
    df_all['SegmentID'].isin(visible_segment_ids)
    & (df_all['FuelTime'] < display_start_time)
].copy()
df_history = df_history.sort_values(['SegmentID', 'FuelTime'], kind='stable')
burn_in_rows = 120
burn_in_tail = df_history.groupby('SegmentID', sort=False, dropna=False).tail(burn_in_rows)
burn_in_last_valid = (
    df_history[pd.to_numeric(df_history['FuelLevel'], errors='coerce') > 5.0]
    .groupby('SegmentID', sort=False, dropna=False)
    .tail(1)
)
df_burn_in = pd.concat([burn_in_tail, burn_in_last_valid]).loc[lambda frame: ~frame.index.duplicated(keep='last')]
df_seg = pd.concat([df_burn_in, df_visible]).sort_values(['SegmentID', 'FuelTime'], kind='stable').copy()
df_seg['_DisplayRow'] = df_seg.index.isin(display_indices)

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
        "source_col": "FuelLevel" if ai_filter_mode == "realtime" else "CleanedFuel",
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
    df_seg['AI_Enhanced_Kalman_Realtime'] = np.nan
    
    def apply_by_segment(frame, operation):
        """Run a transform per segment without losing SegmentID on pandas 2.x/3.x."""
        segment_by_row = frame["SegmentID"].copy()
        result = (
            frame.groupby("SegmentID", sort=False, dropna=False, group_keys=False)
            .apply(operation)
        )
        if "SegmentID" not in result.columns:
            result["SegmentID"] = segment_by_row.reindex(result.index).fillna(0)
        return result

    # 1. Tiền xử lý dị thường vật lý
    if ai_filter_mode == "offline":
        anomaly_detector = FuelAnomalyDetector(capacity=estimated_capacity)
        df_seg = apply_by_segment(df_seg, lambda g: anomaly_detector.detect_and_clean(g))
    else:
        df_seg["CleanedFuel"] = df_seg["FuelLevel"]
        df_seg["FuelAnomalyType"] = "NORMAL"
    
    if "Acceleration" not in df_seg.columns:
        if "Speed" in df_seg.columns:
            time_gap_sec = df_seg["TimeGapMinutes"].fillna(5.0) * 60.0
            safe_time_gap = time_gap_sec.replace(0, 1.0)
            df_seg["Acceleration"] = df_seg.groupby("SegmentID", dropna=False)["Speed"].diff().fillna(0.0) / safe_time_gap
        else:
            df_seg["Acceleration"] = 0.0

    # 2. Trích đặc trưng & Phân loại trạng thái AI
    if ai_state_model is not None and ai_state_metadata is not None:
        df_seg = apply_by_segment(
            df_seg,
            lambda group: filter_with_ai_state(
                group,
                model=ai_state_model,
                metadata=ai_state_metadata,
                profile=vehicle_profile,
                mode=ai_filter_mode,
            ),
        )
    else:
        df_seg["AI_State"] = "MODEL_NOT_FOUND"
        df_seg["AI_State_Raw"] = "MODEL_NOT_FOUND"
        df_seg["AI_State_Confidence"] = np.nan
        df_seg["AI_CleanInput"] = np.nan

    # 3. Chạy 2 bộ lọc: Kalman Truyền Thống & AI-Enhanced Kalman Realtime
    for seg_id, group in df_seg.groupby('SegmentID', sort=False, dropna=False):
        valid_gaps = group['TimeGapMinutes'].dropna()
        reference_gap = valid_gaps[valid_gaps > 0].median()
        if pd.isna(reference_gap) or reference_gap <= 0:
            reference_gap = 5.0
        
        # 3.1. Kalman Filter Truyền Thống
        kf_std = None
        kalman_std_vals = []
        khoang_thoi_gian_tich_luy_std = 0.0
        
        for dong in group.itertuples():
            gap = getattr(dong, "TimeGapMinutes", reference_gap)
            if pd.isna(gap) or gap <= 0:
                gap = reference_gap
            
            measurement = float(getattr(dong, 'FuelLevel', np.nan))
            if not is_valid_measurement(measurement, getattr(dong, 'FeatureStatus', '')):
                khoang_thoi_gian_tich_luy_std += gap
                kalman_std_vals.append(np.nan)
                continue
            
            khoang_thoi_gian_std = gap + khoang_thoi_gian_tich_luy_std
            khoang_thoi_gian_tich_luy_std = 0.0
            dt_ratio = khoang_thoi_gian_std / reference_gap
            
            if kf_std is None:
                kf_std = kalman.BoLocKalmanTieuChuan1D(trang_thai_ban_dau=measurement, nhieu_qua_trinh=1.0, nhieu_do_luong=kalman_r)
                kalman_std_vals.append(measurement)
            else:
                kalman_std_vals.append(kf_std.cap_nhat(measurement, ty_le_dt=dt_ratio))
        
        df_seg.loc[group.index, 'Custom_Kalman'] = kalman_std_vals

        # 3.2. AI-Enhanced Kalman (Realtime / Causal)
        df_seg_subset = df_seg.loc[group.index]
        df_seg.loc[group.index, 'AI_Enhanced_Kalman_Realtime'] = filter_ai_enhanced_adaptive_realtime(df_seg_subset, config=ai_kalman_config)
# Restore original order just in case
df_seg = df_seg.sort_values("_OriginalOrder", kind="stable").drop(columns="_OriginalOrder")
# Hide burn-in context after every derived column has been calculated.
df_seg = df_seg[df_seg['_DisplayRow']].copy().drop(columns="_DisplayRow")

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
                    subplot_titles=("1. Đối chứng: Mức Nhiên Liệu Gốc vs Kalman Truyền Thống & AI-Enhanced Realtime", "2. Tốc độ di chuyển", "3. Độ nhiễu cục bộ (Rolling Std)"),
                    row_heights=[0.6, 0.2, 0.2])

# 1. Fuel Plot & Custom Algorithm Traces
show_legend = True
for seg_id, group in df_seg.groupby('SegmentID', sort=False):
    # 1. Raw Fuel (Dữ liệu gốc)
    fig.add_trace(go.Scatter(
        x=group['FuelTime'], y=group['FuelLevel'], 
        mode='lines+markers', name='Raw FuelLevel (Thô)', legendgroup='raw', showlegend=show_legend,
        line=dict(color='red', width=1), marker=dict(size=4), connectgaps=True
    ), row=1, col=1, secondary_y=False)
                             
    # 2. Kalman Filter Truyen Thong
    fig.add_trace(go.Scatter(
        x=group['FuelTime'], y=group['Custom_Kalman'], 
        mode='lines', name=f'Kalman Filter Truyền Thống (R={kalman_r})', legendgroup='kalman', showlegend=show_legend,
        line=dict(color='#00CC96', width=2), connectgaps=True
    ), row=1, col=1, secondary_y=False)

    # 3. AI-Enhanced Kalman Realtime / Causal (Thuật toán chính)
    if 'AI_Enhanced_Kalman_Realtime' in group.columns:
        fig.add_trace(go.Scatter(
            x=group['FuelTime'], y=group['AI_Enhanced_Kalman_Realtime'],
            mode='lines', name='AI-Enhanced Kalman (Realtime / Causal)',
            legendgroup='ai_enhanced_kalman_realtime', showlegend=show_legend,
            line=dict(color='#FF7F0E', width=2.5), connectgaps=True
        ), row=1, col=1, secondary_y=False)

    show_legend = False

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
    st.dataframe(df_seg[cols_to_show], height=350, width='stretch')

st.info("💡 **Hướng dẫn:** Các thuật toán lọc thích nghi và AI hiện đã được đồng bộ nguồn dữ liệu làm phẳng sóng nhiễu transient. Sử dụng bảng Data Inspector ở trên để kiểm tra chính xác nhãn AI do mô hình Random Forest phân loại trên từng mốc thời gian!")
