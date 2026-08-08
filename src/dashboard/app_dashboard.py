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
from src.core.filters import moving_average as ma
from src.core.filters import median_filter as mf
import src.core.filters.kalman_adaptive as ka
from src.core.filters.kalman_adaptive import BoLocKalmanThichNghi1D, is_valid_measurement
from src.core.filters.kalman_ml import BoLocKalmanAI, predict_batch
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

st.title("🔬 Đấu Trường Thuật Toán: Lọc Nhiễu Nhiên Liệu")
st.markdown("**Trực quan hóa và so sánh hiệu năng 3 thuật toán: Moving Average, Median Filter và Standard Kalman Filter**")

@st.cache_data
def load_data(car_id):
    # Dữ liệu đã qua tiền xử lý, bao gồm các Flag và SegmentID
    file_path = f"data/processed/CarFuelHistory_Processed_Car{car_id}.csv"
    if not os.path.exists(file_path):
        st.error(f"Không tìm thấy file dữ liệu: {file_path}")
        return pd.DataFrame()
    df = pd.read_csv(file_path)
    df['FuelTime'] = pd.to_datetime(df['FuelTime'], errors="coerce")
    df['FuelLevel'] = pd.to_numeric(df['FuelLevel'], errors="coerce")
    df['SegmentID'] = pd.to_numeric(df['SegmentID'], errors="coerce")
    return df

import glob

# Get available cars
csv_files = glob.glob("data/processed/CarFuelHistory_Processed_*.csv")
if not csv_files:
    st.error("Không tìm thấy dữ liệu đã xử lý. Vui lòng chạy file preprocess_pipeline.py trước.")
    st.stop()

cars = [os.path.basename(f).replace("CarFuelHistory_Processed_Car", "").replace(".csv", "") for f in csv_files]

with st.sidebar:
    st.header("⚙️ Cấu hình Dữ liệu")
    selected_car = st.selectbox("🚗 Chọn Xe (Vehicle ID)", cars)
    
df = load_data(selected_car)

min_date = df['FuelTime'].min().date()
max_date = df['FuelTime'].max().date()

with st.sidebar:
    st.markdown("---")
    st.header("📅 Bộ lọc Thời gian")
    
    # Check if car changed, if so, reset the date range
    if "last_car" not in st.session_state or st.session_state.last_car != selected_car:
        st.session_state.date_range_key = (min_date, max_date)
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

# Áp dụng filter ngày
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
    st.header("📏 Chọn Phân Đoạn")
    segment_options = ["Toàn bộ dữ liệu trong khoảng thời gian (All)"] + list(df['SegmentID'].dropna().unique())
    selected_segment = st.selectbox("Hiển thị theo phân đoạn", segment_options)
    
    st.markdown("---")
    st.header("🎛️ Tinh chỉnh Thuật toán")
    ma_window = st.slider("Cửa sổ Moving Average (N)", min_value=3, max_value=50, value=10, step=1)
    med_window = st.slider("Cửa sổ Median Filter (N)", min_value=3, max_value=50, value=10, step=1)
    kalman_r = st.slider("Nhiễu đo lường Kalman (R)", min_value=1, max_value=1000, value=9, step=1)
    
    st.subheader("🤖 Cấu hình Adaptive Kalman")
    adapt_threshold = st.slider("Ngưỡng bắt nhảy (Threshold) - Lít", min_value=1.0, max_value=20.0, value=10.0, step=0.5, help="Chênh lệch tối thiểu để bắt đầu nghi ngờ có bơm/rút xăng")
    adapt_persistence = st.slider("Số nhịp chờ xác nhận (Persistence)", min_value=1, max_value=15, value=3, step=1, help="Số chu kỳ tín hiệu phải duy trì ở mức cao/thấp để xác nhận là đổ/rút thật (lọc đỉnh nhiễu)")

# Filter data
if selected_segment == "Toàn bộ dữ liệu trong khoảng thời gian (All)":
    df_seg = df.copy()
else:
    df_seg = df[df['SegmentID'] == selected_segment].copy()

# Sắp xếp thời gian an toàn theo chuẩn Reviewer
df_seg["_OriginalOrder"] = np.arange(len(df_seg))
df_seg = df_seg.sort_values(["SegmentID", "FuelTime", "_OriginalOrder"], kind="stable")

# Apply Custom Algorithms
with st.spinner("Đang chạy 3 thuật toán lọc nhiễu..."):
    df_seg['Custom_MA'] = np.nan
    df_seg['Custom_Median'] = np.nan
    df_seg['Custom_Kalman'] = np.nan
    df_seg['Custom_Adaptive_Kalman'] = np.nan
    df_seg['Custom_ML_Kalman'] = np.nan
    
    # Tiền xử lý: Tính Gia tốc (Acceleration) nếu chưa có
    if "Acceleration" not in df_seg.columns:
        if "Speed" in df_seg.columns:
            time_gap_sec = df_seg["TimeGapMinutes"].fillna(5.0) * 60.0
            safe_time_gap = time_gap_sec.replace(0, 1.0)
            df_seg["Acceleration"] = df_seg.groupby("SegmentID", dropna=False)["Speed"].diff().fillna(0.0) / safe_time_gap
        else:
            df_seg["Acceleration"] = 0.0

    # Chạy trên từng Segment, gán bằng Index để tránh xô lệch dòng
    for seg_id, group in df_seg.groupby('SegmentID', sort=False, dropna=False):
        fuels = group['FuelLevel'].tolist()
        
        # MA & Median (Imported from our robust modules)
        df_seg.loc[group.index, 'Custom_MA'] = ma.tinh_loc_trung_binh_dong(fuels, ma_window)
        df_seg.loc[group.index, 'Custom_Median'] = mf.tinh_loc_trung_vi(fuels, med_window)
        
        kf_std = None
        kf_adapt = None
        kf_ml = None
        
        kalman_std_vals = []
        kalman_adapt_vals = []
        kalman_ml_vals = []
        
        reference_gap = 5.0
        khoang_thoi_gian_tich_luy_std = 0.0
        khoang_thoi_gian_tich_luy_adapt = 0.0
        
        # Batch predict cho ML
        ml_labels = predict_batch(group)
        
        for i_loc, dong in enumerate(group.itertuples()):
            idx = dong.Index
            ml_lbl = ml_labels[i_loc]
            gap = getattr(dong, "TimeGapMinutes", reference_gap)
            if pd.isna(gap) or gap <= 0: gap = reference_gap
            
            # Feature logic
            movement_state = 0 if str(getattr(dong, "MovementState", "Moving")).strip().upper() == "STOPPED" else 1
            acceleration = float(getattr(dong, "Acceleration", 0.0))
            
            if not is_valid_measurement(getattr(dong, 'FuelLevel', None), getattr(dong, 'FeatureStatus', '')):
                khoang_thoi_gian_tich_luy_std += gap
                khoang_thoi_gian_tich_luy_adapt += gap
                kalman_std_vals.append(np.nan)
                kalman_adapt_vals.append(np.nan)
                kalman_ml_vals.append(np.nan)
                continue
            
            measurement = float(dong.FuelLevel)
            
            # Kalman Standard
            khoang_thoi_gian_std = gap + khoang_thoi_gian_tich_luy_std
            khoang_thoi_gian_tich_luy_std = 0.0
            dt_ratio = khoang_thoi_gian_std / reference_gap
            
            if kf_std is None:
                kf_std = kalman.BoLocKalmanTieuChuan1D(trang_thai_ban_dau=measurement, nhieu_qua_trinh=1.0, nhieu_do_luong=kalman_r)
                kalman_std_vals.append(measurement)
            else:
                kalman_std_vals.append(kf_std.cap_nhat(measurement, ty_le_dt=dt_ratio))
                
            # Kalman Adaptive
            khoang_thoi_gian_adapt = gap + khoang_thoi_gian_tich_luy_adapt
            khoang_thoi_gian_tich_luy_adapt = 0.0
            if kf_adapt is None:
                kf_adapt = BoLocKalmanThichNghi1D(trang_thai_ban_dau=measurement, sai_so_uoc_luong_ban_dau=9.0, nhieu_qua_trinh=1.0, r_co_ban=kalman_r, nguong_bat_nhay=adapt_threshold, nhip_cho_xac_nhan=adapt_persistence)
                kalman_adapt_vals.append(measurement)
            else:
                kalman_adapt_vals.append(kf_adapt.cap_nhat(measurement, ty_le_dt=khoang_thoi_gian_adapt/reference_gap, trang_thai_chuyen_dong=movement_state, gia_toc=acceleration))
                
            # Kalman ML
            if kf_ml is None:
                kf_ml = BoLocKalmanAI(trang_thai_ban_dau=measurement, r_co_ban=kalman_r)
                kalman_ml_vals.append(measurement)
            else:
                kalman_ml_vals.append(kf_ml.cap_nhat(measurement, ty_le_dt=dt_ratio, nhan_ai=ml_lbl))
                    
        df_seg.loc[group.index, 'Custom_Kalman'] = kalman_std_vals
        df_seg.loc[group.index, 'Custom_Adaptive_Kalman'] = kalman_adapt_vals
        df_seg.loc[group.index, 'Custom_ML_Kalman'] = kalman_ml_vals

# Restore original order just in case
df_seg = df_seg.sort_values("_OriginalOrder", kind="stable").drop(columns="_OriginalOrder")

# Insert NaN rows between segments to prevent Plotly from connecting lines across huge time gaps
if selected_segment == "Toàn bộ dữ liệu trong khoảng thời gian (All)":
    mask = df_seg['SegmentID'] != df_seg['SegmentID'].shift(1)
    mask.iloc[0] = False
    if mask.any():
        nan_rows = df_seg[mask].copy()
        for col in nan_rows.columns:
            if col != 'FuelTime':
                nan_rows[col] = np.nan
        # Shift time slightly so it inserts perfectly between the two segments
        nan_rows['FuelTime'] = nan_rows['FuelTime'] - pd.Timedelta(seconds=1)
        df_seg = pd.concat([df_seg, nan_rows]).sort_values("FuelTime")

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
for seg_id, group in df_seg.groupby('SegmentID', sort=False):
    # Raw Fuel
    fig.add_trace(go.Scattergl(x=group['FuelTime'], y=group['FuelLevel'], 
                             mode='lines+markers', name='Raw FuelLevel', legendgroup='raw', showlegend=show_legend,
                             line=dict(color='rgba(255, 0, 0, 0.6)', width=1.5),
                             marker=dict(size=4, color='red')), row=1, col=1, secondary_y=False)
                             
    # Moving Average
    fig.add_trace(go.Scattergl(x=group['FuelTime'], y=group['Custom_MA'], 
                             mode='lines', name=f'Moving Average (N={ma_window})', legendgroup='ma', showlegend=show_legend,
                             line=dict(color='#FFA15A', width=2), visible='legendonly', connectgaps=True), row=1, col=1, secondary_y=False)
                             
    # Median Filter
    fig.add_trace(go.Scattergl(x=group['FuelTime'], y=group['Custom_Median'], 
                             mode='lines', name=f'Median Filter (N={med_window})', legendgroup='med', showlegend=show_legend,
                             line=dict(color='#AB63FA', width=2), visible='legendonly', connectgaps=True), row=1, col=1, secondary_y=False)
                             
    # Kalman Standard
    fig.add_trace(go.Scattergl(x=group['FuelTime'], y=group['Custom_Kalman'], 
                             mode='lines', name=f'Kalman Filter (R={kalman_r})', legendgroup='kalman', showlegend=show_legend,
                             line=dict(color='#00CC96', width=2), connectgaps=True), row=1, col=1, secondary_y=False)
                             
    # Kalman Adaptive
    fig.add_trace(go.Scattergl(x=group['FuelTime'], y=group['Custom_Adaptive_Kalman'], 
                             mode='lines', name=f'Adaptive Kalman (Dynamic R)', legendgroup='adapt', showlegend=show_legend,
                             line=dict(color='blue', width=2, dash='dash'), connectgaps=True), row=1, col=1, secondary_y=False)
                             
    # Kalman ML
    fig.add_trace(go.Scattergl(x=group['FuelTime'], y=group['Custom_ML_Kalman'], 
                             mode='lines', name=f'Kalman Filter (AI/ML)', legendgroup='ml', showlegend=show_legend,
                             line=dict(color='deeppink', width=3), visible='legendonly', connectgaps=True), row=1, col=1, secondary_y=False)
                             
    show_legend = False

# Highlight anomalous Deltas
anomalies = df_seg[df_seg['FlagLargeDelta'] == 1]
if not anomalies.empty:
    fig.add_trace(go.Scattergl(x=anomalies['FuelTime'], y=anomalies['FuelLevel'],
                             mode='markers', name='Sự kiện Sụt giảm mạnh',
                             visible='legendonly', # Ẩn mặc định, ấn vào legend mới hiện
                             marker=dict(color='red', size=10, symbol='x', line=dict(width=2, color='black'))), row=1, col=1, secondary_y=False)

if 'Speed' in df_seg.columns:
    # Overlay speed for TOOLTIP ONLY (transparent line, no fill, no legend)
    fig.add_trace(go.Scattergl(x=df_seg['FuelTime'], y=df_seg['Speed'], 
                             mode='lines', name='Speed (km/h)', 
                             line=dict(color='rgba(0,0,0,0)', width=0),
                             showlegend=False), row=1, col=1, secondary_y=True)

# 2. Speed Plot (Dedicated)
if 'Speed' in df_seg.columns:
    fig.add_trace(go.Scattergl(x=df_seg['FuelTime'], y=df_seg['Speed'], 
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

st.plotly_chart(fig, use_container_width=True)

st.info("💡 **Hướng dẫn:** Các thuật toán Moving Average và Median Filter hiện đã được cập nhật logic an toàn (Reset khi mất sóng) và loại bỏ khoảng Burn-in (Chỉ vẽ đường thẳng khi đã gom đủ dữ liệu). Hãy thử kéo thanh trượt N để thấy sự khác biệt!")
