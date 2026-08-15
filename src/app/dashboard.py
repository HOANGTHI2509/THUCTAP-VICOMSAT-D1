import streamlit as st
import pandas as pd
import numpy as np
import time
import glob
import os
import torch
import sys

# Add root directory to path so imports work
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.pipeline.realtime_inference import RealtimeFuelFilter
from src.core.filters.kalman_traditional import BoLocKalmanTieuChuan1D
from src.core.filters.kalman_adaptive import BoLocKalmanThichNghi1D

# Page Config
st.set_page_config(page_title="Vcomsat Fuel Tracker", layout="wide")
st.title("🛰️ Vcomsat Real-time Fuel Tracking")
st.markdown("Hệ thống Lọc nhiễu nhiên liệu Thời gian thực (Real-time Streaming Simulator)")

# Data Loading Cache
@st.cache_data
def get_unseen_vehicles():
    all_files = glob.glob('data/processed/CarFuelHistory_Processed_*.csv')
    seen_cars = ['Car1.csv', 'Car2.csv', 'Car3.csv', 'Car4.csv', 'Car5.csv']
    unseen_files = [f for f in all_files if not any(sc in f for sc in seen_cars)]
    return unseen_files

unseen_files = get_unseen_vehicles()
if not unseen_files:
    st.error("Không tìm thấy dữ liệu xe Unseen!")
    st.stop()

# Select Vehicle
vehicle_names = [os.path.basename(f).replace('CarFuelHistory_Processed_', '').replace('.csv', '') for f in unseen_files]
selected_vehicle = st.selectbox("Vehicle", vehicle_names)
selected_file = unseen_files[vehicle_names.index(selected_vehicle)]

# Load DataFrame
@st.cache_data
def load_data(file_path):
    df = pd.read_csv(file_path)
    return df

df = load_data(selected_file)

# Control Buttons
col1, col2, col3 = st.columns(3)
with col1:
    start_btn = st.button("▶ Start Streaming")
with col2:
    stop_btn = st.button("■ Stop Streaming")
with col3:
    sim_interval = st.slider("Simulation Interval (ms)", 10, 500, 50)

# Layout: Chart + Stats
st.markdown("### Live Fuel Signal")
chart_placeholder = st.empty()
stats_placeholder = st.empty()

# Initialization for streaming
if "is_streaming" not in st.session_state:
    st.session_state.is_streaming = False

if start_btn:
    st.session_state.is_streaming = True
if stop_btn:
    st.session_state.is_streaming = False

if st.session_state.is_streaming:
    # Initialize filters
    rt_gru = RealtimeFuelFilter(model_path='models/gru/best_gru_final.pth', window_size=30)
    
    # Initialize Kalman filters with the first data point
    initial_fuel = df['FuelLevel'].iloc[0]
    kf_std = BoLocKalmanTieuChuan1D(trang_thai_ban_dau=initial_fuel)
    kf_adp = BoLocKalmanThichNghi1D(trang_thai_ban_dau=initial_fuel)
    
    # Prepare empty lists to collect data for chart plotting
    plot_data = pd.DataFrame(columns=['Raw Fuel', 'Standard Kalman', 'Adaptive Kalman', 'Time-aware GRU'])
    
    st.write(f"Đang giả lập nhận dữ liệu thời gian thực từ `{selected_vehicle}`...")
    
    # Iterate row by row (simulating sensor stream)
    for i, row in df.iterrows():
        if not st.session_state.is_streaming:
            break
            
        t_start = time.perf_counter()
        
        raw_fuel = row['FuelLevel']
        speed = row['Speed']
        mov = row['MovementState']
        tg = row.get('TimeGapMinutes', 1.0)
        if pd.isna(tg): tg = 1.0
        
        # 1. Run Standard Kalman
        kf_std.cap_nhat(raw_fuel, ty_le_dt=tg)
        std_val = kf_std.x
        
        # 2. Run Adaptive Kalman
        accel = row.get('Acceleration', 0.0) if not pd.isna(row.get('Acceleration')) else 0.0
        kf_adp.cap_nhat(
            gia_tri_do=raw_fuel, 
            ty_le_dt=tg, 
            trang_thai_chuyen_dong=1 if mov == 'Moving' else 0,
            gia_toc=accel
        )
        adp_val = kf_adp.x
        
        # 3. Run GRU
        gru_val = rt_gru.push_sensor_data(
            timestamp=row.get('LogTime', i),
            fuel=raw_fuel,
            speed=speed,
            movement_state=mov,
            time_gap_minutes=tg
        )
        
        inference_time_ms = (time.perf_counter() - t_start) * 1000
        
        # Add to plot data
        new_row = pd.DataFrame({
            'Raw Fuel': [raw_fuel],
            'Standard Kalman': [std_val],
            'Adaptive Kalman': [adp_val],
            'Time-aware GRU': [gru_val]
        })
        
        # Using Streamlit's native add_rows for efficient appending
        if i == 0:
            chart = chart_placeholder.line_chart(new_row, color=["#000000", "#1f77b4", "#ff7f0e", "#d62728"])
        else:
            chart.add_rows(new_row)
            
        stats_placeholder.markdown(f"""
        **Status:** 🟢 RUNNING  
        **Current Fuel (GRU):** `{gru_val:.2f} L`  
        **Simulation interval:** `{sim_interval} ms`  
        **Model inference:** `{inference_time_ms:.2f} ms`  
        """)
        
        time.sleep(sim_interval / 1000.0)
else:
    st.info("Nhấn 'Start Streaming' để bắt đầu luồng dữ liệu giả lập.")
