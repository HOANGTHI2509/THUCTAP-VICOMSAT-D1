import os
import torch
import numpy as np
import pandas as pd
from collections import deque

from src.models.time_aware_gru import FuelTimeAwareGRU
from src.pipeline.build_gru_dataset import process_timegap

class RealtimeFuelFilter:
    def __init__(self, model_path='models/gru/best_gru_final.pth', window_size=30, max_gap_minutes=15.0):
        self.window_size = window_size
        self.max_gap_minutes = max_gap_minutes
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = FuelTimeAwareGRU(input_dim=6, hidden_dim=64).to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        
        # State buffers
        self.buffer = deque(maxlen=window_size)
        self.fuel_buffer_5 = deque(maxlen=5) # For RollingStd
        self.last_time = None
        self.last_speed = None
        
    def reset(self):
        """Xóa toàn bộ state (dùng khi Gap quá lớn hoặc khởi động lại xe)"""
        self.buffer.clear()
        self.fuel_buffer_5.clear()
        self.last_time = None
        self.last_speed = None

    def push_sensor_data(self, timestamp, fuel, speed, movement_state, time_gap_minutes=None, accel_override=None, rstd_override=None):
        """
        Nhận 1 điểm dữ liệu mới từ Cảm biến và trả về Clean Fuel.
        Nếu chưa đủ window_size, model có thể vẫn trả về Fuel hiện tại (bỏ qua filter).
        """
        # Nếu Gap quá lớn -> Khởi động segment mới
        if time_gap_minutes is not None and time_gap_minutes > self.max_gap_minutes:
            self.reset()
            
        # Tính gia tốc (Accel)
        if accel_override is not None:
            accel = accel_override
        elif self.last_speed is not None and time_gap_minutes is not None and time_gap_minutes > 0:
            accel = (speed - self.last_speed) / max(time_gap_minutes, 0.01)
        else:
            accel = 0.0
            
        self.fuel_buffer_5.append(fuel)
        if rstd_override is not None:
            rstd = rstd_override
        else:
            rstd = np.std(self.fuel_buffer_5, ddof=1) if len(self.fuel_buffer_5) > 1 else 0.0
            
        mov = 1.0 if movement_state == 'Moving' else 0.0
        tg = time_gap_minutes if time_gap_minutes is not None else 1.0
        tg_clipped = process_timegap(tg)
        
        # Lưu vào buffer (Fuel, Speed, Accel, Mov, Rstd, Tg)
        self.buffer.append([fuel, speed, accel, mov, rstd, tg_clipped])
        
        self.last_time = timestamp
        self.last_speed = speed
        
        # Inference
        if len(self.buffer) < self.window_size:
            return fuel # Trả về giá trị thô nếu chưa đủ cửa sổ để suy diễn
            
        # Build tensor
        # Anchor là điểm đầu tiên trong cửa sổ
        arr = np.array(self.buffer)
        anchor = arr[0, 0]
        
        # w_fuel_res = noisy - anchor
        arr[:, 0] = arr[:, 0] - anchor
        
        X = np.expand_dims(arr, axis=0) # Shape: (1, 30, 6)
        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        
        with torch.no_grad():
            res_pred = self.model(X_tensor).item()
            
        clean_fuel = anchor + res_pred
        return clean_fuel

class StreamingSimulator:
    def __init__(self, df, filter_instance):
        self.df = df
        self.filter = filter_instance
        
    def simulate(self):
        clean_preds = []
        
        for i, row in self.df.iterrows():
            tg = row.get('TimeGapMinutes', 5.0)
            if pd.isna(tg): tg = 5.0
            
            accel = row.get('Acceleration', 0.0) if not pd.isna(row.get('Acceleration')) else 0.0
            rstd = self.df['FuelLevel'].rolling(window=5, min_periods=1).std().fillna(0).values[i]
            
            clean_val = self.filter.push_sensor_data(
                timestamp=row['LogTime'] if 'LogTime' in row else i,
                fuel=row['FuelLevel'],
                speed=row['Speed'],
                movement_state=row['MovementState'],
                time_gap_minutes=tg,
                accel_override=accel,
                rstd_override=rstd
            )
            
            clean_preds.append(clean_val)
            
        return clean_preds

def test_streaming_vs_batch():
    from src.pipeline.evaluate_real_vcomsat import run_gru
    
    print("\n[Simulator] Testing Streaming vs Batch Tolerance...")
    df = pd.read_csv('data/processed/CarFuelHistory_Processed_36C-31893.csv')
    df = df.iloc[:500].copy()
    
    # 1. Batch Inference
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = FuelTimeAwareGRU(input_dim=6, hidden_dim=64).to(device)
    model.load_state_dict(torch.load('models/gru/best_gru_final.pth', map_location=device))
    model.eval()
    
    batch_preds = run_gru(df, model, N=30)
    
    # 2. Streaming Inference
    rt_filter = RealtimeFuelFilter(window_size=30)
    simulator = StreamingSimulator(df, rt_filter)
    stream_preds = simulator.simulate()
    
    # Compare from index 29 onwards (where window is full)
    batch_valid = batch_preds[29:]
    stream_valid = stream_preds[29:]
    
    # Debug: Print the 30th element's features in batch vs stream
    print(f"Batch prediction at idx 29: {batch_preds[29]}")
    print(f"Stream prediction at idx 29: {stream_preds[29]}")
    
    errs = np.abs(np.array(batch_valid) - np.array(stream_valid))
    max_err = np.max(errs)
    max_idx = np.argmax(errs)
    print(f"Max Absolute Error giữa Streaming và Batch: {max_err} at array index {max_idx} (DataFrame index {max_idx + 29})")
    print(f"Values at max error -> Batch: {batch_valid[max_idx]}, Stream: {stream_valid[max_idx]}")
    
    if max_err < 1e-4:
        print("=> PASS: RealtimeFilter hoạt động chính xác 100% so với Batch (Causal Stream).")
    else:
        print("=> FAIL: Có sai số giữa Streaming và Batch!")

if __name__ == '__main__':
    test_streaming_vs_batch()
