import os
import sys
import glob
import numpy as np
import pandas as pd
import torch
import time

sys.stdout.reconfigure(encoding="utf-8")
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.models.cnn_gated_attention import FuelCNN1DGatedAttention
from src.core.filters.anomaly_detector import FuelAnomalyDetector

def process_timegap(tg):
    if isinstance(tg, pd.Series):
        return tg.fillna(5.0).clip(lower=1.0, upper=30.0)
    else:
        tg = np.array(tg)
        tg = np.where(np.isnan(tg), 5.0, tg)
        return np.clip(tg, 1.0, 30.0)

def is_valid_measurement(fuel, feature_status) -> bool:
    fuel_val = pd.to_numeric(fuel, errors="coerce")
    if pd.isna(fuel_val) or fuel_val <= 0:
        return False
    status = str(feature_status).strip().upper()
    invalid_statuses = {"FUEL_ZERO", "PREVIOUS_INVALID", "INVALID"}
    return status not in invalid_statuses

def extract_features_realtime(window_df, time_gap_mins, capacity):
    """
    Trích xuất đặc trưng cho 1 cửa sổ N=10.
    Features: FuelLevel, DeltaFuel, Speed, MovEncoded, RollingStd, TimeGapClipped
    """
    raw_fuel = window_df['FuelLevel'].values
    
    # 1. FuelLevel / Capacity
    feat_fuel = raw_fuel / capacity
    
    # 2. DeltaFuel / Capacity
    delta_fuel = np.diff(raw_fuel, prepend=raw_fuel[0]) / capacity
    
    speed = window_df['Speed'].values
    mov = window_df['MovEncoded'].values
    
    rstd = window_df['RollingStd'].values
    tg_clipped = process_timegap(time_gap_mins)
    
    features = np.column_stack((
        feat_fuel, 
        delta_fuel,
        speed / 100.0, 
        mov, 
        rstd / capacity, 
        tg_clipped / 30.0
    ))
    return features

class EventAwareEMA:
    def __init__(self, capacity: float, persistence: int = 3):
        self.capacity = capacity
        self.threshold = max(5.0, 0.02 * capacity)
        self.persistence = persistence
        self.state = None
        self.suspicious_count = 0
        self.sign = 0
        
    def update(self, z: float) -> float:
        if self.state is None:
            self.state = z
            return self.state
            
        diff = z - self.state
        
        if abs(diff) > self.threshold:
            current_sign = np.sign(diff)
            if current_sign == self.sign or self.sign == 0:
                self.suspicious_count += 1
            else:
                self.suspicious_count = 1
            self.sign = current_sign
        else:
            self.suspicious_count = 0
            self.sign = 0
            
        if self.suspicious_count >= self.persistence:
            self.state = z
            self.suspicious_count = 0
            self.sign = 0
        elif self.suspicious_count > 0:
            pass # Freeze state
        else:
            self.state = 0.1 * z + 0.9 * self.state
            
        return self.state

def load_cnn_model(N=10):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = FuelCNN1DGatedAttention(input_dim=6, hidden_dim=64).to(device)
    model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'models/cnn_real/best_model_real.pt'))
    if not os.path.exists(model_path):
        print(f"Không tìm thấy model tại {model_path}. Vui lòng chạy train_cnn_real.py trước.")
        return None, device
        
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print(f"Loaded model from {model_path} to {device}")
    return model, device

def run_realtime_inference(N=10):
    print("=== BẮT ĐẦU CHẠY INFERENCE CNN-GA REALTIME ===")
    model, device = load_cnn_model(N)
    if model is None:
        return

    # Check multiple locations for data files
    data_dirs = [
        "data/processed/CarFuelHistory_Processed_*.csv",
        "DU_lieu_VICOMSAT/data_processed/CarFuelHistory_Processed_*.csv",
        "CarFuelHistory_Processed_*.csv"
    ]
    
    danh_sach_file = []
    for pattern in data_dirs:
        abs_pattern = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', pattern))
        found = sorted(glob.glob(abs_pattern))
        if found:
            # Lọc bỏ các file output cũ để tránh vòng lặp
            danh_sach_file = [f for f in found if "_CNN" not in f and "_Kalman" not in f]
            if danh_sach_file:
                break

    if not danh_sach_file:
        print("Không tìm thấy file CarFuelHistory_Processed_*.csv nào để inference.")
        return

    for duong_dan_file in danh_sach_file:
        ma_xe = os.path.basename(duong_dan_file).replace("CarFuelHistory_Processed_", "").replace(".csv", "")
        print(f"Đang chạy CNN Realtime cho xe {ma_xe}...")
        
        df = pd.read_csv(duong_dan_file)
        
        # Tiền xử lý
        df["FuelTime"] = pd.to_datetime(df["FuelTime"], errors="coerce")
        df["FuelLevel"] = pd.to_numeric(df["FuelLevel"], errors="coerce")
        df["Speed"] = pd.to_numeric(df["Speed"], errors="coerce").fillna(0)
        df["TimeGapMinutes"] = pd.to_numeric(df["TimeGapMinutes"], errors="coerce").fillna(5.0)
        
        # Assign SegmentID if not exists
        if "SegmentID" not in df.columns:
            if "FlagLongGap" in df.columns:
                df["SegmentID"] = (df["FlagLongGap"] == True).cumsum()
            else:
                df["SegmentID"] = (df["TimeGapMinutes"] > 30.0).cumsum()
                
        df["_OriginalOrder"] = np.arange(len(df))
        df = df.sort_values(["SegmentID", "FuelTime", "_OriginalOrder"], kind="stable")
        
        df["CNN_Realtime"] = np.nan
        df["CNN_EMA"] = np.nan
        
        start_time = time.time()
        
        # 2. Extract Capacity
        est_capacity = df["FuelLevel"].quantile(0.99)
        if pd.isna(est_capacity) or est_capacity < 50: est_capacity = 200.0
            
        df['RollingStd'] = df.groupby('SegmentID')['FuelLevel'].transform(lambda x: x.rolling(5, min_periods=1).std().fillna(0.0))
        df['MovEncoded'] = (df['MovementState'] == 'Moving').astype(float)
        
        for ma_doan, nhom in df.groupby("SegmentID", sort=False, dropna=False):
            indices = nhom.index.tolist()
            window_indices = []
            window_gaps = []
            window_speeds = []
            khoang_thoi_gian_tich_luy = 0.0
            ema_filter = EventAwareEMA(est_capacity, persistence=3)
            
            for i, idx in enumerate(indices):
                row = df.loc[idx]
                gap = row["TimeGapMinutes"]
                speed = row["Speed"]
                if pd.isna(gap) or gap <= 0: gap = 5.0
                
                if not is_valid_measurement(row["FuelLevel"], row.get("FeatureStatus", "")):
                    khoang_thoi_gian_tich_luy += gap
                    continue
                    
                khoang_thoi_gian = gap + khoang_thoi_gian_tich_luy
                khoang_thoi_gian_tich_luy = 0.0
                
                window_indices.append(idx)
                window_gaps.append(khoang_thoi_gian)
                window_speeds.append(speed)
                
                # Cắt lấy tối đa N điểm cuối
                current_window = window_indices[-N:]
                tg_seq = window_gaps[-N:]
                
                # Edge Padding
                if len(current_window) < N:
                    val = float(row["FuelLevel"])
                    df.loc[idx, "CNN_Realtime"] = val
                    df.loc[idx, "CNN_EMA"] = round(ema_filter.update(val), 1)
                    continue
                else:
                    padded_window = current_window
                    
                window_df = df.loc[padded_window].copy()
                
                # 4. Trích xuất đặc trưng
                features = extract_features_realtime(window_df, tg_seq, est_capacity)
                
                # 5. Dự đoán
                X_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(device)
                
                raw_fuel = window_df['FuelLevel'].values[-1]
                speed_val = window_df['Speed'].values[-1]
                
                if np.isclose(raw_fuel, 794.7) and np.isclose(speed_val, 34.0):
                    print(f"\nDEBUG INFERENCE AT TARGET POINT ({ma_xe}):")
                    print(f"Raw Fuel: {raw_fuel}, Speed: {speed_val}")
                    print(f"FuelLevel (feat): {features[-1, 0]:.4f}")
                    print(f"DeltaFuel (feat): {features[-1, 1]:.4f}")
                    print(f"Speed (feat): {features[-1, 2]:.4f}")
                    print(f"Movement (feat): {features[-1, 3]:.4f}")
                    print(f"RollingStd (feat): {features[-1, 4]:.4f}")
                    print(f"TimeGap (feat): {features[-1, 5]:.4f}")
                    
                with torch.no_grad():
                    pred_residual_pct = model(X_tensor)
                    pred_residual_pct = pred_residual_pct.item()
                    
                # Áp dụng Deadzone
                DEADZONE = 0.001
                if abs(pred_residual_pct) < DEADZONE:
                    pred_residual_pct = 0.0
                    
                raw_fuel = window_df['FuelLevel'].values[-1]
                prediction = raw_fuel + (pred_residual_pct * est_capacity)
                prediction = max(0.0, prediction) # Ràng buộc không âm
                prediction = round(prediction, 1) # Làm tròn 1 chữ số thập phân
                
                if np.isclose(raw_fuel, 794.7) and np.isclose(speed_val, 34.0):
                    print(f"Prediction correction %: {pred_residual_pct:.4f}")
                    print(f"Correction Liters:       {pred_residual_pct * est_capacity:.2f}")
                    print(f"Raw Fuel:                {raw_fuel}")
                    print(f"CNN Fuel:                {prediction}")
                
                df.loc[idx, "CNN_Realtime"] = prediction
                
                # Event-Aware EMA
                df.loc[idx, "CNN_EMA"] = round(ema_filter.update(prediction), 1)

        inference_time = time.time() - start_time
        print(f"Hoàn tất {ma_xe} trong {inference_time:.2f}s! ({len(df)} samples, ~{inference_time/max(1,len(df))*1000:.2f}ms/sample)")
        
        df = df.sort_values("_OriginalOrder", kind="stable").drop(columns="_OriginalOrder")
        duong_dan_xuat = duong_dan_file.replace(".csv", "_CNN_Realtime.csv")
        df.to_csv(duong_dan_xuat, index=False, encoding="utf-8-sig")
        print(f"Đã xuất: {duong_dan_xuat}\n")

    print("=== ĐÃ CHẠY XONG CNN-GA REALTIME ===")

if __name__ == "__main__":
    run_realtime_inference()
