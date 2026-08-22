import pandas as pd
import numpy as np
import os
import sys
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.core.filters.anomaly_detector import FuelAnomalyDetector
from src.core.filters.kalman_adaptive import BoLocKalmanThichNghi1D, rts_smooth_1d, is_valid_measurement

def process_timegap(tg_series):
    return tg_series.fillna(5.0).clip(lower=1.0, upper=30.0)

def extract_windows_from_segment(group, capacity, window_size=10):
    X_list = []
    y_list = []

    raw_fuels = group['FuelLevel'].values
    speeds = group['Speed'].values
    movements = group['MovEncoded'].values
    rolling_stds = group['RollingStd'].values
    time_gaps = group['TimeGapClipped'].values
    teacher_fuels = group['TeacherFuel'].values

    n_samples = len(group)

    if n_samples < window_size:
        return [], []

    for i in range(window_size - 1, n_samples):
        start_idx = i - window_size + 1

        window_fuels = raw_fuels[start_idx:i+1]
        window_speeds = speeds[start_idx:i+1]
        window_movements = movements[start_idx:i+1]
        window_rolling_stds = rolling_stds[start_idx:i+1]
        window_time_gaps = time_gaps[start_idx:i+1]

        feat_fuel = window_fuels / capacity

        delta_fuel = np.diff(
            window_fuels,
            prepend=window_fuels[0]
        ) / capacity

        features = np.column_stack([
            feat_fuel,
            delta_fuel,
            window_speeds / 100.0,
            window_movements,
            window_rolling_stds / capacity,
            window_time_gaps / 30.0
        ])

        target_correction = (
            teacher_fuels[i] - raw_fuels[i]
        ) / capacity

        X_list.append(features)
        y_list.append(target_correction)

    return X_list, y_list

def generate_teacher_signal(df_seg, capacity):
    print("  Running Anomaly Detector...")
    
    detector = FuelAnomalyDetector(
        capacity=capacity, 
        look_ahead_hours=6.0, 
        min_low_minutes=30.0,
        spike_threshold=max(10.0, 0.05 * capacity)
    )
    df_seg = detector.detect_and_clean(df_seg)
    
    # Calculate Acceleration and RollingStd
    time_gap_sec = df_seg["TimeGapMinutes"].fillna(5.0) * 60.0
    safe_time_gap = time_gap_sec.replace(0, 1.0)
    df_seg["Acceleration"] = df_seg.groupby("SegmentID", dropna=False)["Speed"].diff().fillna(0.0) / safe_time_gap
    
    df_seg['RollingStd'] = df_seg.groupby('SegmentID')['FuelLevel'].transform(lambda x: x.rolling(5, min_periods=1).std().fillna(0.0))
    df_seg['MovEncoded'] = (df_seg['MovementState'] == 'Moving').astype(float)
    df_seg['TimeGapClipped'] = process_timegap(df_seg['TimeGapMinutes'])
    
    df_seg['TeacherFuel'] = np.nan
    
    print("  Running Adaptive Kalman + RTS Smoother...")
    for seg_id, group in df_seg.groupby('SegmentID', sort=False, dropna=False):
        group_rolling_std = group['RollingStd'].tolist()
        kf_adapt = None
        
        kalman_r = int(max(64.0, (0.04 * capacity)**2))
        adapt_threshold = float(max(15.0, 0.075 * capacity))
        
        x_f, P_f, x_p, P_p = [], [], [], []
        reference_gap = 5.0
        khoang_thoi_gian_tich_luy = 0.0
        
        for i_loc, dong in enumerate(group.itertuples()):
            gap = getattr(dong, "TimeGapMinutes", reference_gap)
            if pd.isna(gap) or gap <= 0: gap = reference_gap
            
            movement_state = 0 if str(getattr(dong, "MovementState", "Moving")).strip().upper() == "STOPPED" else 1
            acceleration = float(getattr(dong, "Acceleration", 0.0))
            
            if not is_valid_measurement(getattr(dong, 'CleanedFuel', None), getattr(dong, 'FeatureStatus', '')):
                khoang_thoi_gian_tich_luy += gap
                x_f.append(np.nan)
                P_f.append(0.0)
                x_p.append(np.nan)
                P_p.append(0.0)
                continue
                
            measurement = float(getattr(dong, 'CleanedFuel', dong.FuelLevel))
            khoang_thoi_gian = gap + khoang_thoi_gian_tich_luy
            khoang_thoi_gian_tich_luy = 0.0
            dt_ratio = khoang_thoi_gian / reference_gap
            
            if kf_adapt is None:
                kf_adapt = BoLocKalmanThichNghi1D(
                    trang_thai_ban_dau=measurement, 
                    sai_so_uoc_luong_ban_dau=4.0, 
                    nhieu_qua_trinh=1.0, 
                    r_co_ban=kalman_r, 
                    r_nhieu_dot_bien=kalman_r * 2.0,
                    nguong_bat_nhay_co_ban=5.0,
                    nguong_toi_da=max(25.0, 0.125 * capacity),
                    nhip_cho_xac_nhan=3,
                    muc_tieu_thu_100km=capacity * 0.05
                )
                x_f.append(measurement)
                P_f.append(0.0)
                x_p.append(measurement)
                P_p.append(0.0)
            else:
                x_forward, P_forward, x_predict, P_predict = kf_adapt.cap_nhat(
                    measurement, 
                    ty_le_dt=dt_ratio, 
                    trang_thai_chuyen_dong=movement_state, 
                    gia_toc=acceleration,
                    rolling_std=float(group_rolling_std[i_loc]),
                    van_toc=float(getattr(dong, 'Speed', 0.0) or 0.0)
                )
                x_f.append(x_forward)
                P_f.append(P_forward)
                x_p.append(x_predict)
                P_p.append(P_predict)
                
        # Teacher is causal adaptive kalman
        x_f_arr = pd.Series(x_f).ffill().bfill().values
        df_seg.loc[group.index, 'TeacherFuel'] = x_f_arr
        
    return df_seg

def build_dataset():
    train_cars = ['Car1', 'Car2', 'Car3', 'Car5']
    val_cars = ['92H-07095']
    
    out_dir = 'data/real_dataset/windows'
    os.makedirs(out_dir, exist_ok=True)
    
    for split_name, cars in [('train', train_cars), ('val', val_cars)]:
        all_X = []
        all_y = []
        
        for car in cars:
            print(f"\nProcessing {car} for {split_name}...")
            file_path = f"data/processed/CarFuelHistory_Processed_{car}.csv"
            if not os.path.exists(file_path):
                print(f"Warning: {file_path} not found. Skipping.")
                continue
                
            df = pd.read_csv(file_path)
            df['FuelTime'] = pd.to_datetime(df['FuelTime'], errors="coerce")
            df['FuelLevel'] = pd.to_numeric(df['FuelLevel'], errors="coerce")
            df['SegmentID'] = pd.to_numeric(df['SegmentID'], errors="coerce")
            df["_OriginalOrder"] = np.arange(len(df))
            df = df.sort_values(["SegmentID", "FuelTime", "_OriginalOrder"], kind="stable")
            
            capacity = df['FuelLevel'].quantile(0.99)
            if pd.isna(capacity) or capacity < 50: capacity = 200.0
            
            df = generate_teacher_signal(df, capacity)
            
            print("  Extracting windows...")
            for seg_id, group in tqdm(df.groupby('SegmentID', sort=False)):
                X_list, y_list = extract_windows_from_segment(group, capacity, window_size=10)
                all_X.extend(X_list)
                all_y.extend(y_list)
                
        if len(all_X) > 0:
            X = np.array(all_X, dtype=np.float32)
            y = np.array(all_y, dtype=np.float32).reshape(-1, 1)
            np.save(os.path.join(out_dir, f'X_{split_name}_N10.npy'), X)
            np.save(os.path.join(out_dir, f'y_{split_name}_N10.npy'), y)
            print(f"Saved {split_name}: X shape {X.shape}, y shape {y.shape}")

if __name__ == '__main__':
    build_dataset()
