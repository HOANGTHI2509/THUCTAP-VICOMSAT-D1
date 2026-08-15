import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import torch
import time

from src.models.time_aware_gru import FuelTimeAwareGRU
from src.core.filters.kalman_traditional import BoLocKalmanTieuChuan1D
from src.core.filters.kalman_adaptive import BoLocKalmanThichNghi1D
from src.pipeline.build_gru_dataset import process_timegap

def run_kalman(df, filter_class, is_adaptive=False):
    initial_fuel = float(df[df['FuelLevel'] > 0]['FuelLevel'].iloc[0])
    kf = filter_class(trang_thai_ban_dau=initial_fuel)
    smoothed = []
    
    for i, row in df.iterrows():
        z = row['FuelLevel']
        
        if is_adaptive:
            gia_toc = row['Acceleration'] if pd.notnull(row['Acceleration']) else 0
            trang_thai = 1 if row['MovementState'] == 'Moving' else 0
            dt = row['TimeGapMinutes']
            if pd.isna(dt) or dt <= 0 or dt > 60: dt = 1.0 # default to 1 min relative
            
            # Use cap_nhat for Adaptive Kalman
            kf.cap_nhat(z, ty_le_dt=dt, trang_thai_chuyen_dong=trang_thai, gia_toc=gia_toc)
            smoothed.append(kf.x)
        else:
            dt = row['TimeGapMinutes']
            if pd.isna(dt) or dt <= 0 or dt > 60: dt = 1.0
            
            # Use cap_nhat for Standard Kalman
            kf.cap_nhat(z, ty_le_dt=dt)
            smoothed.append(kf.x)
    
    return np.array(smoothed)

def run_gru(df, model, N=30):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    noisy_fuel = df['FuelLevel'].values
    speed = df['Speed'].fillna(0).values
    accel = df['Acceleration'].fillna(0).values
    mov = (df['MovementState'] == 'Moving').astype(float).values
    
    # Calculate RollingStd
    rstd = df['FuelLevel'].rolling(window=5, min_periods=1).std().fillna(0).values
    tg = process_timegap(df['TimeGapMinutes'].fillna(5.0).values)
    
    n_points = len(df)
    preds = np.zeros(n_points)
    preds[:N-1] = noisy_fuel[:N-1]
    
    X_infer = []
    for i in range(N - 1, n_points):
        w_noisy = noisy_fuel[i - N + 1 : i + 1]
        w_speed = speed[i - N + 1 : i + 1]
        w_accel = accel[i - N + 1 : i + 1]
        w_mov = mov[i - N + 1 : i + 1]
        w_rstd = rstd[i - N + 1 : i + 1]
        w_tg = tg[i - N + 1 : i + 1]
        
        anchor = w_noisy[0]
        w_fuel_res = w_noisy - anchor
        
        X_infer.append(np.column_stack((w_fuel_res, w_speed, w_accel, w_mov, w_rstd, w_tg)))
        
    if len(X_infer) > 0:
        X_tensor = torch.tensor(np.array(X_infer, dtype=np.float32)).to(device)
        with torch.no_grad():
            y_pred = model(X_tensor).cpu().numpy().flatten()
            
        for idx, i in enumerate(range(N - 1, n_points)):
            preds[i] = y_pred[idx] + noisy_fuel[i - N + 1]
            
    return preds

def evaluate_real_data(car_name, title, output_name):
    print(f"\nProcessing {car_name}...")
    df = pd.read_csv(f'data/processed/CarFuelHistory_Processed_{car_name}.csv')
    
    # Run filters
    std_kalman = run_kalman(df, BoLocKalmanTieuChuan1D, is_adaptive=False)
    adp_kalman = run_kalman(df, BoLocKalmanThichNghi1D, is_adaptive=True)
    
    # Run GRU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = FuelTimeAwareGRU(input_dim=6, hidden_dim=64).to(device)
    model.load_state_dict(torch.load('models/gru/best_gru_N30.pth', map_location=device))
    model.eval()
    
    start_time = time.time()
    gru_preds = run_gru(df, model, N=30)
    latency = (time.time() - start_time) / max(1, len(df) - 29) * 1000
    print(f"GRU Inference Latency: {latency:.3f} ms/point")
    
    # Find the largest segment
    segment_counts = df['SegmentID'].value_counts()
    target_segment = segment_counts.idxmax()
    
    segment_df = df[df['SegmentID'] == target_segment]
    
    # Take a 2000-point slice that has the most variance to ensure events are visible
    slice_size = 2000
    if len(segment_df) > slice_size:
        # Find a 2000-point window with highest variance
        rolling_var = segment_df['FuelLevel'].rolling(window=slice_size).var()
        end_idx = rolling_var.idxmax()
        if pd.isna(end_idx):
            end_idx = segment_df.index[slice_size]
        start_idx = end_idx - slice_size + 1
        idx = range(start_idx, end_idx + 1)
    else:
        idx = segment_df.index
    
    plt.figure(figsize=(15, 7))
    plt.plot(df.loc[idx, 'FuelLevel'], color='black', alpha=0.3, marker='.', label='Raw Fuel')
    plt.plot(pd.Series(std_kalman, index=df.index)[idx], color='blue', alpha=0.6, label='Standard Kalman')
    plt.plot(pd.Series(adp_kalman, index=df.index)[idx], color='orange', alpha=0.8, linewidth=2, label='Adaptive Kalman')
    plt.plot(pd.Series(gru_preds, index=df.index)[idx], color='red', alpha=0.9, linewidth=2, label='Time-aware GRU (N=30)')
    
    print(f"Debug {title}:")
    print(f"  Std Kalman Min/Max: {pd.Series(std_kalman, index=df.index)[idx].min()} / {pd.Series(std_kalman, index=df.index)[idx].max()}")
    print(f"  Adp Kalman Min/Max: {pd.Series(adp_kalman, index=df.index)[idx].min()} / {pd.Series(adp_kalman, index=df.index)[idx].max()}")
    
    plt.title(f"Real Data Evaluation: {title} (Segment {target_segment})")
    plt.xlabel("Time Step")
    plt.ylabel("Fuel Level (L)")
    plt.legend()
    plt.grid(True)
    
    # Save the slice to investigate the hook artifact
    debug_df = df.loc[idx].copy()
    debug_df['GRU_Pred'] = pd.Series(gru_preds, index=df.index)[idx]
    debug_df.to_csv(f'artifacts/debug_{output_name}.csv', index=False)
    
    os.makedirs('artifacts', exist_ok=True)
    plt.savefig(f'artifacts/{output_name}.png')
    plt.close()
    print(f"Saved plot to artifacts/{output_name}.png")

if __name__ == '__main__':
    # Seen by Kalman (Car1)
    evaluate_real_data("Car1", "Seen Vehicle (Car1)", "Real_Eval_Seen")
    
    # Unseen by Kalman (19B-04587)
    evaluate_real_data("19B-04587", "Unseen Vehicle (19B-04587)", "Real_Eval_Unseen")
