import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import torch
import time

from src.core.filters.kalman_traditional import BoLocKalmanTieuChuan1D
from src.core.filters.kalman_adaptive import BoLocKalmanThichNghi1D

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


def evaluate_real_data(car_name, title, output_name):
    print(f"\nProcessing {car_name}...")
    df = pd.read_csv(f'data/processed/CarFuelHistory_Processed_{car_name}.csv')
    
    # Run filters
    std_kalman = run_kalman(df, BoLocKalmanTieuChuan1D, is_adaptive=False)
    adp_kalman = run_kalman(df, BoLocKalmanThichNghi1D, is_adaptive=True)    
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
    print(f"Debug {title}:")
    print(f"  Std Kalman Min/Max: {pd.Series(std_kalman, index=df.index)[idx].min()} / {pd.Series(std_kalman, index=df.index)[idx].max()}")
    print(f"  Adp Kalman Min/Max: {pd.Series(adp_kalman, index=df.index)[idx].min()} / {pd.Series(adp_kalman, index=df.index)[idx].max()}")
    
    plt.title(f"Real Data Evaluation: {title} (Segment {target_segment})")
    plt.xlabel("Time Step")
    plt.ylabel("Fuel Level (L)")
    plt.legend()
    plt.grid(True)
    
    # Save the slice to investigate the hook artifact
    debug_df = df.loc[idx].copy()    debug_df.to_csv(f'artifacts/debug_{output_name}.csv', index=False)
    
    os.makedirs('artifacts', exist_ok=True)
    plt.savefig(f'artifacts/{output_name}.png')
    plt.close()
    print(f"Saved plot to artifacts/{output_name}.png")

if __name__ == '__main__':
    # Seen by Kalman (Car1)
    evaluate_real_data("Car1", "Seen Vehicle (Car1)", "Real_Eval_Seen")
    
    # Unseen by Kalman (19B-04587)
    evaluate_real_data("19B-04587", "Unseen Vehicle (19B-04587)", "Real_Eval_Unseen")
