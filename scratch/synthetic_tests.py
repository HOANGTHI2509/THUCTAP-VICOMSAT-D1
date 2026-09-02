import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
import os
import sys
from datetime import datetime, timedelta

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.models.cnn_gated_attention import FuelCNN1DGatedAttention

def generate_synthetic_data():
    np.random.seed(42)
    start_time = datetime(2026, 1, 1, 0, 0, 0)
    
    dfs = []
    
    # helper to build df
    def build_df(segment_id, base_levels):
        n = len(base_levels)
        times = [start_time + timedelta(minutes=5*i) for i in range(n)]
        df = pd.DataFrame({
            'SegmentID': [segment_id] * n,
            'FuelTime': times,
            'FuelLevel': base_levels,
            'Speed': [30.0] * n  # Constant speed
        })
        return df

    # TEST 1: Nhiễu nhỏ
    t1_levels = np.random.normal(400, 1.0, 100)
    # Add a fake 800 point to set capacity easily
    t1_levels[0] = 800
    t1_levels[1] = 400
    dfs.append(build_df(1, t1_levels))
    
    # TEST 2: Spike đơn 400 -> 415 -> 400
    t2_levels = np.full(100, 400.0)
    t2_levels[50] = 415.0
    t2_levels[0] = 800
    t2_levels[1] = 400
    dfs.append(build_df(2, t2_levels))
    
    # TEST 3: Tăng thật 400 -> 415
    t3_levels = np.full(100, 400.0)
    t3_levels[50:60] = np.linspace(400, 415, 10)
    t3_levels[60:] = 415.0
    t3_levels[0] = 800
    t3_levels[1] = 400
    dfs.append(build_df(3, t3_levels))
    
    # TEST 4: Giảm thật 400 -> 384
    t4_levels = np.full(100, 400.0)
    t4_levels[50:55] = np.linspace(400, 384, 5)
    t4_levels[60:] = 384.0
    t4_levels[0] = 800
    t4_levels[1] = 400
    dfs.append(build_df(4, t4_levels))
    
    # TEST 5: Lỗi cực đoan 400 -> 760 -> 400
    t5_levels = np.full(100, 400.0)
    t5_levels[50] = 760.0
    t5_levels[0] = 800
    t5_levels[1] = 400
    dfs.append(build_df(5, t5_levels))
    
    return pd.concat(dfs, ignore_index=True)

def run_synthetic_tests():
    print("Generating synthetic data...")
    df = generate_synthetic_data()
    df["_OriginalOrder"] = np.arange(len(df))
    df = df.sort_values(["SegmentID", "FuelTime", "_OriginalOrder"], kind="stable")
    
    # Capacity extraction per segment (we put 800 at index 0 for all)
    capacities = df.groupby('SegmentID')['FuelLevel'].transform(lambda x: x.max())
    
    # Load Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FuelCNN1DGatedAttention(input_dim=6).to(device)
    model.load_state_dict(torch.load('models/cnn_real/best_model_real.pt', map_location=device))
    model.eval()
    
    # Feature engineering similar to predict_cnn.py
    df['TimeGap'] = df.groupby('SegmentID')['FuelTime'].diff().dt.total_seconds().fillna(0) / 60.0
    df['RollingStd'] = df.groupby('SegmentID')['FuelLevel'].transform(lambda x: x.rolling(5, min_periods=1).std().fillna(0.0))
    
    df["CNN_Realtime"] = np.nan
    df["EMA_0.3"] = np.nan
    df["EMA_0.4"] = np.nan
    df["EMA_0.5"] = np.nan
    
    N = 10
    
    print("Running CNN inference...")
    for segment_id, group in df.groupby('SegmentID'):
        indices = group.index.tolist()
        capacity = group['FuelLevel'].max()
        
        current_window = []
        current_gaps = []
        current_speeds = []
        
        ema_3 = None
        ema_4 = None
        ema_5 = None
        
        for i, idx in enumerate(indices):
            row = df.loc[idx]
            fuel = float(row['FuelLevel'])
            speed = float(row['Speed'])
            gap = float(row['TimeGap'])
            std = float(row['RollingStd'])
            
            current_window.append(fuel)
            current_gaps.append(gap)
            current_speeds.append(speed)
            
            if len(current_window) > N:
                current_window.pop(0)
                current_gaps.pop(0)
                current_speeds.pop(0)
                
            if len(current_window) < N:
                prediction = fuel
            else:
                w_fuels = np.array(current_window)
                w_gaps = np.array(current_gaps)
                w_speeds = np.array(current_speeds)
                
                feat_fuel = w_fuels / capacity
                feat_delta = np.diff(w_fuels, prepend=w_fuels[0]) / capacity
                feat_speed = w_speeds / 100.0
                feat_move = (w_speeds > 5).astype(float)
                feat_gap = w_gaps / 30.0
                
                feat_std = np.zeros(N)
                for k in range(N):
                    start_k = max(0, k - 4)
                    feat_std[k] = np.std(w_fuels[start_k:k+1]) / capacity if k > 0 else 0
                
                features = np.stack([feat_fuel, feat_delta, feat_speed, feat_move, feat_std, feat_gap], axis=1)
                
                with torch.no_grad():
                    X = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(device)
                    y_pred = model(X).item()
                
                # Bounded ±5%
                y_pred = np.clip(y_pred, -0.05, 0.05)
                
                raw_fuel = fuel
                correction = y_pred * capacity
                prediction = raw_fuel + correction
                
                # Deadzone 0.1%
                if abs(prediction - raw_fuel) <= 0.001 * capacity:
                    prediction = raw_fuel
            
            df.loc[idx, "CNN_Realtime"] = prediction
            
            # EMA calculations
            if ema_3 is None:
                ema_3 = ema_4 = ema_5 = prediction
            else:
                ema_3 = 0.3 * prediction + 0.7 * ema_3
                ema_4 = 0.4 * prediction + 0.6 * ema_4
                ema_5 = 0.5 * prediction + 0.5 * ema_5
                
            df.loc[idx, "EMA_0.3"] = ema_3
            df.loc[idx, "EMA_0.4"] = ema_4
            df.loc[idx, "EMA_0.5"] = ema_5

    print("Plotting results...")
    fig, axes = plt.subplots(5, 1, figsize=(15, 20))
    titles = ["TEST 1: Nhiễu nhỏ", "TEST 2: Spike đơn", "TEST 3: Tăng thật", "TEST 4: Giảm thật", "TEST 5: Lỗi cực đoan"]
    
    for i, (segment_id, group) in enumerate(df.groupby('SegmentID')):
        ax = axes[i]
        # Skip the first two points which are for capacity trick
        g = group.iloc[2:].reset_index(drop=True)
        
        ax.plot(g.index, g['FuelLevel'], label='Raw', color='gray', linestyle='--', alpha=0.6)
        ax.plot(g.index, g['CNN_Realtime'], label='CNN Output', color='red', alpha=0.3)
        ax.plot(g.index, g['EMA_0.3'], label='EMA 0.3', color='blue', linewidth=2)
        ax.plot(g.index, g['EMA_0.4'], label='EMA 0.4', color='green', linewidth=2)
        ax.plot(g.index, g['EMA_0.5'], label='EMA 0.5', color='orange', linewidth=2)
        
        ax.set_title(titles[i])
        ax.grid(True, alpha=0.3)
        ax.legend()
        
    plt.tight_layout()
    plt.savefig('scratch/synthetic_cases.png', dpi=150)
    print("Saved to scratch/synthetic_cases.png")

if __name__ == '__main__':
    run_synthetic_tests()
