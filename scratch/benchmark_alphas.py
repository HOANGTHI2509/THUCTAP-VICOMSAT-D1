import pandas as pd
import numpy as np
import glob
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.pipeline.build_real_dataset import generate_teacher_signal

def calculate_smoothness(series):
    return np.mean(np.abs(np.diff(series.dropna())))

def run_benchmark():
    csv_files = glob.glob('data/processed/CarFuelHistory_Processed_*_CNN_Realtime.csv')
    train_val_cars = ['Car1', 'Car2', 'Car3', 'Car5', '92H-07095']
    test_files = [f for f in csv_files if not any(c in f for c in train_val_cars)]
    
    alphas = [0.3, 0.4, 0.5]
    
    metrics = {a: {'rmse': [], 'event_mae': [], 'smoothness': [], 
                  'delay_50': [], 'delay_80': [], 'delay_90': [], 'retention': []} for a in alphas}
    
    print(f"Evaluating over {len(test_files)} test files...")
    
    for file in test_files:
        df = pd.read_csv(file)
        if 'CNN_Realtime' not in df.columns: continue
        
        orig_file = file.replace('_CNN_Realtime.csv', '.csv')
        df_orig = pd.read_csv(orig_file)
        df_orig['FuelTime'] = pd.to_datetime(df_orig['FuelTime'], errors="coerce")
        df_orig['FuelLevel'] = pd.to_numeric(df_orig['FuelLevel'], errors="coerce")
        df_orig['SegmentID'] = pd.to_numeric(df_orig['SegmentID'], errors="coerce")
        df_orig["_OriginalOrder"] = np.arange(len(df_orig))
        df_orig = df_orig.sort_values(["SegmentID", "FuelTime", "_OriginalOrder"], kind="stable")
        
        capacity = df_orig['FuelLevel'].quantile(0.99)
        if pd.isna(capacity) or capacity < 50: capacity = 200.0
        
        df_teacher = generate_teacher_signal(df_orig, capacity)
        df['TeacherFuel'] = df_teacher['TeacherFuel']
        
        valid_mask = ~df['CNN_Realtime'].isna() & ~df['TeacherFuel'].isna()
        if valid_mask.sum() < 10: continue
        
        df_valid = df.loc[valid_mask].copy()
        
        for alpha in alphas:
            df_valid[f'EMA_{alpha}'] = df_valid.groupby('SegmentID')['CNN_Realtime'].transform(
                lambda x: x.ewm(alpha=alpha, adjust=False).mean()
            )
            
        y_true = df_valid['TeacherFuel'].values
        y_raw = df_valid['FuelLevel'].values
        y_cnn = df_valid['CNN_Realtime'].values
        
        target_magnitude = np.abs((y_true - y_raw) / capacity)
        event_mask = target_magnitude >= 0.01
        
        for alpha in alphas:
            y_ema = df_valid[f'EMA_{alpha}'].values
            
            rmse = np.sqrt(np.mean((y_ema - y_true)**2))
            smooth = calculate_smoothness(pd.Series(y_ema))
            ev_mae = np.mean(np.abs(y_ema[event_mask] - y_true[event_mask])) if np.sum(event_mask) > 0 else 0
            
            metrics[alpha]['rmse'].append(rmse)
            metrics[alpha]['smoothness'].append(smooth)
            if ev_mae > 0: metrics[alpha]['event_mae'].append(ev_mae)
            
        # Delay and Retention calculation
        for segment, group in df_valid.groupby('SegmentID'):
            if len(group) < 20: continue
            cnn_vals = group['CNN_Realtime'].values
            
            diffs = np.abs(np.diff(cnn_vals))
            event_indices = np.where(diffs > 10.0)[0]
            
            for idx in event_indices:
                start_val = cnn_vals[idx]
                target_val = cnn_vals[idx + 1]
                jump = target_val - start_val
                
                for alpha in alphas:
                    ema_vals = group[f'EMA_{alpha}'].values
                    
                    # Compute delay
                    d50, d80, d90 = -1, -1, -1
                    th50 = start_val + 0.5 * jump
                    th80 = start_val + 0.8 * jump
                    th90 = start_val + 0.9 * jump
                    
                    max_ema_jump = 0
                    search_range = min(15, len(ema_vals) - idx - 1)
                    
                    for k in range(1, search_range):
                        current_ema = ema_vals[idx + k]
                        current_jump = current_ema - start_val
                        
                        if jump > 0:
                            max_ema_jump = max(max_ema_jump, current_jump)
                            if d50 == -1 and current_ema >= th50: d50 = k
                            if d80 == -1 and current_ema >= th80: d80 = k
                            if d90 == -1 and current_ema >= th90: d90 = k
                        else:
                            max_ema_jump = min(max_ema_jump, current_jump)
                            if d50 == -1 and current_ema <= th50: d50 = k
                            if d80 == -1 and current_ema <= th80: d80 = k
                            if d90 == -1 and current_ema <= th90: d90 = k
                            
                    # Retention
                    retention = (max_ema_jump / jump) * 100 if jump != 0 else 100
                    metrics[alpha]['retention'].append(retention)
                    
                    # Convert steps to minutes (avg 5 mins per step)
                    if d50 != -1: metrics[alpha]['delay_50'].append(d50 * 5)
                    if d80 != -1: metrics[alpha]['delay_80'].append(d80 * 5)
                    if d90 != -1: metrics[alpha]['delay_90'].append(d90 * 5)

    print(f"\n| Alpha | RMSE ↓ | Event MAE ↓ | Smoothness ↓ | 50% Delay ↓ | 80% Delay ↓ | 90% Delay ↓ | Retention ↑ |")
    print(f"|------:|-------:|------------:|-------------:|------------:|------------:|------------:|------------:|")
    
    for alpha in alphas:
        rmse = np.mean(metrics[alpha]['rmse'])
        mae = np.mean(metrics[alpha]['event_mae'])
        smooth = np.mean(metrics[alpha]['smoothness'])
        d50 = np.mean(metrics[alpha]['delay_50']) if metrics[alpha]['delay_50'] else 0
        d80 = np.mean(metrics[alpha]['delay_80']) if metrics[alpha]['delay_80'] else 0
        d90 = np.mean(metrics[alpha]['delay_90']) if metrics[alpha]['delay_90'] else 0
        ret = np.mean(metrics[alpha]['retention']) if metrics[alpha]['retention'] else 0
        
        print(f"| {alpha:>5.1f} | {rmse:>6.2f} | {mae:>11.2f} | {smooth:>12.3f} | {d50:>8.1f}p | {d80:>8.1f}p | {d90:>8.1f}p | {ret:>10.1f}% |")

if __name__ == '__main__':
    run_benchmark()
