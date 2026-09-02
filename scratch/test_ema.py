import pandas as pd
import numpy as np
import glob
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.core.filters.kalman_adaptive import rts_smooth_1d, is_valid_measurement
from src.pipeline.build_real_dataset import generate_teacher_signal

def calculate_smoothness(series):
    return np.mean(np.abs(np.diff(series.dropna())))

def run_ema_test():
    csv_files = glob.glob('data/processed/CarFuelHistory_Processed_*_CNN_Realtime.csv')
    train_val_cars = ['Car1', 'Car2', 'Car3', 'Car5', '92H-07095']
    test_files = [f for f in csv_files if not any(c in f for c in train_val_cars)]
    
    alphas = [0.1, 0.2, 0.3]
    results = {
        'Raw': {'rmse': [], 'event_mae': [], 'smoothness': []},
        'CNN': {'rmse': [], 'event_mae': [], 'smoothness': []},
    }
    for alpha in alphas:
        results[f'EMA_{alpha}'] = {'rmse': [], 'event_mae': [], 'smoothness': []}
        
    for file in test_files:
        df = pd.read_csv(file)
        
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
        
        # We need to apply EMA per segment to avoid smoothing across big gaps
        valid_mask = ~df['CNN_Realtime'].isna() & ~df['TeacherFuel'].isna()
        if valid_mask.sum() < 10:
            continue
            
        df_valid = df.loc[valid_mask].copy()
        
        for alpha in alphas:
            # Apply EMA grouping by SegmentID so it doesn't smooth across gaps
            df_valid[f'EMA_{alpha}'] = df_valid.groupby('SegmentID')['CNN_Realtime'].transform(
                lambda x: x.ewm(alpha=alpha, adjust=False).mean()
            )

        y_true = df_valid['TeacherFuel'].values
        y_raw = df_valid['FuelLevel'].values
        y_cnn = df_valid['CNN_Realtime'].values
        
        # Calculate for Raw
        results['Raw']['rmse'].append(np.sqrt(np.mean((y_raw - y_true)**2)))
        results['Raw']['smoothness'].append(calculate_smoothness(pd.Series(y_raw)))
        
        # Calculate for CNN
        results['CNN']['rmse'].append(np.sqrt(np.mean((y_cnn - y_true)**2)))
        results['CNN']['smoothness'].append(calculate_smoothness(pd.Series(y_cnn)))
        
        target_magnitude = np.abs((y_true - y_raw) / capacity)
        event_mask = target_magnitude >= 0.01
        
        if np.sum(event_mask) > 0:
            results['CNN']['event_mae'].append(np.mean(np.abs(y_cnn[event_mask] - y_true[event_mask])))
            results['Raw']['event_mae'].append(np.mean(np.abs(y_raw[event_mask] - y_true[event_mask])))
            
        for alpha in alphas:
            y_ema = df_valid[f'EMA_{alpha}'].values
            
            results[f'EMA_{alpha}']['rmse'].append(np.sqrt(np.mean((y_ema - y_true)**2)))
            results[f'EMA_{alpha}']['smoothness'].append(calculate_smoothness(pd.Series(y_ema)))
            if np.sum(event_mask) > 0:
                results[f'EMA_{alpha}']['event_mae'].append(np.mean(np.abs(y_ema[event_mask] - y_true[event_mask])))

    print(f"{'Model':<15} | {'RMSE':<6} | {'Event MAE':<10} | {'Smoothness':<10}")
    print("-" * 50)
    
    for model_name, metrics in results.items():
        rmse = np.mean(metrics['rmse']) if metrics['rmse'] else 0
        ev_mae = np.mean(metrics['event_mae']) if metrics['event_mae'] else 0
        smooth = np.mean(metrics['smoothness']) if metrics['smoothness'] else 0
        
        print(f"{model_name:<15} | {rmse:<6.2f} | {ev_mae:<10.2f} | {smooth:<10.3f}")

if __name__ == '__main__':
    run_ema_test()
