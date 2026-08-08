import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import pandas as pd
import numpy as np
import glob
import os
import sys
sys.stdout.reconfigure(encoding='utf-8')

from src.core.filters.moving_average import moving_average_filter
from src.core.filters.median_filter import median_filter
from src.core.filters.kalman_traditional import StandardKalmanFilter1D, is_valid_measurement
from src.core.filters.kalman_adaptive import AdaptiveKalmanFilter1D

def calculate_smoothness(series):
    valid_series = series[~np.isnan(series)]
    diffs = np.diff(valid_series)
    if len(diffs) == 0: return np.nan
    return np.mean(np.abs(diffs))

def get_dsp_metrics(raw, filtered, shift_steps=0, burn_in=10):
    if shift_steps > 0:
        aligned_filtered = np.roll(filtered, -shift_steps)
    else:
        aligned_filtered = filtered.copy()
        
    mask = ~np.isnan(raw) & ~np.isnan(aligned_filtered) & (raw > 0)
    mask[:burn_in] = False
    if shift_steps > 0: mask[-shift_steps:] = False
        
    if not np.any(mask): return np.nan, np.nan, np.nan
        
    valid_raw = raw[mask]
    valid_filt = aligned_filtered[mask]
    
    rmse = np.sqrt(np.mean((valid_raw - valid_filt)**2))
    max_err = np.max(np.abs(valid_raw - valid_filt))
    
    noise = valid_raw - valid_filt
    var_signal = np.var(valid_filt)
    var_noise = np.var(noise)
    snr = 10 * np.log10(var_signal / var_noise) if var_noise > 0 else float('inf')
    
    return rmse, snr, max_err

def evaluate():
    files = glob.glob("CarFuelHistory_Processed_*.csv")
    if not files: return
    
    file_path = files[0]
    print(f"Đang đánh giá trên dữ liệu: {os.path.basename(file_path)}")
    df = pd.read_csv(file_path)
    
    results = []
    
    for seg_id, group in df.groupby('SegmentID', sort=False, dropna=False):
        fuels = group['FuelLevel'].values
        if len(fuels) < 50: continue 
        
        # 1. Moving Average
        ma_vals = np.array(moving_average_filter(fuels.tolist(), window_size=10))
        ma_rmse_real, ma_snr_real, ma_max_real = get_dsp_metrics(fuels, ma_vals, shift_steps=0)
        ma_rmse_shape, ma_snr_shape, ma_max_shape = get_dsp_metrics(fuels, ma_vals, shift_steps=5)
        
        # 2. Median Filter
        med_vals = np.array(median_filter(fuels.tolist(), window_size=10))
        med_rmse_real, med_snr_real, med_max_real = get_dsp_metrics(fuels, med_vals, shift_steps=0)
        med_rmse_shape, med_snr_shape, med_max_shape = get_dsp_metrics(fuels, med_vals, shift_steps=5)
        
        # 3. Standard Kalman (Tuning mới: Q=1.0, R=9.0, reset khi mất sóng)
        kf = None
        kalman_vals = []
        for index, row in group.iterrows():
            if not is_valid_measurement(row.get('FuelLevel'), row.get('FeatureStatus', '')):
                kalman_vals.append(np.nan)
                kf = None # Bắt buộc reset
                continue
                
            measurement = float(row['FuelLevel'])
            if kf is None:
                kf = StandardKalmanFilter1D(initial_state=measurement, process_noise=1.0, measurement_noise=9.0)
                kalman_vals.append(measurement)
            else:
                kalman_vals.append(kf.update(measurement))
                
        kalman_vals = np.array(kalman_vals)
        kal_rmse_real, kal_snr_real, kal_max_real = get_dsp_metrics(fuels, kalman_vals, shift_steps=0)
        
        # 4. Adaptive Kalman (Innovation Gating, default parameters)
        kf_adapt = None
        kalman_adapt_vals = []
        reference_gap = 5.0
        for index, row in group.iterrows():
            if not is_valid_measurement(row.get('FuelLevel'), row.get('FeatureStatus', '')):
                kalman_adapt_vals.append(np.nan)
                kf_adapt = None
                continue
                
            measurement = float(row['FuelLevel'])
            time_gap = row.get("TimeGapMinutes", reference_gap)
            if pd.isna(time_gap) or time_gap <= 0:
                time_gap = reference_gap
            dt_ratio = time_gap / reference_gap
            
            # Feature: MovementState
            movement_state_raw = row.get("MovementState", "Moving")
            if pd.isna(movement_state_raw):
                movement_state = 1
            else:
                movement_state_str = str(movement_state_raw).strip().upper()
                movement_state = 0 if movement_state_str == "STOPPED" else 1
            
            if kf_adapt is None:
                kf_adapt = AdaptiveKalmanFilter1D(
                    initial_state=measurement, 
                    initial_estimate_error=9.0,
                    process_noise=1.0, 
                    innovation_threshold=10.0,
                    persistence_required=3
                )
                kalman_adapt_vals.append(measurement)
            else:
                kalman_adapt_vals.append(kf_adapt.update(measurement, dt_ratio=dt_ratio, movement_state=movement_state))
                
        kalman_adapt_vals = np.array(kalman_adapt_vals)
        adapt_rmse_real, adapt_snr_real, adapt_max_real = get_dsp_metrics(fuels, kalman_adapt_vals, shift_steps=0)
        
        results.append({
            'Algorithm': 'Moving Average',
            'MAD (Độ mượt)': calculate_smoothness(ma_vals),
            'RMSE (Real-time)': ma_rmse_real,
            'Max Error (Real-time)': ma_max_real,
            'RMSE (Shape Aligned)': ma_rmse_shape,
            'Max Error (Shape Aligned)': ma_max_shape
        })
        results.append({
            'Algorithm': 'Median Filter',
            'MAD (Độ mượt)': calculate_smoothness(med_vals),
            'RMSE (Real-time)': med_rmse_real,
            'Max Error (Real-time)': med_max_real,
            'RMSE (Shape Aligned)': med_rmse_shape,
            'Max Error (Shape Aligned)': med_max_shape
        })
        results.append({
            'Algorithm': 'Standard Kalman (Q=1, R=9)',
            'MAD (Độ mượt)': calculate_smoothness(kalman_vals),
            'RMSE (Real-time)': kal_rmse_real,
            'Max Error (Real-time)': kal_max_real,
            'RMSE (Shape Aligned)': kal_rmse_real,
            'Max Error (Shape Aligned)': kal_max_real
        })
        results.append({
            'Algorithm': 'Adaptive Kalman (Dynamic R)',
            'MAD (Độ mượt)': calculate_smoothness(kalman_adapt_vals),
            'RMSE (Real-time)': adapt_rmse_real,
            'Max Error (Real-time)': adapt_max_real,
            'RMSE (Shape Aligned)': adapt_rmse_real,
            'Max Error (Shape Aligned)': adapt_max_real
        })

    df_res = pd.DataFrame(results)
    avg_res = df_res.groupby('Algorithm').mean().reset_index()
    
    print("\n================ BẢNG ĐÁNH GIÁ CHUẨN KỸ SƯ DSP ================")
    print(avg_res.to_string(index=False))
    print("================================================================")

if __name__ == '__main__':
    evaluate()
