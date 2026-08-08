import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from src.core.filters.kalman_adaptive import AdaptiveKalmanFilter1D

def is_valid_measurement(fuel_level, feature_status):
    if pd.isna(fuel_level) or fuel_level == 0:
        return False
    if pd.isna(feature_status):
        return True
    return 'Invalid' not in str(feature_status)

df = pd.read_csv('data/processed/CarFuelHistory_Processed_Car1.csv')
df['FuelTime'] = pd.to_datetime(df['FuelTime'])
df.sort_values(by=['SegmentID', 'FuelTime'], inplace=True)

max_global_err = 0
max_info = ""

# To hold the max info for plotting
best_group = None
best_raw = None
best_kf = None
best_max_idx = -1
best_kalman_vals = None
best_err_val = 0

for segment_id, group in df.groupby('SegmentID'):
    fuels = group['FuelLevel'].values
    if len(fuels) < 50: continue
    
    kf_adapt = None
    kalman_vals = []
    reference_gap = 5.0
    for idx, row in group.iterrows():
        if not is_valid_measurement(row.get('FuelLevel'), row.get('FeatureStatus', '')):
            kalman_vals.append(np.nan)
            kf_adapt = None
            continue
            
        meas = float(row['FuelLevel'])
        time_gap = row.get("TimeGapMinutes", reference_gap)
        if pd.isna(time_gap) or time_gap <= 0:
            time_gap = reference_gap
        dt_ratio = time_gap / reference_gap
        
        movement_state_raw = row.get("MovementState", "Moving")
        if pd.isna(movement_state_raw):
            movement_state = 1
        else:
            movement_state_str = str(movement_state_raw).strip().upper()
            movement_state = 0 if movement_state_str == "STOPPED" else 1

        if kf_adapt is None:
            kf_adapt = AdaptiveKalmanFilter1D(
                initial_state=meas, 
                initial_estimate_error=9.0,
                process_noise=1.0, 
                innovation_threshold=10.0,
                persistence_required=3
            )
            kalman_vals.append(meas)
        else:
            kalman_vals.append(kf_adapt.update(meas, dt_ratio=dt_ratio, movement_state=movement_state))
            
    kalman_adapt_vals = np.array(kalman_vals)
    
    aligned_filtered = kalman_adapt_vals.copy()
    valid_features = np.array(['Invalid' not in str(fs) for fs in group.get('FeatureStatus', [''] * len(fuels))])
    
    mask = ~np.isnan(fuels) & ~np.isnan(aligned_filtered) & (fuels > 0) & valid_features
    mask[:10] = False
    
    if not np.any(mask): continue
        
    diffs = np.zeros(len(fuels))
    diffs[mask] = np.abs(fuels[mask] - aligned_filtered[mask])
    seg_max_err = np.max(diffs)
    
    if seg_max_err > best_err_val:
        best_err_val = seg_max_err
        best_max_idx = np.argmax(diffs)
        best_group = group
        best_kalman_vals = kalman_adapt_vals
        best_raw = fuels

print(f"Max Error: {best_err_val:.4f} at Time: {best_group.iloc[best_max_idx]['FuelTime']}, Raw: {best_raw[best_max_idx]}, KF: {best_kalman_vals[best_max_idx]}")

start_idx = max(0, best_max_idx - 50)
end_idx = min(len(best_group), best_max_idx + 50)

window = best_group.iloc[start_idx:end_idx]
window_kf = best_kalman_vals[start_idx:end_idx]

plt.figure(figsize=(12, 6))
plt.plot(window['FuelTime'], window['FuelLevel'], label='Raw Fuel', color='gray', alpha=0.5, marker='.')
plt.plot(window['FuelTime'], window_kf, label='Adaptive Kalman (Gating+Movement)', color='red', linestyle='dashed', linewidth=2)

time_val = best_group.iloc[best_max_idx]['FuelTime']
raw_val = best_raw[best_max_idx]
kf_val = best_kalman_vals[best_max_idx]

# highlight max error point
plt.scatter([time_val], [raw_val], color='black', s=100, zorder=5, label='Raw at Max Error')
plt.scatter([time_val], [kf_val], color='red', s=100, zorder=5, label='KF at Max Error')

plt.title(f'Vùng Xảy ra Max Error (Lỗi cực đại: {best_err_val:.1f} Lít)')
plt.xlabel('Thời gian')
plt.ylabel('Mức xăng (Lít)')
plt.legend()
plt.grid(True)

import os
artifact_dir = r"C:\Users\Admin\.gemini\antigravity-ide\brain\b9a1babd-7697-47da-a007-f5dc6eb5d79c"
out_path = os.path.join(artifact_dir, "max_error_chart.png")
plt.savefig(out_path)
print(f"Saved plot to {out_path}")
