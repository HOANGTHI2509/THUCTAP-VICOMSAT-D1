import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import numpy as np
import pandas as pd
from src.core.filters.moving_average import moving_average_filter
from src.core.filters.median_filter import median_filter
from src.core.filters.kalman_traditional import StandardKalmanFilter1D, is_valid_measurement
from src.core.filters.kalman_adaptive import AdaptiveKalmanFilter1D

# Metrics functions
def calc_mean_delta(y):
    y = np.array(y)
    valid_y = y[~np.isnan(y)]
    if len(valid_y) < 2: return 0.0
    return np.mean(np.abs(np.diff(valid_y)))

def calc_tracking_rmse(y, r):
    y = np.array(y)
    r = np.array(r)
    mask = ~np.isnan(y) & ~np.isnan(r)
    if np.sum(mask) == 0: return 0.0
    return np.sqrt(np.mean((y[mask] - r[mask])**2))

def calc_max_dev(y, r):
    y = np.array(y)
    r = np.array(r)
    mask = ~np.isnan(y) & ~np.isnan(r)
    if np.sum(mask) == 0: return 0.0
    return np.max(np.abs(y[mask] - r[mask]))

df = pd.read_csv('data/processed/CarFuelHistory_Processed_Car1.csv')
# Sort properly
df["_OriginalOrder"] = np.arange(len(df))
df = df.sort_values(["SegmentID", "FuelTime", "_OriginalOrder"])

ma_window = 10
med_window = 10
kalman_r = 9

results = {
    'Method': ['Raw Data', f'Moving Average (N={ma_window})', f'Median Filter (N={med_window})', f'Standard Kalman (R={kalman_r})', 'Adaptive Kalman'],
    'Mean_Delta': [],
    'Tracking_RMSE': [],
    'Max_Deviation': []
}

all_raw = []
all_ma = []
all_med = []
all_kalman = []
all_adapt = []

for seg_id, group in df.groupby('SegmentID'):
    fuels = group['FuelLevel'].tolist()
    all_raw.extend(fuels)
    
    ma_vals = moving_average_filter(fuels, ma_window)
    all_ma.extend(ma_vals)
    
    med_vals = median_filter(fuels, med_window)
    all_med.extend(med_vals)
    
    kal_vals = []
    kf = None
    for _, row in group.iterrows():
        if not is_valid_measurement(row.get('FuelLevel'), row.get('FeatureStatus', '')):
            kal_vals.append(np.nan)
            kf = None
        else:
            z = float(row['FuelLevel'])
            if kf is None:
                kf = StandardKalmanFilter1D(initial_state=z, process_noise=1.0, measurement_noise=kalman_r)
                kal_vals.append(z)
            else:
                kal_vals.append(kf.update(z))
    all_kalman.extend(kal_vals)
    
    adapt_vals = []
    kf_adapt = None
    for _, row in group.iterrows():
        if not is_valid_measurement(row.get('FuelLevel'), row.get('FeatureStatus', '')):
            adapt_vals.append(np.nan)
            kf_adapt = None
        else:
            z = float(row['FuelLevel'])
            tg = row.get("TimeGapMinutes", 5.0)
            if pd.isna(tg) or tg <= 0: tg = 5.0
            dt = tg / 5.0
            
            m_state = str(row.get("MovementState", "MOVING")).upper().strip()
            movement_state = 0 if m_state == "STOPPED" else 1
            
            if kf_adapt is None:
                kf_adapt = AdaptiveKalmanFilter1D(initial_state=z, initial_estimate_error=9.0, process_noise=1.0, r_base=9.0, innovation_threshold=10.0, persistence_required=3)
                adapt_vals.append(z)
            else:
                adapt_vals.append(kf_adapt.update(z, dt_ratio=dt, movement_state=movement_state))
    all_adapt.extend(adapt_vals)

y_list = [all_raw, all_ma, all_med, all_kalman, all_adapt]

for y in y_list:
    results['Mean_Delta'].append(calc_mean_delta(y))
    results['Tracking_RMSE'].append(calc_tracking_rmse(y, all_raw))
    results['Max_Deviation'].append(calc_max_dev(y, all_raw))

res_df = pd.DataFrame(results)
print(res_df.to_string(float_format="%.3f"))
