import pandas as pd
import numpy as np
from src.core.filters.kalman_adaptive import BoLocKalmanThichNghi1D, rts_smooth_1d
from src.core.filters.kalman_traditional import BoLocKalmanTieuChuan1D
from src.core.filters.anomaly_detector import FuelAnomalyDetector

df = pd.read_csv('data/raw/CarFuelHistory_New.csv')
df['FuelTime'] = pd.to_datetime(df['FuelTime'])

cars = df['VehicleID'].unique()[:5]

results = []

for car_id in cars:
    group = df[df['VehicleID'] == car_id].copy().sort_values('FuelTime')
    if len(group) == 0: continue
    
    # Simple Movement State and Gap
    group['TimeGapMinutes'] = group['FuelTime'].diff().dt.total_seconds() / 60.0
    group['TimeGapMinutes'] = group['TimeGapMinutes'].fillna(5.0)
    group['MovementState'] = np.where(group['Speed'] > 0, 1, 0)
    
    # Anomaly Detection (Tầng 1)
    estimated_capacity = group['FuelLevel'].max() * 1.1
    if pd.isna(estimated_capacity) or estimated_capacity < 50: estimated_capacity = 100.0
    detector = FuelAnomalyDetector(capacity=estimated_capacity)
    group = detector.detect_and_clean(group)
    
    # Rolling Std for Adaptive
    group['RollingStd_Noise'] = group['CleanedFuel'].rolling(window=12, min_periods=1).std().fillna(0)
    
    # Setup Filters
    r_base = max(64.0, (0.04 * estimated_capacity)**2)
    threshold = max(15.0, 0.05 * estimated_capacity)
    
    kf_std = BoLocKalmanTieuChuan1D(trang_thai_ban_dau=group['CleanedFuel'].iloc[0], nhieu_qua_trinh=1.0, nhieu_do_luong=r_base)
    kf_adapt = BoLocKalmanThichNghi1D(
        trang_thai_ban_dau=group['CleanedFuel'].iloc[0],
        nhieu_qua_trinh=1.0,
        r_co_ban=r_base,
        r_nhieu_dot_bien=r_base * 2.0,
        nguong_bat_nhay_co_ban=threshold,
        nguong_toi_da=max(25.0, 0.125 * estimated_capacity)
    )
    
    std_vals = []
    x_f, P_f, x_p, P_p = [], [], [], []
    
    tich_luy_std = 0.0
    tich_luy_adapt = 0.0
    
    for row in group.itertuples():
        gap = row.TimeGapMinutes
        meas = row.CleanedFuel
        if pd.isna(meas):
            tich_luy_std += gap
            tich_luy_adapt += gap
            std_vals.append(np.nan)
            x_f.append(np.nan); P_f.append(0.0); x_p.append(np.nan); P_p.append(0.0)
            continue
            
        # Standard
        dt_std = (gap + tich_luy_std) / 5.0
        tich_luy_std = 0.0
        std_vals.append(kf_std.cap_nhat(meas, ty_le_dt=dt_std))
        
        # Adaptive
        dt_adp = (gap + tich_luy_adapt) / 5.0
        tich_luy_adapt = 0.0
        xf, Pf, xp, Pp = kf_adapt.cap_nhat(
            meas, ty_le_dt=dt_adp, trang_thai_chuyen_dong=row.MovementState, 
            rolling_std=row.RollingStd_Noise, van_toc=row.Speed
        )
        x_f.append(xf); P_f.append(Pf); x_p.append(xp); P_p.append(Pp)
        
    x_f_arr = pd.Series(x_f).ffill().bfill().values
    P_f_arr = pd.Series(P_f).ffill().bfill().values
    x_p_arr = pd.Series(x_p).ffill().bfill().values
    P_p_arr = pd.Series(P_p).ffill().bfill().values
    
    adapt_rts_vals = rts_smooth_1d(x_f_arr, P_f_arr, x_p_arr, P_p_arr)
    
    group['Std_Kalman'] = std_vals
    group['Adapt_RTS'] = adapt_rts_vals
    
    # Calculate Metrics
    # Filter out nans
    valid = group.dropna(subset=['CleanedFuel', 'Std_Kalman', 'Adapt_RTS'])
    
    def calc_tv(s): return np.abs(np.diff(s)).mean()
    def calc_mae(s1, s2): return np.abs(s1 - s2).mean()
    
    raw_tv = calc_tv(valid['CleanedFuel'])
    
    std_tv = calc_tv(valid['Std_Kalman'])
    std_mae = calc_mae(valid['Std_Kalman'], valid['CleanedFuel'])
    
    adp_tv = calc_tv(valid['Adapt_RTS'])
    adp_mae = calc_mae(valid['Adapt_RTS'], valid['CleanedFuel'])
    
    results.append({
        'Car ID': car_id,
        'Dung tích': int(estimated_capacity),
        'Nhiễu Gốc (TV)': round(raw_tv, 3),
        'Std Kalman - Mượt (TV)': round(std_tv, 3),
        'Std Kalman - Bám sát (MAE)': round(std_mae, 3),
        'Adapt RTS - Mượt (TV)': round(adp_tv, 3),
        'Adapt RTS - Bám sát (MAE)': round(adp_mae, 3)
    })

res_df = pd.DataFrame(results)
print(res_df.to_markdown(index=False))
