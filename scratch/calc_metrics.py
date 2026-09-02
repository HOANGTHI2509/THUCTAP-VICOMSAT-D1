import sys
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

sys.path.append(r"d:\THUCTAP_VICOMSAT")
from src.core.filters import kalman_traditional as kalman
from src.core.filters.kalman_adaptive import BoLocKalmanThichNghi1D, is_valid_measurement
from src.dashboard.app_dashboard_tienxuly import load_ai_state_model, load_tcn_state_model, classify_signal_modes, clean_transient_shapes
from src.core.filters.fuel_state_filter import filter_fuel_series
from src.core.filters.ai_state_filter import filter_with_ai_state

ai_state_model, ai_state_metadata = load_ai_state_model()
tcn_state_model, tcn_state_metadata = load_tcn_state_model()

def calc_smoothness(series):
    s = pd.Series(series).dropna().to_numpy()
    if len(s) < 2: return 0.0
    return np.mean(np.abs(np.diff(s)))

def rmse(a, b):
    mask = (~np.isnan(a)) & (~np.isnan(b))
    return np.sqrt(np.mean((a[mask] - b[mask])**2))

df = pd.read_csv(r"D:\THUCTAP_VICOMSAT\data\synthetic_dataset\synthetic_dataset.csv").head(20000)
df['FuelTime'] = pd.to_datetime('2026-01-01') + pd.to_timedelta(df['TimeGapMinutes'].fillna(5).cumsum(), unit='m')
if 'Speed' not in df.columns: df['Speed'] = df.get('MotionSpeed', 0)
df['SegmentID'] = df.get('SegmentID', 0)
df['FuelLevel'] = df['NoisyFuel']
df['CleanedFuel'] = df['CleanFuel']

class Profile: pass
profile = Profile()
capacity = 200.0
profile.capacity_est = capacity
profile.noise_sigma_liters = max(0.5, 0.002 * capacity)
profile.flat_jitter_threshold = max(0.8, 0.003 * capacity)
profile.spike_threshold = max(3.0, 0.012 * capacity)
profile.event_threshold = max(6.0, 0.035 * capacity)

group = df.copy()
group['TimeGapMinutes'] = group['TimeGapMinutes'].fillna(5.0)
if "Acceleration" not in group.columns:
    group["Acceleration"] = group["Speed"].diff().fillna(0.0) / (group['TimeGapMinutes'] * 60.0).replace(0, 1.0)
group["DeltaFuel"] = group["FuelLevel"].diff().fillna(0.0)

group = classify_signal_modes(group, profile, source_col="FuelLevel", output_col="ProfileCleanFuel", mode_col="SignalMode", lookback=7, lookahead=5)
group = filter_fuel_series(group, profile, source_col="FuelLevel", lookback=7, lookahead=5)
group = clean_transient_shapes(group, profile, source_col="FuelLevel", output_col="ShapeCleanFuel", flag_col="ShapeCleanFlag", max_points=12)
group = filter_with_ai_state(group, model=ai_state_model, metadata=ai_state_metadata, tcn_model=tcn_state_model, tcn_metadata=tcn_state_metadata, profile=profile)

adaptive_source_col = "ShapeCleanFuel"
kf_std = None
kf_adapt = None
kalman_std_vals = []
kalman_adapt_vals = []
x_f, P_f, x_p, P_p = [], [], [], []
kalman_r = int(max(64.0, (0.04 * capacity)**2))

for dong in group.itertuples():
    measurement = getattr(dong, adaptive_source_col)
    raw = getattr(dong, "FuelLevel")
    gap = getattr(dong, "TimeGapMinutes", 5.0)
    if pd.isna(measurement): measurement = raw
    
    if not is_valid_measurement(measurement, getattr(dong, 'FeatureStatus', '')):
        kalman_std_vals.append(np.nan)
        kalman_adapt_vals.append(np.nan)
        x_f.append(np.nan); P_f.append(0); x_p.append(np.nan); P_p.append(0)
        continue
        
    if kf_std is None:
        kf_std = kalman.BoLocKalmanTieuChuan1D(trang_thai_ban_dau=measurement, nhieu_qua_trinh=1.0, nhieu_do_luong=kalman_r)
        kalman_std_vals.append(measurement)
    else:
        kalman_std_vals.append(kf_std.cap_nhat(measurement, ty_le_dt=gap/5.0))
        
    if kf_adapt is None:
        kf_adapt = BoLocKalmanThichNghi1D(trang_thai_ban_dau=measurement, capacity=capacity, sai_so_uoc_luong_ban_dau=4.0, nhieu_qua_trinh=0.2, r_co_ban=9.0, nhip_cho_xac_nhan=3)
        kalman_adapt_vals.append(measurement)
        x_f.append(measurement); P_f.append(4.0); x_p.append(measurement); P_p.append(4.0)
    else:
        v = float(getattr(dong, 'Speed', 0.0))
        a = float(getattr(dong, 'Acceleration', 0.0))
        rstd = float(getattr(dong, 'RollingStd', 0.0))
        val = kf_adapt.cap_nhat(measurement, ty_le_dt=gap/5.0, trang_thai_chuyen_dong=1 if v>3 else 0, van_toc=v, gia_toc=a, rolling_std=rstd)
        kalman_adapt_vals.append(val)
        x_f.append(kf_adapt.x); P_f.append(kf_adapt.P); x_p.append(kf_adapt.x); P_p.append(kf_adapt.P)

group['Custom_Kalman'] = kalman_std_vals
group['Custom_Adaptive_Kalman'] = kalman_adapt_vals

raw = group["FuelLevel"].to_numpy(dtype=float)
base = group["Custom_Adaptive_Kalman"].to_numpy(dtype=float)
states = group["AI_State"].astype(str).to_numpy()
flat = group["flat_jitter_threshold"].to_numpy(dtype=float)
rolling_std = group["FuelLevel"].rolling(window=12, min_periods=1).std().fillna(0.0).to_numpy(dtype=float)
noise_sigma = group["noise_sigma_liters"].to_numpy(dtype=float)
event_threshold = group["event_threshold"].to_numpy(dtype=float)
speed = group["Speed"].to_numpy(dtype=float)
enhanced = np.full(len(group), np.nan, dtype=float)
output = np.nan
pending_drain = None
recent_base = []
noise_hold = 0
noise_anchor = np.nan
noise_dir = 0
noise_dir_count = 0
refuel_settle = 0
refuel_floor = np.nan
refuel_drop_count = 0

for i in range(len(group)):
    z = raw[i]; b = base[i]; state = states[i]
    jitter = max(float(flat[i]), 0.1); noise = max(float(noise_sigma[i]), 0.1)
    event = max(float(event_threshold[i]), jitter*5, noise*4, 2.0)
    current_speed = max(float(speed[i]), 0.0)
    high_noise = float(rolling_std[i]) >= max(jitter*1.5, noise*3.0)
    
    if pd.isna(z) or z <= 0: enhanced[i] = output; continue
    if pd.isna(b): b = z
    recent_base.append(float(b)); 
    if len(recent_base)>15: recent_base.pop(0)
    base_median = float(np.median(recent_base))
    
    if high_noise or state == "SLOSHING_NOISE":
        if noise_hold == 0 or pd.isna(noise_anchor):
            noise_anchor = float(output) if not pd.isna(output) else float(b)
            noise_dir = 0; noise_dir_count = 0
        noise_hold = 3
    else:
        noise_hold = max(0, noise_hold-1)
        if noise_hold == 0: noise_anchor = np.nan; noise_dir = 0; noise_dir_count = 0
        
    if pd.isna(output): output = float(b); enhanced[i] = output; continue
    
    if state == "SPIKE": target = output; alpha = 0.0; pending_drain = None
    elif state == "REFUEL":
        next_window = raw[i+1:i+4]; next_window = next_window[~np.isnan(next_window)]
        future_median = float(np.median(next_window)) if len(next_window) else float(z)
        delta_up = float(z) - output
        refuel_gate = max(event*0.55, jitter*4, noise*3, 2.0)
        future_stays_high = (len(next_window)>0 and future_median>=output+refuel_gate*0.6 and abs(future_median-float(z))<=max(abs(delta_up)*0.45, event*0.6, jitter*5))
        slow_or_stopped_confirm = (current_speed<=3.0 and len(next_window)>0 and future_median>=output+refuel_gate*0.45)
        if delta_up>=refuel_gate and (future_stays_high or slow_or_stopped_confirm):
            target = max(float(z), future_median); alpha = 0.96; refuel_floor = target - max(event*0.2, jitter*2, noise*2, 1.0)
            refuel_settle = 4; refuel_drop_count = 0; noise_hold = 0; noise_anchor = np.nan; noise_dir = 0; noise_dir_count = 0
        else: target = output; alpha = 0.0
        pending_drain = None
    elif state == "DRAIN":
        refuel_settle = 0; refuel_floor = np.nan; refuel_drop_count = 0
        if pending_drain is None: pending_drain = float(z); target = min(float(b), float(z)); alpha = 0.45
        else: target = min(float(b), float(z), pending_drain); alpha = 0.90
        noise_hold = 0; noise_anchor = np.nan; noise_dir = 0; noise_dir_count = 0
    elif state == "SLOSHING_NOISE": target = base_median; alpha = 0.035; pending_drain = None
    elif state == "STABLE_JITTER":
        if abs(z-output) <= jitter*0.35: target = output; alpha = 0.0
        else: target = float(b); alpha = 0.85
        pending_drain = None
    else: target = float(b); alpha = 1.0; pending_drain = None

    in_noise_episode = noise_hold > 0 and state not in {"REFUEL", "DRAIN", "SPIKE", "CONSUMPTION"}
    if in_noise_episode:
        if pd.isna(noise_anchor): noise_anchor = output
        deviation_from_anchor = base_median - noise_anchor
        raw_deviation_from_anchor = float(z) - noise_anchor
        current_dir = int(np.sign(deviation_from_anchor)) if abs(deviation_from_anchor) > jitter else 0
        if current_dir != 0 and current_dir == noise_dir: noise_dir_count += 1
        elif current_dir != 0: noise_dir = current_dir; noise_dir_count = 1
        else: noise_dir = 0; noise_dir_count = 0
        
        sustained_level_shift = (noise_dir_count>=3 and current_dir!=0 and abs(deviation_from_anchor)>=max(event*0.35, jitter*3, noise*2.5) and abs(raw_deviation_from_anchor)>=max(event*0.35, jitter*3, noise*2.5) and np.sign(raw_deviation_from_anchor)==current_dir)
        if sustained_level_shift: noise_hold = 0; noise_anchor = np.nan; target = base_median; alpha = max(alpha, 0.55); in_noise_episode = False
        elif noise_dir_count >= 8: target = base_median; alpha = min(alpha, 0.035)
        else: target = noise_anchor; alpha = min(alpha, 0.015)
        
    next_output = output + alpha * (target - output)
    if refuel_settle > 0 and state not in {"DRAIN", "SPIKE"} and not pd.isna(refuel_floor):
        if float(z) < refuel_floor - max(event*0.25, jitter*2, noise*2): refuel_drop_count += 1
        else: refuel_drop_count = 0
        if refuel_drop_count < 2: next_output = max(next_output, refuel_floor); refuel_settle -= 1
        else: refuel_settle = 0; refuel_floor = np.nan; refuel_drop_count = 0
        
    if in_noise_episode:
        max_step = max(jitter*0.12, noise*0.2, 0.06)
        next_output = float(np.clip(next_output, output-max_step, output+max_step))
    output = next_output
    if output < 0: output = 0.0
    enhanced[i] = output

group['AI_Enhanced_Kalman'] = enhanced
metrics = ["Model|RMSE (L)|Smoothness (L/step)|SNR (dB)"]
true_fuel = group['CleanedFuel'].to_numpy() if 'CleanedFuel' in group.columns else group['TrueFuel'].to_numpy()
signal_power = np.mean(true_fuel**2)

for col, name in [("FuelLevel", "Raw"), ("Custom_Kalman", "Standard Kalman"), ("Custom_Adaptive_Kalman", "Adaptive Kalman"), ("AI_Enhanced_Kalman", "AI Enhanced Kalman")]:
    s = group[col].to_numpy()
    sm = calc_smoothness(s)
    r = rmse(s, true_fuel)
    snr = 10 * np.log10(signal_power / (r**2 + 1e-6))
    metrics.append(f"{name}|{r:.2f}|{sm:.2f}|{snr:.2f}")

print("\n".join(metrics))
