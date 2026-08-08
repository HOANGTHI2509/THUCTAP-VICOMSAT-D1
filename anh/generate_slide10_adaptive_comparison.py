import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.dates as mdates
import matplotlib.patches as patches

# Load data
df = pd.read_csv('data/processed/CarFuelHistory_Processed_Car1.csv')
df['FuelTime'] = pd.to_datetime(df['FuelTime'])

# Filter a specific time window that has both a spike and a real drop
start_time = pd.to_datetime('2026-03-07 12:00:00')
end_time = pd.to_datetime('2026-03-07 16:00:00')
df_plot = df[(df['FuelTime'] >= start_time) & (df['FuelTime'] <= end_time)].copy()

# Run filters manually
from src.core.filters.kalman_traditional import StandardKalmanFilter1D
from src.core.filters.kalman_adaptive import AdaptiveKalmanFilter1D

kf_std = None
kf_adapt = None
std_vals = []
adapt_vals = []

for idx, row in df_plot.iterrows():
    val = float(row['FuelLevel'])
    if pd.isna(val):
        std_vals.append(np.nan)
        adapt_vals.append(np.nan)
        continue
    
    # Standard
    if kf_std is None:
        kf_std = StandardKalmanFilter1D(initial_state=val, process_noise=1.0, measurement_noise=9.0)
        std_vals.append(val)
    else:
        std_vals.append(kf_std.update(val))
        
    # Adaptive
    if kf_adapt is None:
        kf_adapt = AdaptiveKalmanFilter1D(initial_state=val, initial_estimate_error=9.0, process_noise=1.0, r_base=9.0, r_spike=49.0, innovation_threshold=10.0, persistence_required=3)
        adapt_vals.append(val)
    else:
        m_state = row.get("MovementState", "MOVING")
        movement_state = 0 if m_state == "STOPPED" else 1
        dt_ratio = row.get("TimeGapMinutes", 5.0) / 5.0
        if pd.isna(dt_ratio) or dt_ratio <= 0: dt_ratio = 1.0
        adapt_vals.append(kf_adapt.update(val, dt_ratio=dt_ratio, movement_state=movement_state))

df_plot['Kalman_Standard'] = std_vals
df_plot['Kalman_Adaptive'] = adapt_vals

# Plotting
plt.figure(figsize=(14, 7), facecolor='#F8F9FA')
ax = plt.gca()
ax.set_facecolor('#F8F9FA')

# Lines
plt.plot(df_plot['FuelTime'], df_plot['FuelLevel'], color='red', alpha=0.5, marker='.', linestyle='-', label='Raw FuelLevel', zorder=1)
plt.plot(df_plot['FuelTime'], df_plot['Kalman_Standard'], color='#00CC96', linewidth=2, label='Standard Kalman (R=9)', zorder=2)
plt.plot(df_plot['FuelTime'], df_plot['Kalman_Adaptive'], color='#1f77b4', linewidth=2.5, linestyle='--', label='Adaptive Kalman', zorder=3)

# Formatting
plt.title("Kết quả cải thiện: Adaptive Kalman so với Standard Kalman (Ngày 07/03/2026)", fontsize=16, fontweight='bold', pad=20)
plt.xlabel("Thời gian", fontsize=12)
plt.ylabel("Mức nhiên liệu (Lít)", fontsize=12)
plt.grid(True, linestyle='--', alpha=0.5)

# X-axis formatting
ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
plt.xticks(rotation=0)

# ================= HIGHLIGHT REGIONS =================

# Real drop region
real_drop_start = pd.to_datetime('2026-03-07 13:40:00')
real_drop_end = pd.to_datetime('2026-03-07 14:15:00')
ax.axvspan(real_drop_start, real_drop_end, color='blue', alpha=0.1, label='Vùng thay đổi mức thực (Nạp/Rút)')

# Spike region
spike_start = pd.to_datetime('2026-03-07 13:10:00')
spike_end = pd.to_datetime('2026-03-07 13:25:00')
ax.axvspan(spike_start, spike_end, color='red', alpha=0.1, label='Vùng nhiễu (Spike / Khung nhiễu mạnh)')

# Annotations
plt.annotate('Adaptive Kalman phớt lờ nhiễu\n(Giữ đường tín hiệu ổn định)', 
             xy=(pd.to_datetime('2026-03-07 13:17:00'), 250), xytext=(pd.to_datetime('2026-03-07 12:40:00'), 280),
             arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=8),
             fontsize=11, bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="red", lw=1.5))

plt.annotate('Adaptive Kalman BỨT TỐC bám theo\nthực tế ngay lập tức (Không bị trễ)', 
             xy=(pd.to_datetime('2026-03-07 13:50:00'), 280), xytext=(pd.to_datetime('2026-03-07 14:20:00'), 320),
             arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=8),
             fontsize=11, bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="blue", lw=1.5))

plt.annotate('Standard Kalman (Xanh lá)\nvẫn bị bo tròn và trễ', 
             xy=(pd.to_datetime('2026-03-07 14:00:00'), 305), xytext=(pd.to_datetime('2026-03-07 14:30:00'), 250),
             arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=8),
             fontsize=11, bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", lw=1.5))

plt.legend(loc='lower left', fontsize=12)
plt.tight_layout()

# Save image
plt.savefig('anh/Slide10_Adaptive_Comparison.png', dpi=300, bbox_inches='tight')
print("Saved Slide10_Adaptive_Comparison.png")
