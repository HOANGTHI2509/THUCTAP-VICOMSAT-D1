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

# Filter specific time window
start_time = pd.to_datetime('2026-03-07 13:00:00')
end_time = pd.to_datetime('2026-03-07 16:00:00')
df_plot = df[(df['FuelTime'] >= start_time) & (df['FuelTime'] <= end_time)].copy()

# 1. Moving Average (N=10) with Burn-in
N = 10
ma_vals = []
window = []
for val in df_plot['FuelLevel']:
    window.append(val)
    if len(window) > N:
        window.pop(0)
    ma_vals.append(sum(window) / len(window))
df_plot['MA_10'] = ma_vals

# 2. Standard Kalman (Q=1, R=9)
from src.core.filters.kalman_traditional import StandardKalmanFilter1D
kf = StandardKalmanFilter1D(initial_state=df_plot['FuelLevel'].iloc[0], process_noise=1.0, measurement_noise=9.0)
kf_vals = []
for val in df_plot['FuelLevel']:
    kf_vals.append(kf.update(val))
df_plot['Kalman_9'] = kf_vals

# Plotting
plt.figure(figsize=(14, 7), facecolor='#F8F9FA')
ax = plt.gca()
ax.set_facecolor('#F8F9FA')

# Lines
plt.plot(df_plot['FuelTime'], df_plot['FuelLevel'], color='gray', alpha=0.3, marker='.', linestyle='-', label='Raw FuelLevel', zorder=1)
plt.plot(df_plot['FuelTime'], df_plot['MA_10'], color='#FFA15A', linewidth=2, label='Moving Average (N=10)', zorder=2)
plt.plot(df_plot['FuelTime'], df_plot['Kalman_9'], color='#00CC96', linewidth=2, label='Standard Kalman (Q=1, R=9)', zorder=3)

# Formatting
plt.title("So sánh trực quan: Moving Average vs Standard Kalman", fontsize=16, fontweight='bold', pad=20)
plt.xlabel("Thời gian", fontsize=12)
plt.ylabel("Mức nhiên liệu (Lít)", fontsize=12)
plt.grid(True, linestyle='--', alpha=0.5)

# X-axis formatting
ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
plt.xticks(rotation=0)

# ================= ANNOTATIONS =================

# 1. Khúc nhảy vọt (Jump)
jump_time = pd.to_datetime('2026-03-07 13:45:00')
plt.annotate('Kalman vọt lên nhanh hơn MA\n(Phản ứng nhạy hơn)', 
             xy=(jump_time, 240), xytext=(pd.to_datetime('2026-03-07 13:55:00'), 150),
             arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=8),
             fontsize=11, bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#00CC96", lw=1.5))

# 2. Khúc bo tròn trên đỉnh
top_curve_time = pd.to_datetime('2026-03-07 13:55:00')
plt.annotate('Kalman vẫn bị cong vòng,\nmất thời gian để tiệm cận đỉnh\n(Trễ do R=9 cố định)', 
             xy=(top_curve_time, 305), xytext=(pd.to_datetime('2026-03-07 14:15:00'), 250),
             arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=8),
             fontsize=11, bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", lw=1))

# 3. Vùng dao động ngang
flat_time = pd.to_datetime('2026-03-07 14:30:00')
plt.annotate('Cả hai thuật toán đều\nlàm mượt tốt dao động nhỏ', 
             xy=(flat_time, 298), xytext=(pd.to_datetime('2026-03-07 14:40:00'), 220),
             arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=8),
             fontsize=11, bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#FFA15A", lw=1.5))

plt.legend(loc='lower right', fontsize=12)
plt.tight_layout()

# Save image
plt.savefig('anh/Slide7_Visual_Comparison.png', dpi=300, bbox_inches='tight')
print("Saved Slide7_Visual_Comparison.png")
