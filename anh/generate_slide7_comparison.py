import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import pandas as pd
from src.core.filters.kalman_traditional import StandardKalmanFilter1D

# 1. Đọc dữ liệu thật
df = pd.read_csv('data/processed/CarFuelHistory_Processed_Car1.csv')
raw_data_segment = df['FuelLevel'].iloc[1500:1800].values.copy()
time_segment = pd.to_datetime(df['FuelTime'].iloc[1500:1800])

# Cố tình tạo 1 spike nhỏ tại index 220 để dễ bề minh họa "Điểm bất thường ngắn"
# (Nếu dữ liệu thật không có spike rõ ràng trong đoạn này)
raw_data_segment[220] = raw_data_segment[220] - 60
raw_fuel = raw_data_segment

# 2. Tính Moving Average (N = 10)
N = 10
ma_fuel = np.zeros_like(raw_fuel)
for i in range(len(raw_fuel)):
    if i < N:
        ma_fuel[i] = np.mean(raw_fuel[:i+1])
    else:
        ma_fuel[i] = np.mean(raw_fuel[i-N+1:i+1])

# 3. Tính Standard Kalman (Q=1, R=9)
kf = StandardKalmanFilter1D(initial_state=raw_fuel[0], process_noise=1.0, measurement_noise=9.0)
kf_fuel = []
for z in raw_fuel:
    kf_fuel.append(kf.update(z))
kf_fuel = np.array(kf_fuel)

# 4. Vẽ biểu đồ
fig, ax = plt.subplots(figsize=(14, 7))

ax.plot(time_segment, raw_fuel, color='lightgray', marker='.', linestyle='--', label='Raw FuelLevel', alpha=0.9)
ax.plot(time_segment, ma_fuel, color='blue', linewidth=2.5, label=f'Moving Average (N={N})')
ax.plot(time_segment, kf_fuel, color='red', linewidth=2.5, label='Standard Kalman (Q=1, R=9)')

# Annotate "Vùng dao động ngắn hạn" (index 30 to 80)
t_noisy_start = time_segment.iloc[30]
t_noisy_end = time_segment.iloc[80]
ax.axvspan(t_noisy_start, t_noisy_end, color='green', alpha=0.15)
ax.text(t_noisy_start, max(raw_fuel) - 5, 'Dao động ngắn hạn\n(Cả hai làm mượt)', color='green', fontsize=11, fontweight='bold')

# Annotate "Vùng thay đổi mức nhanh" (index 100 to 115)
t_drop_start = time_segment.iloc[100]
t_drop_end = time_segment.iloc[115]
ax.axvspan(t_drop_start, t_drop_end, color='orange', alpha=0.15)
ax.text(t_drop_start, 180, 'Thay đổi mức nhanh\n(MA bị trễ, Kalman bám nhanh hơn)', color='orange', fontsize=11, fontweight='bold')

# Annotate "Spike ngắn" (index 215 to 225)
t_spike_start = time_segment.iloc[215]
t_spike_end = time_segment.iloc[225]
ax.axvspan(t_spike_start, t_spike_end, color='purple', alpha=0.15)
ax.text(t_spike_start, min(raw_fuel) + 10, 'Spike bất thường\n(Kalman ít bị kéo lệch hơn MA)', color='purple', fontsize=11, fontweight='bold')

ax.set_title("So sánh trực quan Moving Average và Standard Kalman (Dữ liệu thực tế Car 1)", fontsize=14, fontweight='bold')
ax.set_xlabel("Thời gian (25/11/2025 - 26/11/2025)", fontsize=12)
ax.set_ylabel("Mức nhiên liệu (Lít)", fontsize=12)
ax.legend(loc='lower left')
ax.grid(True, linestyle=':', alpha=0.6)
ax.tick_params(axis='x', rotation=45)

plt.tight_layout()
output_path = "slide7_comparison.png"
plt.savefig(output_path, dpi=300)
