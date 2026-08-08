import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import pandas as pd

# 1. Đọc dữ liệu thật
df = pd.read_csv('data/processed/CarFuelHistory_Processed_Car1.csv')
raw_data_segment = df['FuelLevel'].iloc[1500:1800].values
time_segment = pd.to_datetime(df['FuelTime'].iloc[1500:1800])
raw_fuel = raw_data_segment

# 2. Tính Moving Average (N = 10)
N = 10
ma_fuel = np.zeros_like(raw_fuel)
for i in range(len(raw_fuel)):
    if i < N:
        ma_fuel[i] = np.mean(raw_fuel[:i+1])
    else:
        ma_fuel[i] = np.mean(raw_fuel[i-N+1:i+1])

# 3. Vẽ biểu đồ
fig, ax = plt.subplots(figsize=(12, 6))

ax.plot(time_segment, raw_fuel, color='lightgray', marker='.', linestyle='--', label='Raw FuelLevel', alpha=0.8)
ax.plot(time_segment, ma_fuel, color='blue', linewidth=2.5, label=f'Moving Average (N={N})')

# Annotate "Vùng nhiễu ngắn hạn"
# Region where data is fluctuating before the jump (index 30 to 80)
t_noisy_start = time_segment.iloc[30]
t_noisy_end = time_segment.iloc[80]
ax.axvspan(t_noisy_start, t_noisy_end, color='green', alpha=0.15)
ax.text(t_noisy_start, max(raw_fuel) - 5, 'Vùng nhiễu ngắn hạn\n(MA làm mượt tốt)', color='green', fontsize=11, fontweight='bold')

# Annotate "Vùng thay đổi nhanh"
# The jump happens at index 101 (from 155L to 289L)
t_drop_start = time_segment.iloc[100]
t_drop_end = time_segment.iloc[115]
ax.axvspan(t_drop_start, t_drop_end, color='red', alpha=0.15)
ax.text(t_drop_start, 180, 'Vùng thay đổi mức\n(MA bị trễ nặng)', color='red', fontsize=11, fontweight='bold')

ax.set_title("Hiệu ứng của Moving Average Filter (Dữ liệu thực tế Car 1)", fontsize=14, fontweight='bold')
ax.set_xlabel("Thời gian (25/11/2025 - 26/11/2025)", fontsize=12)
ax.set_ylabel("Mức nhiên liệu (Lít)", fontsize=12)
ax.legend(loc='lower left')
ax.grid(True, linestyle=':', alpha=0.6)
ax.tick_params(axis='x', rotation=45)

plt.tight_layout()
output_path = "slide5_ma_real.png"
plt.savefig(output_path, dpi=300)
print(f"Đã lưu biểu đồ Moving Average thành công tại {output_path}")
