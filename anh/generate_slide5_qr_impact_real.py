import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.core.filters.kalman_traditional import StandardKalmanFilter1D

# 1. Đọc dữ liệu thật
df = pd.read_csv('data/processed/CarFuelHistory_Processed_Car1.csv')
# Chọn một đoạn dữ liệu thật (Ví dụ từ index 1000 đến 1250 để thấy được mức độ biến động)
raw_data_segment = df['FuelLevel'].iloc[1500:1800].values
raw_fuel = raw_data_segment
time_segment = pd.to_datetime(df['FuelTime'].iloc[1500:1800])

# 2. Định nghĩa hàm chạy Kalman
def run_kalman(raw_data, Q, R):
    kf = StandardKalmanFilter1D(initial_state=raw_data[0], process_noise=Q, measurement_noise=R)
    filtered = []
    for z in raw_data:
        filtered.append(kf.update(z))
    return np.array(filtered)

# 3. Chạy 4 cấu hình
# Biểu đồ 1: Tác động của Q (R = 9 cố định)
q_small = run_kalman(raw_fuel, Q=0.01, R=9)
q_large = run_kalman(raw_fuel, Q=5.0, R=9)

# Biểu đồ 2: Tác động của R (Q = 1 cố định)
r_small = run_kalman(raw_fuel, Q=1.0, R=1.0)
r_large = run_kalman(raw_fuel, Q=1.0, R=100.0)

# 4. Vẽ biểu đồ
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

# ---- Biểu đồ bên trái: Tác động của Q ----
ax1.plot(time_segment, raw_fuel, color='lightgray', marker='.', linestyle='--', label='Raw FuelLevel', alpha=0.7)
ax1.plot(time_segment, q_small, color='blue', linewidth=2, label='Kalman (Q nhỏ = 0.01, R=9)')
ax1.plot(time_segment, q_large, color='red', linewidth=2, label='Kalman (Q lớn = 5, R=9)')
ax1.set_title("Tác động của Q (Giữ R = 9 cố định) - Dữ liệu thật Car 1", fontsize=14, fontweight='bold')
ax1.set_xlabel("Thời gian (25/11/2025 - 26/11/2025)", fontsize=12)
ax1.set_ylabel("Mức nhiên liệu (Lít)", fontsize=12)
ax1.legend(loc='upper right')
ax1.grid(True, linestyle=':', alpha=0.6)
ax1.tick_params(axis='x', rotation=45)

# ---- Biểu đồ bên phải: Tác động của R ----
ax2.plot(time_segment, raw_fuel, color='lightgray', marker='.', linestyle='--', label='Raw FuelLevel', alpha=0.7)
ax2.plot(time_segment, r_small, color='green', linewidth=2, label='Kalman (R nhỏ = 1, Q=1)')
ax2.plot(time_segment, r_large, color='purple', linewidth=2, label='Kalman (R lớn = 100, Q=1)')
ax2.set_title("Tác động của R (Giữ Q = 1 cố định) - Dữ liệu thật Car 1", fontsize=14, fontweight='bold')
ax2.set_xlabel("Thời gian (25/11/2025 - 26/11/2025)", fontsize=12)
ax2.set_ylabel("Mức nhiên liệu (Lít)", fontsize=12)
ax2.legend(loc='upper right')
ax2.grid(True, linestyle=':', alpha=0.6)
ax2.tick_params(axis='x', rotation=45)

plt.tight_layout()
output_path = "slide5_qr_impact_real.png"
plt.savefig(output_path, dpi=300)
print(f"Đã lưu biểu đồ dữ liệu thật thành công tại {output_path}")
