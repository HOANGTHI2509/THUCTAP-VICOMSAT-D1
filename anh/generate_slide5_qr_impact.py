import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import matplotlib.pyplot as plt
import numpy as np

from src.core.filters.kalman_traditional import StandardKalmanFilter1D

# 1. Tạo dữ liệu giả lập (Synthetic data) với nhiễu và một bước nhảy (step change)
np.random.seed(42)
n_samples = 100
true_fuel = np.zeros(n_samples)
true_fuel[:40] = 200.0
true_fuel[40:60] = 150.0  # Rút trộm 50L
true_fuel[60:] = 150.0

noise = np.random.normal(0, 3, n_samples)
raw_fuel = true_fuel + noise
# Thêm vài điểm nhiễu gai (spike) lớn
raw_fuel[20] += 25
raw_fuel[80] -= 20

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
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# ---- Biểu đồ bên trái: Tác động của Q ----
ax1.plot(raw_fuel, color='lightgray', marker='.', linestyle='--', label='Raw FuelLevel', alpha=0.7)
ax1.plot(q_small, color='blue', linewidth=2, label='Kalman (Q nhỏ = 0.01, R=9)')
ax1.plot(q_large, color='red', linewidth=2, label='Kalman (Q lớn = 5, R=9)')
ax1.set_title("Tác động của Q (Giữ R = 9 cố định)", fontsize=14, fontweight='bold')
ax1.set_xlabel("Time (samples)")
ax1.set_ylabel("Fuel Level (Liters)")
ax1.legend(loc='upper right')
ax1.grid(True, linestyle=':', alpha=0.6)

# ---- Biểu đồ bên phải: Tác động của R ----
ax2.plot(raw_fuel, color='lightgray', marker='.', linestyle='--', label='Raw FuelLevel', alpha=0.7)
ax2.plot(r_small, color='green', linewidth=2, label='Kalman (R nhỏ = 1, Q=1)')
ax2.plot(r_large, color='purple', linewidth=2, label='Kalman (R lớn = 100, Q=1)')
ax2.set_title("Tác động của R (Giữ Q = 1 cố định)", fontsize=14, fontweight='bold')
ax2.set_xlabel("Time (samples)")
ax2.set_ylabel("Fuel Level (Liters)")
ax2.legend(loc='upper right')
ax2.grid(True, linestyle=':', alpha=0.6)

plt.tight_layout()
output_path = "slide5_qr_impact.png"
plt.savefig(output_path, dpi=300)
print(f"Đã lưu biểu đồ thành công tại {output_path}")
