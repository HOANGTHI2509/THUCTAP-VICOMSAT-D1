import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import numpy as np
import matplotlib.pyplot as plt

# Generate synthetic data
np.random.seed(42)
time = np.arange(0, 100)
fuel_level = np.zeros(100)
fuel_level[0] = 100

for i in range(1, 100):
    if 30 <= i <= 50:
        # Volatile period: high noise, random fluctuations
        fuel_level[i] = fuel_level[i-1] - 0.2 + np.random.normal(0, 3)
    else:
        # Stable period: slow continuous drop with minimal noise
        fuel_level[i] = fuel_level[i-1] - 0.2 + np.random.normal(0, 0.2)

# Calculate DeltaFuel
delta_fuel = np.diff(fuel_level, prepend=fuel_level[0])
delta_fuel[0] = 0 # No delta for the first point

# Create plot
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True, gridspec_kw={'height_ratios': [2, 1]})

# Top plot: FuelLevel
ax1.plot(time, fuel_level, color='#1f77b4', linewidth=2, label='FuelLevel')
ax1.set_ylabel('FuelLevel (Liters)', fontsize=12)
ax1.set_title('Đồ thị Mức nhiên liệu và Biến thiên (DeltaFuel)', fontsize=14, fontweight='bold')
ax1.grid(True, linestyle='--', alpha=0.6)

# Highlight stable segment
ax1.axvspan(10, 25, color='green', alpha=0.2, label='Đoạn ổn định')
# Highlight volatile segment
ax1.axvspan(30, 50, color='red', alpha=0.2, label='Đoạn biến động mạnh')
ax1.legend(loc='upper right', fontsize=10)

# Bottom plot: DeltaFuel
ax2.bar(time, delta_fuel, color='#ff7f0e', alpha=0.8, label='DeltaFuel')
ax2.set_xlabel('Thời gian (t)', fontsize=12)
ax2.set_ylabel('DeltaFuel', fontsize=12)
ax2.grid(True, linestyle='--', alpha=0.6)

# Highlight regions on bottom plot as well
ax2.axvspan(10, 25, color='green', alpha=0.2)
ax2.axvspan(30, 50, color='red', alpha=0.2)
ax2.legend(loc='upper right', fontsize=10)

plt.tight_layout()
plt.savefig('anh/Slide2_chart.png', dpi=300)
print("Image saved to slide2_chart.png")
