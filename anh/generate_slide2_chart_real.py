import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Load data
df = pd.read_csv('data/processed/CarFuelHistory_Processed_Car1.csv')
df['FuelTime'] = pd.to_datetime(df['FuelTime'])
df = df.sort_values('FuelTime').reset_index(drop=True)

# Recalculate DeltaFuel
df['DeltaFuel'] = df['FuelLevel'].diff().fillna(0)

# Exclude extreme outliers for rolling std calculation (e.g. refuels > 10 liters)
valid_mask = df['DeltaFuel'].abs() < 10
df_filtered = df[valid_mask].copy()

# Use 15-point rolling std to find very localized high volatility
df_filtered['RollingStd'] = df_filtered['DeltaFuel'].rolling(15).std()

# Find the maximum rolling std to center our "zoomed in" view
# Ensure we don't pick the very end or beginning
search_area = df_filtered.iloc[100:-100]
highest_std_idx = search_area['RollingStd'].idxmax()

center_idx = highest_std_idx

# Zoom in tightly: 50 points before and 50 points after (100 points total)
start_idx = max(0, center_idx - 50)
end_idx = min(len(df), center_idx + 50)

slice_df = df.iloc[start_idx:end_idx].copy().reset_index(drop=True)
date_str = slice_df['FuelTime'].dt.date.iloc[0]

# Recalculate rolling std on this small slice to find the exact highlight windows
rolling_std_slice = slice_df['DeltaFuel'].rolling(10).std().fillna(0)

# Volatile region (highlight the most volatile part)
volatile_end = rolling_std_slice.idxmax()
volatile_start = max(0, volatile_end - 15)

# Stable region (highlight the most stable part in this window)
stable_candidates = rolling_std_slice.copy()
# Avoid overlapping with volatile region
overlap_start = max(0, volatile_start - 10)
overlap_end = min(len(slice_df), volatile_end + 10)
stable_candidates.iloc[overlap_start:overlap_end] = 999 

stable_end = stable_candidates.idxmin()
stable_start = max(0, stable_end - 10)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True, gridspec_kw={'height_ratios': [2, 1]})

x = np.arange(len(slice_df))

# Top plot
ax1.plot(x, slice_df['FuelLevel'], color='#1f77b4', linewidth=2, marker='o', markersize=3, label='FuelLevel')
ax1.set_ylabel('FuelLevel (Liters)', fontsize=12)
ax1.set_title(f'Đồ thị Phóng to: Biến thiên Mức nhiên liệu (Xe 1, Ngày {date_str})', fontsize=14, fontweight='bold')
ax1.grid(True, linestyle='--', alpha=0.6)

ax1.axvspan(stable_start, stable_end, color='green', alpha=0.2, label='Đoạn tương đối ổn định')
ax1.axvspan(volatile_start, volatile_end, color='red', alpha=0.2, label='Đoạn biến động mạnh')
ax1.legend(loc='upper right', fontsize=10)

# Bottom plot
# Using stem or bar for zoomed in data makes it look better
ax2.bar(x, slice_df['DeltaFuel'], color='#ff7f0e', alpha=0.8, label='DeltaFuel', width=0.6)
ax2.set_xlabel('Thời gian', fontsize=12)
ax2.set_ylabel('DeltaFuel', fontsize=12)
ax2.grid(True, linestyle='--', alpha=0.6)

ax2.axvspan(stable_start, stable_end, color='green', alpha=0.2)
ax2.axvspan(volatile_start, volatile_end, color='red', alpha=0.2)
ax2.legend(loc='upper right', fontsize=10)

# Set xticks to show time
# Pick about 6-7 ticks across the zoomed in window
ticks = np.linspace(0, len(x)-1, 7, dtype=int)
tick_labels = slice_df['FuelTime'].iloc[ticks].dt.strftime('%H:%M:%S')
ax2.set_xticks(ticks)
ax2.set_xticklabels(tick_labels)

plt.tight_layout()
plt.savefig('anh/Slide2_chart_real_zoomed.png', dpi=300)
print(f"Image saved to slide2_chart_real_zoomed.png. Date: {date_str}, Vehicle: Car 1")
