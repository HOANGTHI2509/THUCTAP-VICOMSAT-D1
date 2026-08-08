import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import warnings
import matplotlib.patches as mpatches
warnings.filterwarnings('ignore')

# Load data
df = pd.read_csv('data/processed/CarFuelHistory_Processed_Car1.csv')
df['FuelTime'] = pd.to_datetime(df['FuelTime'])
df = df.sort_values('FuelTime').reset_index(drop=True)

# Find a good slice where the car transitions between STOPPED and MOVING multiple times
# and actually reaches decent speeds
df['MovementState_Upper'] = df['MovementState'].str.upper()
df['StateChange'] = (df['MovementState_Upper'] != df['MovementState_Upper'].shift()).astype(int)

# Use a rolling window to find a region with a few transitions and high max speed
df['StateChanges_Rolling'] = df['StateChange'].rolling(300).sum()
df['MaxSpeed_Rolling'] = df['Speed'].rolling(300).max()

# We want a section where it stops and moves a bit (e.g. 2-5 changes) AND max speed is > 20km/h
candidate_idx = df[(df['StateChanges_Rolling'] >= 2) & 
                   (df['StateChanges_Rolling'] <= 6) & 
                   (df['MaxSpeed_Rolling'] > 20)].index

if len(candidate_idx) > 0:
    # Pick a random candidate from the middle to avoid edge cases
    end_idx = candidate_idx[len(candidate_idx) // 2]
else:
    end_idx = 1000

start_idx = max(0, end_idx - 300)
slice_df = df.iloc[start_idx:end_idx].copy().reset_index(drop=True)
date_str = slice_df['FuelTime'].dt.date.iloc[0]

# Compute RollingStd with window 5 to match slide definition exactly
slice_df['RollingStd_w5'] = slice_df['FuelLevel'].rolling(5).std().fillna(0)

# Create 3 subplots sharing X axis
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
x = np.arange(len(slice_df))

# Plot 1: FuelLevel
ax1.plot(x, slice_df['FuelLevel'], color='#1f77b4', linewidth=2)
ax1.set_ylabel('FuelLevel (Liters)', fontsize=11)
ax1.set_title(f'Đặc trưng chuyển động và Thống kê (Thực tế - Xe 1, Ngày {date_str})', fontsize=14, fontweight='bold')
ax1.grid(True, linestyle='--', alpha=0.6)

# Plot 2: Speed
ax2.plot(x, slice_df['Speed'], color='#2ca02c', linewidth=2)
ax2.axhline(5, color='red', linestyle='--', label='Ngưỡng 5 km/h')
ax2.set_ylabel('Speed (km/h)', fontsize=11)
ax2.grid(True, linestyle='--', alpha=0.6)
ax2.legend(loc='upper right', fontsize=10)

# Plot 3: RollingStd (Window 5)
ax3.plot(x, slice_df['RollingStd_w5'], color='#d62728', linewidth=2)
ax3.set_ylabel('RollingStd (w=5)', fontsize=11)
ax3.set_xlabel('Thời gian', fontsize=11)
ax3.grid(True, linestyle='--', alpha=0.6)

# Highlight Movement States on all axes
# We use yellow for MOVING, light grey for STOPPED
for i in range(len(slice_df) - 1):
    state = slice_df['MovementState_Upper'].iloc[i]
    color = 'yellow' if state == 'MOVING' else 'lightgrey'
    ax1.axvspan(i, i+1, color=color, alpha=0.3, lw=0)
    ax2.axvspan(i, i+1, color=color, alpha=0.3, lw=0)
    ax3.axvspan(i, i+1, color=color, alpha=0.3, lw=0)

# Custom legend for Movement State on the top plot
moving_patch = mpatches.Patch(color='yellow', alpha=0.3, label='MOVING (> 5km/h)')
stopped_patch = mpatches.Patch(color='lightgrey', alpha=0.4, label='STOPPED (≤ 5km/h)')
ax1.legend(handles=[moving_patch, stopped_patch], loc='upper right', fontsize=10)

# Set xticks to show time properly
ticks = np.linspace(0, len(x)-1, 6, dtype=int)
tick_labels = slice_df['FuelTime'].iloc[ticks].dt.strftime('%H:%M:%S')
ax3.set_xticks(ticks)
ax3.set_xticklabels(tick_labels)

plt.tight_layout()
plt.savefig('anh/Slide3_charts_real.png', dpi=300)
print(f"Image saved to slide3_charts_real.png. Date: {date_str}, Vehicle: Car 1")
