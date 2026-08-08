import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import pandas as pd
import matplotlib.pyplot as plt
import os
import glob
import matplotlib.dates as mdates

# Set up output directory
os.makedirs("eda_plots", exist_ok=True)

# Find processed files
csv_files = glob.glob("CarFuelHistory_Processed_*.csv")

for file in csv_files:
    vehicle_id = file.replace("CarFuelHistory_Processed_", "").replace(".csv", "")
    print(f"Generating EDA plots for {vehicle_id}...")
    
    df = pd.read_csv(file)
    df['FuelTime'] = pd.to_datetime(df['FuelTime'])
    
    # We will pick the largest segment to plot to have a continuous line
    if df.empty or 'SegmentID' not in df.columns:
        continue
        
    largest_segment = df['SegmentID'].value_counts().idxmax()
    df_seg = df[df['SegmentID'] == largest_segment].copy()
    
    # If the segment is too long, we just take the first 500 rows for a clearer view of the noise
    df_seg = df_seg.head(1000)
    
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 12), sharex=True)
    
    # --- Plot 1: Raw Fuel vs Rolling Median ---
    ax1.plot(df_seg['FuelTime'], df_seg['FuelLevel'], label='Raw FuelLevel', color='lightcoral', alpha=0.7, marker='.', markersize=3)
    if 'RollingMedian' in df_seg.columns:
        ax1.plot(df_seg['FuelTime'], df_seg['RollingMedian'], label='Rolling Median (Window=5)', color='darkred', linewidth=2)
    ax1.set_title(f'{vehicle_id} - Biến động Mức Nhiên Liệu (Segment {largest_segment})', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Nhiên Liệu (Lít)', fontsize=12)
    ax1.legend(loc='upper right')
    ax1.grid(True, linestyle='--', alpha=0.5)
    
    # --- Plot 2: Speed (Movement State) ---
    if 'Speed' in df_seg.columns:
        # Tô màu nền dựa trên trạng thái Moving/Stopped
        ax2.plot(df_seg['FuelTime'], df_seg['Speed'], label='Speed (km/h)', color='blue', linewidth=1.5)
        
        # Shade background where moving
        moving_mask = df_seg['MovementState'] == 'Moving'
        ax2.fill_between(df_seg['FuelTime'], 0, df_seg['Speed'].max() + 10, where=moving_mask, 
                         color='blue', alpha=0.1, label='Moving State')
        
        ax2.set_title('Tốc Độ Di Chuyển (Speed)', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Tốc độ (km/h)', fontsize=12)
        ax2.legend(loc='upper right')
        ax2.grid(True, linestyle='--', alpha=0.5)
        
    # --- Plot 3: Rolling Standard Deviation (Noise Variance) ---
    if 'RollingStd' in df_seg.columns:
        ax3.plot(df_seg['FuelTime'], df_seg['RollingStd'], label='Rolling Std (Độ Lệch Chuẩn)', color='green', linewidth=1.5)
        ax3.set_title('Độ Lệch Chuẩn Biên Độ Nhiễu (Rolling Std)', fontsize=14, fontweight='bold')
        ax3.set_ylabel('Std (Lít)', fontsize=12)
        ax3.legend(loc='upper right')
        ax3.grid(True, linestyle='--', alpha=0.5)

    # Formatting x-axis
    ax3.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M\n%d/%m'))
    plt.xticks(rotation=0)
    plt.tight_layout()
    
    # Save plot
    out_path = os.path.join("eda_plots", f"EDA_{vehicle_id}.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved {out_path}")

print("Hoàn tất tạo biểu đồ EDA!")
