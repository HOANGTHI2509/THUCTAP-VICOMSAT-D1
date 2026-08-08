import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import sys
sys.stdout.reconfigure(encoding='utf-8')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
import seaborn as sns

out_dir = r"d:\THUCTAP_VICOMSAT\slide_plots"
os.makedirs(out_dir, exist_ok=True)

print("Đọc dữ liệu...")
csv_files = sorted(glob.glob(r"d:\THUCTAP_VICOMSAT\CarFuelHistory_*_Flags.csv"))
all_data = []

for f in csv_files:
    df = pd.read_csv(f)
    if "VehicleID" not in df.columns: continue
    car = df["VehicleID"].iloc[0]
    if car not in ["Car 1", "Car 2", "Car 3", "Car 4", "Car 5"]: continue
    df["FuelTime"] = pd.to_datetime(df["FuelTime"])
    df = df.sort_values("FuelTime")
    
    if "AbsDeltaFuel" not in df.columns:
        df["AbsDeltaFuel"] = df["FuelLevel"].diff().abs()
        
    all_data.append(df)

data = pd.concat(all_data, ignore_index=True)
data["MovementState"] = np.where(data["Speed"] > 0, "Đang di chuyển", "Dừng (Speed = 0)")

# ==============================================================================
# Biểu đồ 1: Biến thiên FuelLevel theo trạng thái (Có bôi nền khác màu)
# ==============================================================================
print("Vẽ Biểu đồ 1: FuelLevel và Speed có highlight trạng thái...")
car3 = data[data["VehicleID"] == "Car 3"].copy()
car3['Date'] = car3['FuelTime'].dt.date
most_records_date = car3['Date'].value_counts().idxmax()

# Dùng CAR 3 và lọc 13h đến 16h để thấy rõ dao động nhiễu của nhiên liệu
car3_zoom = car3[car3['Date'] == most_records_date].copy()
car3_zoom = car3_zoom[(car3_zoom['FuelTime'].dt.hour >= 13) & (car3_zoom['FuelTime'].dt.hour <= 16)]
car3_zoom.reset_index(drop=True, inplace=True)

fig, ax1 = plt.subplots(figsize=(14, 6))

# Thuật toán vẽ nền màu cho các khoảng thời gian
states = car3_zoom["MovementState"].values
times = car3_zoom["FuelTime"].values

start_idx = 0
for i in range(1, len(states)):
    # Đổi trạng thái hoặc đến điểm cuối cùng
    if states[i] != states[i-1] or i == len(states) - 1:
        end_idx = i
        # Dùng màu xanh lá đậm hơn cho chạy, xám đậm hơn cho dừng
        color = '#a1d99b' if states[i-1] == "Đang di chuyển" else '#bdbdbd'
        ax1.axvspan(times[start_idx], times[end_idx], color=color, alpha=0.5, lw=0)
        start_idx = i

# Vẽ FuelLevel
color_fuel = '#1f77b4'
ax1.plot(car3_zoom['FuelTime'], car3_zoom['FuelLevel'], color=color_fuel, linewidth=2, label='FuelLevel', zorder=3)
ax1.set_xlabel('Thời gian (Giờ:Phút)', fontsize=12)
ax1.set_ylabel('FuelLevel (lít)', color=color_fuel, fontsize=12, fontweight='bold')
ax1.tick_params(axis='y', labelcolor=color_fuel)

# Vẽ Speed
ax2 = ax1.twinx()
color_speed = '#ff7f0e'
ax2.plot(car3_zoom['FuelTime'], car3_zoom['Speed'], color=color_speed, alpha=0.8, linewidth=1.5, label='Speed', zorder=4)
ax2.set_ylabel('Speed (km/h)', color=color_speed, fontsize=12, fontweight='bold')
ax2.tick_params(axis='y', labelcolor=color_speed)

# Format X axis
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
fig.autofmt_xdate()

# Custom Legend bao gồm cả màu nền
moving_patch = mpatches.Patch(color='#a1d99b', alpha=0.5, label='Trạng thái: Đang di chuyển')
stopped_patch = mpatches.Patch(color='#bdbdbd', alpha=0.5, label='Trạng thái: Dừng')

lines_1, labels_1 = ax1.get_legend_handles_labels()
lines_2, labels_2 = ax2.get_legend_handles_labels()
ax1.legend(lines_1 + lines_2 + [moving_patch, stopped_patch], 
           labels_1 + labels_2 + ['Trạng thái: Đang di chuyển', 'Trạng thái: Dừng'], 
           loc="upper right", framealpha=0.9)

plt.title("Biến thiên FuelLevel theo trạng thái chuyển động – CAR 3", fontsize=15, fontweight='bold')
fig.tight_layout()

fig.savefig(os.path.join(out_dir, "Slide5_Fuel_Speed_Colored_Car3.png"), dpi=300)
plt.close(fig)

# ==============================================================================
# Biểu đồ 2: So sánh mức biến động FuelLevel khi xe dừng và di chuyển
# ==============================================================================
print("Vẽ Biểu đồ 2: So sánh Median |DeltaFuel|...")
fig, (ax_box, ax_bar) = plt.subplots(1, 2, figsize=(16, 6))

# Hình A: Boxplot
clean_data = data[data["AbsDeltaFuel"] < 5].copy() # Cắt ngoại lai lớn để boxplot dễ nhìn
sns.boxplot(data=clean_data, x="VehicleID", y="AbsDeltaFuel", hue="MovementState", 
            palette={"Dừng (Speed = 0)": "#ccebc5", "Đang di chuyển": "#fbb4ae"}, 
            ax=ax_box, showfliers=False)
ax_box.set_title("Phân bố mức biến động (|DeltaFuel|) (Boxplot)", fontsize=13)
ax_box.set_ylabel("|DeltaFuel| (lít)", fontsize=12)
ax_box.set_xlabel("")
ax_box.legend(title="Trạng thái", loc='upper left')

# Hình B: Barplot (Median)
median_delta = data.groupby(["VehicleID", "MovementState"])["AbsDeltaFuel"].median().reset_index()
sns.barplot(data=median_delta, x="VehicleID", y="AbsDeltaFuel", hue="MovementState", 
            palette={"Dừng (Speed = 0)": "#ccebc5", "Đang di chuyển": "#fbb4ae"}, 
            ax=ax_bar)

# Gắn text giá trị lên cột bar
for p in ax_bar.patches:
    height = p.get_height()
    if not pd.isna(height) and height > 0:
        ax_bar.text(p.get_x() + p.get_width() / 2., height + 0.002, f'{height:.3f}', ha="center", va="bottom", fontsize=10)

ax_bar.set_title("Trung vị mức biến động (Median |DeltaFuel|)", fontsize=13)
ax_bar.set_ylabel("Trung vị (lít)", fontsize=12)
ax_bar.set_xlabel("")
ax_bar.legend_.remove() # Ẩn legend ở biểu đồ này cho gọn

plt.suptitle("So sánh mức biến động FuelLevel khi xe dừng và di chuyển", fontsize=15, fontweight='bold', y=1.02)
fig.tight_layout()

fig.savefig(os.path.join(out_dir, "Slide5_Comparison_DeltaFuel_State.png"), dpi=300)
plt.close(fig)

print("Hoàn tất! Biểu đồ mới đã được lưu.")
