import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import sys
sys.stdout.reconfigure(encoding='utf-8')
import pandas as pd
import matplotlib.pyplot as plt
import os
import glob
import matplotlib.dates as mdates

out_dir = r"d:\THUCTAP_VICOMSAT\slide_plots"
os.makedirs(out_dir, exist_ok=True)

print("Đọc dữ liệu...")
csv_files = glob.glob(r"d:\THUCTAP_VICOMSAT\CarFuelHistory_*_Flags.csv")
all_data = []

for f in csv_files:
    df = pd.read_csv(f)
    if "VehicleID" not in df.columns: continue
    car = df["VehicleID"].iloc[0]
    if car not in ["Car 1", "Car 2", "Car 3", "Car 4", "Car 5"]: continue
    df["FuelTime"] = pd.to_datetime(df["FuelTime"])
    df = df.sort_values("FuelTime")
    all_data.append(df)

data = pd.concat(all_data, ignore_index=True)

# Lấy CAR 1 làm ví dụ cho Biểu đồ 1 và 2 (có thể thay đổi tùy ý)
car1 = data[data["VehicleID"] == "Car 1"].copy()

# ==============================================================================
# Biểu đồ 1: FuelLevel và Speed theo cùng thời gian
# ==============================================================================
print("Vẽ Biểu đồ 1: FuelLevel và Speed...")
car1['Date'] = car1['FuelTime'].dt.date
most_records_date = car1['Date'].value_counts().idxmax()
# Cắt dữ liệu trong 1 ngày, có thể chọn khoảng 8h - 18h cho rõ ràng
car1_zoom = car1[car1['Date'] == most_records_date].copy()
car1_zoom = car1_zoom[(car1_zoom['FuelTime'].dt.hour >= 8) & (car1_zoom['FuelTime'].dt.hour <= 18)]

fig, ax1 = plt.subplots(figsize=(12, 6))

color_fuel = '#1f77b4'
ax1.set_xlabel('Thời gian', fontsize=12)
ax1.set_ylabel('FuelLevel (lít)', color=color_fuel, fontsize=12, fontweight='bold')
ax1.plot(car1_zoom['FuelTime'], car1_zoom['FuelLevel'], color=color_fuel, linewidth=2, label='FuelLevel')
ax1.tick_params(axis='y', labelcolor=color_fuel)

ax2 = ax1.twinx()
color_speed = '#ff7f0e'
ax2.set_ylabel('Speed (km/h)', color=color_speed, fontsize=12, fontweight='bold')
ax2.plot(car1_zoom['FuelTime'], car1_zoom['Speed'], color=color_speed, alpha=0.8, linewidth=1.5, label='Speed')
ax2.fill_between(car1_zoom['FuelTime'], car1_zoom['Speed'], color=color_speed, alpha=0.15)
ax2.tick_params(axis='y', labelcolor=color_speed)

# Format trục X
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
fig.autofmt_xdate()

plt.title("Biến thiên FuelLevel theo trạng thái chuyển động – CAR 1", fontsize=15, fontweight='bold')
lines_1, labels_1 = ax1.get_legend_handles_labels()
lines_2, labels_2 = ax2.get_legend_handles_labels()
ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc="upper right")
fig.tight_layout()

fig.savefig(os.path.join(out_dir, "Slide5_Fuel_Speed_Car1.png"), dpi=300)
plt.close(fig)

# ==============================================================================
# Biểu đồ 2: Khoảng cách GPS khi Speed = 0
# ==============================================================================
print("Vẽ Biểu đồ 2: DistanceMeters khi Speed = 0...")
# Dùng cùng khoảng thời gian (1 ngày) với Biểu đồ 1 để dễ đối chiếu
car1_stopped = car1_zoom[car1_zoom["Speed"] == 0].copy()

fig, ax = plt.subplots(figsize=(10, 5))
# Dùng biểu đồ thanh đứng (bar) kết hợp scatter cho dễ nhìn từng thời điểm
ax.bar(car1_stopped["FuelTime"], car1_stopped["DistanceMeters"], width=0.002, color='#d62728', alpha=0.7)
ax.scatter(car1_stopped["FuelTime"], car1_stopped["DistanceMeters"], s=15, color='black', zorder=3)

# Đặt giới hạn trục Y (cắt bỏ các nhiễu quá vô lý để thấy được độ trôi GPS thông thường)
if len(car1_stopped) > 0:
    max_y = car1_stopped["DistanceMeters"].quantile(0.95) * 1.5
    if pd.isna(max_y) or max_y < 100: max_y = 100
    ax.set_ylim(0, max_y)

ax.set_title("Dịch chuyển GPS tại các thời điểm Speed = 0\n(Cùng khoảng thời gian với Biểu đồ 1)", fontsize=14, fontweight='bold')
ax.set_xlabel("Thời gian (Giờ:Phút)", fontsize=12)
ax.set_ylabel("Khoảng cách dịch chuyển (Meters)", fontsize=12)

# Format Date
ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
fig.autofmt_xdate()
fig.tight_layout()

fig.savefig(os.path.join(out_dir, "Slide5_Distance_Speed0_Car1.png"), dpi=300)
plt.close(fig)

# ==============================================================================
# Biểu đồ 3: Scatter GPS CAR 2 (Hiển thị các điểm rời rạc)
# ==============================================================================
print("Vẽ Biểu đồ 3: Tọa độ GPS CAR 2...")
car2 = data[data["VehicleID"] == "Car 2"].copy()
car2 = car2.dropna(subset=["Lat", "Lng"])
# Chỉ lọc bỏ (0,0), giữ lại các điểm "bay" để làm ví dụ về nhiễu dữ liệu cần đối chiếu
car2 = car2[(car2["Lat"] != 0) & (car2["Lng"] != 0)]

fig, ax = plt.subplots(figsize=(6, 6))
ax.scatter(car2["Lng"], car2["Lat"], s=8, alpha=0.5, color='#2ca02c')

ax.set_title("Phân bố tọa độ GPS - CAR 2", fontsize=13, fontweight='bold')
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")
fig.tight_layout()

fig.savefig(os.path.join(out_dir, "Slide5_GPS_Scatter_Car2.png"), dpi=300)
plt.close(fig)

print("Hoàn tất! Các biểu đồ cho Slide 5 đã được lưu tại:", out_dir)
