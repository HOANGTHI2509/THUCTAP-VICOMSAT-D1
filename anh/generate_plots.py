import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import glob
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8')

# Thư mục lưu biểu đồ
out_dir = r"d:\THUCTAP_VICOMSAT\plots"
os.makedirs(out_dir, exist_ok=True)

# Lấy danh sách các file CSV đã tạo
csv_files = glob.glob("CarFuelHistory_Car*_Flags.csv")

all_data = []

# Hàm tiện ích để lưu ảnh
def save_plot(fig, filename):
    fig.savefig(os.path.join(out_dir, filename), bbox_inches="tight")
    plt.close(fig)

for f in csv_files:
    print(f"Đọc {f}...")
    df = pd.read_csv(f)
    df["FuelTime"] = pd.to_datetime(df["FuelTime"])
    all_data.append(df)

if not all_data:
    print("Không tìm thấy file CSV nào!")
    exit()

data = pd.concat(all_data, ignore_index=True)

# Boxplot chung cho 5 xe: Phân bố FuelLevel
print("Vẽ: Boxplot chung cho 5 xe (FuelLevel)")
vehicle_ids = []
fuel_values = []
for vehicle_id, group in data.groupby("VehicleID"):
    vehicle_ids.append(vehicle_id)
    fuel_values.append(group["FuelLevel"].dropna())

fig, ax = plt.subplots(figsize=(10, 6))
ax.boxplot(fuel_values, tick_labels=vehicle_ids)
ax.set_title("Phân bố FuelLevel theo từng xe")
ax.set_xlabel("VehicleID")
ax.set_ylabel("FuelLevel (lít)")
save_plot(fig, "boxplot_all_cars_fuellevel.png")

# Boxplot chung cho 5 xe: DeltaFuel theo trạng thái
print("Vẽ: Boxplot DeltaFuel theo trạng thái")
data["MovementState"] = np.where(data["Speed"] == 0, "Stopped", "Moving")
for vehicle_id, group in data.groupby("VehicleID"):
    states = ["Stopped", "Moving"]
    values = [group.loc[group["MovementState"] == state, "DeltaFuel"].dropna() for state in states]
    
    if len(values[0]) > 0 and len(values[1]) > 0:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.boxplot(values, tick_labels=states, showfliers=False)
        ax.set_title(f"Phân bố DeltaFuel theo trạng thái xe - {vehicle_id}")
        ax.set_xlabel("Trạng thái")
        ax.set_ylabel("DeltaFuel (lít)")
        save_plot(fig, f"{vehicle_id}_boxplot_deltafuel_state.png")

# Vẽ từng xe
for vehicle_id, group in data.groupby("VehicleID"):
    print(f"--- Đang vẽ biểu đồ cho {vehicle_id} ---")
    
    # 1. Histogram TimeGapMinutes
    gaps = group["TimeGapMinutes"].dropna()
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(gaps[gaps <= 60], bins=30)
    ax.set_title(f"Phân bố khoảng thời gian gửi dữ liệu - {vehicle_id}")
    ax.set_xlabel("TimeGapMinutes")
    ax.set_ylabel("Số lần xuất hiện")
    save_plot(fig, f"{vehicle_id}_hist_timegap.png")

    # 2. TimeGapMinutes theo thời gian
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(group["FuelTime"], group["TimeGapMinutes"])
    ax.set_title(f"Khoảng thời gian giữa các bản ghi - {vehicle_id}")
    ax.set_xlabel("Thời gian")
    ax.set_ylabel("TimeGapMinutes")
    save_plot(fig, f"{vehicle_id}_timegap_over_time.png")

    # 3. FuelLevel toàn thời gian
    fig, ax = plt.subplots(figsize=(15, 5))
    ax.plot(group["FuelTime"], group["FuelLevel"])
    ax.set_title(f"Mức nhiên liệu theo thời gian - {vehicle_id}")
    ax.set_xlabel("Thời gian")
    ax.set_ylabel("FuelLevel (lít)")
    save_plot(fig, f"{vehicle_id}_fuellevel_over_time.png")

    # 4. FuelLevel trong 1 ngày ngẫu nhiên
    group["Date"] = group["FuelTime"].dt.date
    dates = group["Date"].dropna().unique()
    if len(dates) > 0:
        selected_date = dates[0]
        sample = group[group["Date"] == selected_date]
        fig, ax = plt.subplots(figsize=(14, 5))
        ax.plot(sample["FuelTime"], sample["FuelLevel"], marker="o")
        ax.set_title(f"FuelLevel của {vehicle_id} ngày {selected_date}")
        ax.set_xlabel("Thời gian")
        ax.set_ylabel("FuelLevel (lít)")
        save_plot(fig, f"{vehicle_id}_fuellevel_1day.png")

    # 5. Histogram FuelLevel
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(group["FuelLevel"].dropna(), bins=50)
    ax.set_title(f"Phân bố FuelLevel - {vehicle_id}")
    ax.set_xlabel("FuelLevel (lít)")
    ax.set_ylabel("Số bản ghi")
    save_plot(fig, f"{vehicle_id}_hist_fuellevel.png")

    # 6. DeltaFuel theo thời gian
    fig, ax = plt.subplots(figsize=(15, 5))
    ax.plot(group["FuelTime"], group["DeltaFuel"])
    ax.axhline(0, color='r', linestyle='--')
    ax.set_title(f"Biến thiên nhiên liệu theo thời gian - {vehicle_id}")
    ax.set_xlabel("Thời gian")
    ax.set_ylabel("DeltaFuel (lít)")
    save_plot(fig, f"{vehicle_id}_deltafuel_over_time.png")

    # 7. Histogram DeltaFuel
    delta = group["DeltaFuel"].dropna()
    if not delta.empty:
        lower = delta.quantile(0.01)
        upper = delta.quantile(0.99)
        delta_plot = delta[(delta >= lower) & (delta <= upper)]
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.hist(delta_plot, bins=50)
        ax.set_title(f"Phân bố DeltaFuel - {vehicle_id}")
        ax.set_xlabel("DeltaFuel (lít)")
        ax.set_ylabel("Số lần xuất hiện")
        save_plot(fig, f"{vehicle_id}_hist_deltafuel.png")

    # 8. GPS Scatter
    valid = group.dropna(subset=["Lat", "Lng"])
    # Lọc bỏ các tọa độ nhiễu GPS (ví dụ: 0,0 hoặc bay xa ngoài Việt Nam/Đông Dương)
    valid = valid[
        (valid["Lat"] > 5) & (valid["Lat"] < 30) & 
        (valid["Lng"] > 100) & (valid["Lng"] < 115)
    ]
    
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(valid["Lng"], valid["Lat"], s=5)
    ax.set_title(f"Phân bố tọa độ - {vehicle_id}")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    save_plot(fig, f"{vehicle_id}_gps_scatter.png")

    # 9. DistanceMeters theo thời gian
    fig, ax = plt.subplots(figsize=(15, 5))
    ax.plot(group["FuelTime"], group["DistanceMeters"])
    ax.set_title(f"Khoảng cách GPS giữa hai mẫu - {vehicle_id}")
    ax.set_xlabel("Thời gian")
    ax.set_ylabel("DistanceMeters")
    save_plot(fig, f"{vehicle_id}_distance_over_time.png")

    # 10. Speed theo thời gian
    fig, ax = plt.subplots(figsize=(15, 5))
    ax.plot(group["FuelTime"], group["Speed"])
    ax.set_title(f"Tốc độ theo thời gian - {vehicle_id}")
    ax.set_xlabel("Thời gian")
    ax.set_ylabel("Speed")
    save_plot(fig, f"{vehicle_id}_speed_over_time.png")

    # 11. Histogram Speed
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(group["Speed"].dropna(), bins=40)
    ax.set_title(f"Phân bố Speed - {vehicle_id}")
    ax.set_xlabel("Speed")
    ax.set_ylabel("Số bản ghi")
    save_plot(fig, f"{vehicle_id}_hist_speed.png")

    # 12. FuelLevel và Speed kết hợp
    fig, ax1 = plt.subplots(figsize=(15, 6))
    ax1.plot(group["FuelTime"], group["FuelLevel"], label="FuelLevel", color='blue')
    ax1.set_xlabel("Thời gian")
    ax1.set_ylabel("FuelLevel (lít)", color='blue')
    ax2 = ax1.twinx()
    ax2.plot(group["FuelTime"], group["Speed"], alpha=0.5, label="Speed", color='orange')
    ax2.set_ylabel("Speed", color='orange')
    plt.title(f"FuelLevel và Speed theo thời gian - {vehicle_id}")
    save_plot(fig, f"{vehicle_id}_fuel_speed_combined.png")

    # 13. Điểm lỗi FuelLevel = 0
    flagged = group[group["FlagFuelZero"] == True]
    if len(flagged) > 0:
        fig, ax = plt.subplots(figsize=(15, 5))
        ax.plot(group["FuelTime"], group["FuelLevel"], label="FuelLevel")
        ax.scatter(flagged["FuelTime"], flagged["FuelLevel"], marker="x", color="red", s=70, label="FuelLevel = 0")
        ax.set_title(f"Các điểm FuelLevel = 0 - {vehicle_id}")
        ax.set_xlabel("Thời gian")
        ax.set_ylabel("FuelLevel (lít)")
        ax.legend()
        save_plot(fig, f"{vehicle_id}_fuel_zero_flags.png")

print("Hoàn tất tạo tất cả biểu đồ.")
