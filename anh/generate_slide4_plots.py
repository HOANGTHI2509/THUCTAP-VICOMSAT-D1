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

out_dir = r"d:\THUCTAP_VICOMSAT\slide_plots"
os.makedirs(out_dir, exist_ok=True)

csv_files = sorted(glob.glob(r"d:\THUCTAP_VICOMSAT\CarFuelHistory_*_Flags.csv"))
all_data = []
stats = []

for f in csv_files:
    df = pd.read_csv(f)
    if "VehicleID" not in df.columns: continue
    car = df["VehicleID"].iloc[0]
    if car not in ["Car 1", "Car 2", "Car 3", "Car 4", "Car 5"]: continue
    
    df["FuelTime"] = pd.to_datetime(df["FuelTime"])
    df = df.sort_values("FuelTime")
    df["AbsDeltaFuel"] = df["FuelLevel"].diff().abs()
    
    med_delta = df["AbsDeltaFuel"].median()
    p95_delta = df["AbsDeltaFuel"].quantile(0.95)
    max_delta = df["AbsDeltaFuel"].max()
    fuel_0 = df["FlagFuelZero"].sum() if "FlagFuelZero" in df.columns else (df["FuelLevel"]==0).sum()
    
    stats.append({
        "Xe": car,
        "Trung vị |ΔFuel|": f"{med_delta:.3f}",
        "P95 |ΔFuel|": f"{p95_delta:.3f}",
        "Biến động lớn nhất": f"{max_delta:.1f}",
        "Fuel = 0": f"{fuel_0}"
    })
    
    all_data.append(df)

data = pd.concat(all_data, ignore_index=True)

# Markdown Table
print("=== MARKDOWN TABLE START ===")
print("| Xe | Trung vị |ΔFuel| | P95 |ΔFuel| | Biến động lớn nhất | Fuel = 0 |")
print("|---|---|---|---|---|")
for row in stats:
    print(f"| {row['Xe']} | {row['Trung vị |ΔFuel|']} | {row['P95 |ΔFuel|']} | {row['Biến động lớn nhất']} | {row['Fuel = 0']} |")
print("=== MARKDOWN TABLE END ===")

# Biểu đồ cho CAR 3
car3 = data[data["VehicleID"] == "Car 3"].copy()
car3 = car3.sort_values("FuelTime")

# 1. Biểu đồ toàn thời gian CAR 3
print("Đang vẽ biểu đồ toàn thời gian CAR 3...")
fig, ax = plt.subplots(figsize=(14, 5))
ax.plot(car3["FuelTime"], car3["FuelLevel"], label="FuelLevel", color="#1f77b4", linewidth=1)

# Đánh dấu các điểm Fuel = 0 hoặc biến động lớn (>50 lít)
anomalies_0 = car3[car3["FuelLevel"] == 0]
anomalies_spike = car3[car3["AbsDeltaFuel"] > 50]

ax.scatter(anomalies_0["FuelTime"], anomalies_0["FuelLevel"], color="red", marker="x", s=80, label="FuelLevel = 0", zorder=5)
ax.scatter(anomalies_spike["FuelTime"], anomalies_spike["FuelLevel"], color="orange", marker="v", s=60, label="Biến động > 50L", zorder=4)

ax.set_title("Diễn biến FuelLevel thô theo thời gian – CAR 3", fontsize=14)
ax.set_ylabel("FuelLevel (lít)", fontsize=12)
ax.legend(loc="upper right")
fig.savefig(os.path.join(out_dir, "Slide4_FuelLevel_ToanThoiGian_Car3.png"), bbox_inches="tight", dpi=300)
plt.close(fig)

# 2. Biểu đồ phóng to một đoạn tín hiệu (dao động ngắn hạn)
print("Đang vẽ biểu đồ phóng to CAR 3...")
# Tìm ngày có nhiều bản ghi nhất để đảm bảo có tín hiệu liên tục
car3['Date'] = car3['FuelTime'].dt.date
date_counts = car3['Date'].value_counts()
# Lấy một ngày ở giữa chuỗi để tránh ngày đầu/cuối ít dữ liệu
top_dates = date_counts.head(10).index.tolist()
top_dates.sort()
chosen_date = top_dates[len(top_dates)//2] # Lấy ngày ở giữa

car3_zoom = car3[car3['Date'] == chosen_date].copy()
# Lấy khoảng thời gian từ 8h sáng đến 16h chiều
car3_zoom = car3_zoom[(car3_zoom['FuelTime'].dt.hour >= 8) & (car3_zoom['FuelTime'].dt.hour <= 16)]

fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(car3_zoom["FuelTime"], car3_zoom["FuelLevel"], marker='.', linestyle='-', color="#2ca02c", linewidth=1, markersize=4)

# Format trục X để hiển thị giờ:phút
ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
plt.xticks(rotation=45)

ax.set_title(f"Đoạn tín hiệu dao động ngắn hạn (8:00 - 16:00, ngày {chosen_date})", fontsize=14)
ax.set_ylabel("FuelLevel (lít)", fontsize=12)
ax.set_xlabel("Thời gian (Giờ:Phút)")
fig.savefig(os.path.join(out_dir, "Slide4_FuelLevel_PhongTo_Car3.png"), bbox_inches="tight", dpi=300)
plt.close(fig)

print(f"Hoàn tất! Các biểu đồ đã được lưu tại thư mục: {out_dir}")
