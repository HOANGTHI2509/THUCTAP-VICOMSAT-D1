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
import matplotlib.ticker as mtick

out_dir = r"d:\THUCTAP_VICOMSAT\slide_plots"
os.makedirs(out_dir, exist_ok=True)

print("Đang đọc dữ liệu...")
csv_files = glob.glob(r"d:\THUCTAP_VICOMSAT\CarFuelHistory_*_Flags.csv")
all_data = []
for f in csv_files:
    df = pd.read_csv(f)
    if "VehicleID" not in df.columns:
        continue
    df["FuelTime"] = pd.to_datetime(df["FuelTime"])
    all_data.append(df)
data = pd.concat(all_data, ignore_index=True)
data = data[data["VehicleID"].isin(["Car 1", "Car 2", "Car 3", "Car 4", "Car 5"])]

# 1. Biểu đồ cột chồng (4 nhóm lỗi quan trọng nhất)
print("Đang vẽ biểu đồ tỷ lệ lỗi (Slide2_TyLeLoi.png)...")
record_counts = data["VehicleID"].value_counts().sort_index()
issues = data.groupby("VehicleID").agg({
    "FlagDuplicateTime": "sum",
    "FlagFuelZero": "sum",
    "FlagLongGap": "sum",
    "FlagSpeedGpsConflict": "sum"
})
issues_pct = issues.div(record_counts, axis=0) * 100

fig, ax = plt.subplots(figsize=(10, 6))
bottom = np.zeros(len(issues_pct))
colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99']
labels = ['Timestamp trùng', 'FuelLevel = 0', 'Khoảng mất dữ liệu > 30p', 'Mâu thuẫn Speed-GPS']

for i, col in enumerate(issues.columns):
    ax.bar(issues_pct.index, issues_pct[col], bottom=bottom, label=labels[i], color=colors[i])
    bottom += issues_pct[col]

ax.set_title("Tỷ lệ các nhóm lỗi dữ liệu quan trọng theo từng xe", fontsize=14)
ax.set_ylabel("Tỷ lệ (%)", fontsize=12)
ax.yaxis.set_major_formatter(mtick.PercentFormatter())
ax.legend(title="Nhóm lỗi", bbox_to_anchor=(1.05, 1), loc='upper left')
fig.savefig(os.path.join(out_dir, "Slide2_TyLeLoi.png"), bbox_inches="tight", dpi=300)
plt.close(fig)

# 2. Biểu đồ đánh dấu điểm bất thường (FuelLevel = 0)
# Chọn Car 3 vì xe này có 39 bản ghi lỗi Fuel = 0, phù hợp nhất để minh họa
print("Đang vẽ biểu đồ điểm bất thường (Slide2_DiemBatThuong.png)...")
car3 = data[data["VehicleID"] == "Car 3"].copy()
car3 = car3.sort_values("FuelTime")
anomalies = car3[(car3["FlagFuelZero"] == 1) | (car3["FlagDuplicateTime"] == 1)]

fig, ax = plt.subplots(figsize=(14, 5))
ax.plot(car3["FuelTime"], car3["FuelLevel"], label="FuelLevel", color="#1f77b4", linewidth=1)
ax.scatter(anomalies["FuelTime"], anomalies["FuelLevel"], color="red", marker="x", s=100, label="Lỗi (Fuel=0 / Trùng timestamp)", zorder=5)

ax.set_title("Biến thiên tín hiệu nhiên liệu và các điểm dữ liệu lỗi (Car 3)", fontsize=14)
ax.set_ylabel("FuelLevel (lít)")
ax.legend(loc="upper right")
fig.savefig(os.path.join(out_dir, "Slide2_DiemBatThuong.png"), bbox_inches="tight", dpi=300)
plt.close(fig)

print(f"Hoàn tất! Các biểu đồ đã được lưu tại thư mục: {out_dir}")
