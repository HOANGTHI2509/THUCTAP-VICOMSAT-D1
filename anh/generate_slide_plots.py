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
import seaborn as sns

out_dir = r"d:\THUCTAP_VICOMSAT\slide_plots"
os.makedirs(out_dir, exist_ok=True)

print("Đang đọc dữ liệu...")
csv_files = glob.glob(r"d:\THUCTAP_VICOMSAT\CarFuelHistory_*_Flags.csv")
all_data = []
for f in csv_files:
    df = pd.read_csv(f)
    df["FuelTime"] = pd.to_datetime(df["FuelTime"])
    all_data.append(df)
data = pd.concat(all_data, ignore_index=True)

# Lọc bỏ các xe không hợp lệ nếu có (đảm bảo chỉ có Car 1-5)
data = data[data["VehicleID"].isin(["Car 1", "Car 2", "Car 3", "Car 4", "Car 5"])]

def save_plot(fig, filename):
    filepath = os.path.join(out_dir, filename)
    fig.savefig(filepath, bbox_inches="tight", dpi=300)
    print(f"Đã lưu: {filename}")
    plt.close(fig)

# 1. Biểu đồ cột: Số bản ghi của từng xe
print("Vẽ Biểu đồ 1...")
fig, ax = plt.subplots(figsize=(8, 5))
record_counts = data["VehicleID"].value_counts().sort_index()
bars = ax.bar(record_counts.index, record_counts.values, color='skyblue')
ax.set_title("Số lượng bản ghi dữ liệu theo từng xe", fontsize=14)
ax.set_ylabel("Số lượng bản ghi", fontsize=12)
for bar in bars:
    yval = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, yval + 100, f"{yval:,}", ha='center', va='bottom')
save_plot(fig, "1_SoBanGhi.png")

# 2. Biểu đồ cột chồng: Tỷ lệ vấn đề dữ liệu
print("Vẽ Biểu đồ 2...")
issues = data.groupby("VehicleID").agg({
    "FlagFuelZero": "sum",
    "FlagDuplicateTime": "sum",
    "FlagLongGap": "sum",
    "FlagSpeedGpsConflict": "sum"
})
issues_pct = issues.div(record_counts, axis=0) * 100

fig, ax = plt.subplots(figsize=(10, 6))
bottom = np.zeros(len(issues_pct))
colors = ['#ff9999','#66b3ff','#99ff99','#ffcc99']
labels = ['Fuel = 0', 'Timestamp trùng', 'Mất dữ liệu >30p', 'Mâu thuẫn Speed-GPS']

for i, col in enumerate(issues.columns):
    ax.bar(issues_pct.index, issues_pct[col], bottom=bottom, label=labels[i], color=colors[i])
    bottom += issues_pct[col]

ax.set_title("Tỷ lệ các vấn đề dữ liệu trên tổng bản ghi (%)", fontsize=14)
ax.set_ylabel("Tỷ lệ (%)", fontsize=12)
ax.yaxis.set_major_formatter(mtick.PercentFormatter())
ax.legend(title="Nhóm lỗi", bbox_to_anchor=(1.05, 1), loc='upper left')
save_plot(fig, "2_TyLeLoi.png")

# 3. Phân bố nhóm TimeGap
print("Vẽ Biểu đồ 3...")
bins = [-1, 0.5, 5, 10, 30, float('inf')]
labels_tg = ['0 phút', '1-5 phút', '5-10 phút', '10-30 phút', 'Trên 30 phút']
data['TimeGapCategory'] = pd.cut(data['TimeGapMinutes'].fillna(0), bins=bins, labels=labels_tg)
tg_counts = data.groupby(['VehicleID', 'TimeGapCategory'], observed=False).size().unstack(fill_value=0)
tg_pct = tg_counts.div(tg_counts.sum(axis=1), axis=0) * 100

fig, ax = plt.subplots(figsize=(10, 6))
bottom = np.zeros(len(tg_pct))
colors_tg = ['#2ca02c', '#1f77b4', '#ff7f0e', '#d62728', '#9467bd']
for i, col in enumerate(tg_pct.columns):
    ax.bar(tg_pct.index, tg_pct[col], bottom=bottom, label=col, color=colors_tg[i])
    bottom += tg_pct[col]

ax.set_title("Phân bố tần suất gửi dữ liệu (TimeGap)", fontsize=14)
ax.set_ylabel("Tỷ lệ (%)", fontsize=12)
ax.yaxis.set_major_formatter(mtick.PercentFormatter())
ax.legend(title="Khoảng thời gian", bbox_to_anchor=(1.05, 1), loc='upper left')
save_plot(fig, "3_PhanBoTimeGap.png")

# Lấy Car 1 để vẽ biểu đồ 4, 5, 6
car1 = data[data["VehicleID"] == "Car 1"].copy()
car1 = car1.sort_values("FuelTime")

# 4. FuelLevel toàn thời gian
print("Vẽ Biểu đồ 4...")
fig, ax = plt.subplots(figsize=(14, 5))
ax.plot(car1["FuelTime"], car1["FuelLevel"], linewidth=1, color="#1f77b4")
ax.set_title("Biến thiên tín hiệu FuelLevel toàn thời gian (Car 1)", fontsize=14)
ax.set_ylabel("FuelLevel (lít)")
save_plot(fig, "4_FuelLevel_ToanThoiGian.png")

# 5. FuelLevel phóng to 1 đoạn
print("Vẽ Biểu đồ 5...")
car1['Date'] = car1['FuelTime'].dt.date
most_records_date = car1['Date'].value_counts().idxmax()
car1_zoom = car1[car1['Date'] == most_records_date]

fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(car1_zoom["FuelTime"], car1_zoom["FuelLevel"], marker='.', linestyle='-', color="#d62728")
ax.set_title(f"Tín hiệu FuelLevel được phóng to (Ngày {most_records_date})", fontsize=14)
ax.set_ylabel("FuelLevel (lít)")
save_plot(fig, "5_FuelLevel_PhongTo.png")

# 6. FuelLevel + Speed
print("Vẽ Biểu đồ 6...")
fig, ax1 = plt.subplots(figsize=(14, 6))
ax1.plot(car1_zoom["FuelTime"], car1_zoom["FuelLevel"], color='blue', label='FuelLevel')
ax1.set_ylabel('FuelLevel (lít)', color='blue', fontsize=12)
ax1.tick_params(axis='y', labelcolor='blue')

ax2 = ax1.twinx()
ax2.plot(car1_zoom["FuelTime"], car1_zoom["Speed"], color='orange', alpha=0.7, label='Speed')
ax2.set_ylabel('Speed', color='orange', fontsize=12)
ax2.tick_params(axis='y', labelcolor='orange')

fig.suptitle(f"Mối liên hệ giữa FuelLevel và Tốc độ (Ngày {most_records_date})", fontsize=14)
# Add legends manually to avoid overlapping
lines_1, labels_1 = ax1.get_legend_handles_labels()
lines_2, labels_2 = ax2.get_legend_handles_labels()
ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc="upper right", bbox_to_anchor=(0.95, 0.95))
save_plot(fig, "6_FuelLevel_Speed.png")

# 7. Boxplot DeltaFuel Dừng vs Chạy
print("Vẽ Biểu đồ 7...")
data["MovementState"] = np.where(data["Speed"] > 0, "Đang chạy", "Dừng/Đỗ")
# Bỏ các nhiễu quá khổng lồ hoặc nạp rút xăng (>5 lít) để boxplot nhìn rõ được mức độ dao động nhỏ
clean_data_for_boxplot = data[data["AbsDeltaFuel"] < 5].copy()

fig, ax = plt.subplots(figsize=(10, 6))
sns.boxplot(x="VehicleID", y="AbsDeltaFuel", hue="MovementState", data=clean_data_for_boxplot, ax=ax, showfliers=False, palette={"Dừng/Đỗ": "lightblue", "Đang chạy": "salmon"})
ax.set_title("Mức độ dao động (|DeltaFuel|) khi xe Dừng vs Chạy", fontsize=14)
ax.set_ylabel("Độ dao động |DeltaFuel| (lít)")
ax.set_xlabel("")
save_plot(fig, "7_Boxplot_DeltaFuel_State.png")

print("Hoàn tất!")
