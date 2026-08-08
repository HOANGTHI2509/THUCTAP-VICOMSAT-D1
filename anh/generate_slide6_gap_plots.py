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
    all_data.append(df)

data = pd.concat(all_data, ignore_index=True)

# ==============================================================================
# 1. So sánh tần suất gián đoạn giữa các xe
# ==============================================================================
print("Vẽ biểu đồ so sánh gián đoạn...")
gap_stats = []
for car in data["VehicleID"].unique():
    df_car = data[data["VehicleID"] == car]
    long_gaps = df_car[df_car["TimeGapMinutes"] > 30]
    gap_stats.append({
        "VehicleID": car,
        "Số lần mất tín hiệu (>30p)": len(long_gaps),
        "Tổng thời gian mất (Giờ)": long_gaps["TimeGapMinutes"].sum() / 60
    })

df_gap = pd.DataFrame(gap_stats).sort_values("VehicleID")

fig, ax1 = plt.subplots(figsize=(10, 6))
sns.barplot(data=df_gap, x="VehicleID", y="Số lần mất tín hiệu (>30p)", color="#1f77b4", ax=ax1, alpha=0.8)
ax1.set_ylabel("Số lần gián đoạn (> 30 phút)", color="#1f77b4", fontsize=12, fontweight='bold')
ax1.tick_params(axis='y', labelcolor="#1f77b4")

# Thêm số liệu lên cột
for p in ax1.patches:
    ax1.text(p.get_x() + p.get_width()/2., p.get_height() + 5, f'{int(p.get_height())}', 
            fontsize=11, color='#1f77b4', ha='center', va='bottom', fontweight='bold')

ax2 = ax1.twinx()
sns.lineplot(data=df_gap, x="VehicleID", y="Tổng thời gian mất (Giờ)", color="#d62728", marker="o", linewidth=2.5, markersize=10, ax=ax2)
ax2.set_ylabel("Tổng thời gian mất tín hiệu cộng dồn (Giờ)", color="#d62728", fontsize=12, fontweight='bold')
ax2.tick_params(axis='y', labelcolor="#d62728")

plt.title("So sánh mức độ mất tín hiệu / Gián đoạn truyền dữ liệu giữa các xe", fontsize=15, fontweight='bold')
fig.tight_layout()
fig.savefig(os.path.join(out_dir, "Slide6_SignalLoss_Comparison.png"), dpi=300)
plt.close(fig)

# ==============================================================================
# 2. Chi tiết chu kỳ truyền thiếu ổn định của CAR 5
# ==============================================================================
print("Vẽ biểu đồ chi tiết CAR 5...")
car5 = data[data["VehicleID"] == "Car 5"].copy()

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

# Trục trên: Vẽ số lượng bản ghi mỗi ngày (thấy rõ các ngày bị 0)
car5['Date'] = car5['FuelTime'].dt.date
records_per_day = car5.groupby('Date').size()
# Tạo một dải ngày liên tục để những ngày không có data sẽ là 0
all_days = pd.date_range(start=car5['Date'].min(), end=car5['Date'].max())
records_per_day = records_per_day.reindex(all_days.date, fill_value=0)

ax1.bar(records_per_day.index, records_per_day.values, color='teal', alpha=0.7)
ax1.set_ylabel("Số bản ghi / ngày", fontsize=12, fontweight='bold')
ax1.set_title("Biểu đồ minh họa chu kỳ truyền dữ liệu cực kỳ thiếu ổn định của CAR 5", fontsize=15, fontweight='bold')

# Highlight những khoảng không có dữ liệu (0 bản ghi)
zero_days = records_per_day[records_per_day == 0].index
for zd in zero_days:
    ax1.axvspan(zd, zd + pd.Timedelta(days=1), color='red', alpha=0.2, lw=0)

ax1.plot([], [], color='red', alpha=0.2, label='Các ngày mất hoàn toàn tín hiệu', linewidth=10)
ax1.legend(loc="upper right")

# Trục dưới: Vẽ TimeGapMinutes để thấy các cú "nhảy"
ax2.plot(car5["FuelTime"], car5["TimeGapMinutes"] / 60, color='crimson', marker='.', linestyle='none', markersize=10, alpha=0.7)
ax2.set_ylabel("Độ dài từng lần mất sóng (Giờ)", fontsize=12, fontweight='bold')
ax2.set_xlabel("Thời gian", fontsize=12)

# Đường baseline 0h
ax2.axhline(0, color='black', linewidth=0.5)

ax2.xaxis.set_major_formatter(mdates.DateFormatter('%d/%m'))
fig.autofmt_xdate()
fig.tight_layout()

fig.savefig(os.path.join(out_dir, "Slide6_Car5_Unstable_Transmission.png"), dpi=300)
plt.close(fig)

print("Hoàn tất tạo biểu đồ cho Slide 6.")
