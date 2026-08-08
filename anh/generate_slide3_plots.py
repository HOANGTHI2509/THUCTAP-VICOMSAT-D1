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

csv_files = sorted(glob.glob(r"d:\THUCTAP_VICOMSAT\CarFuelHistory_*_Flags.csv"))
all_data = []
stats = []

for f in csv_files:
    df = pd.read_csv(f)
    if "VehicleID" not in df.columns:
        continue
    car = df["VehicleID"].iloc[0]
    if car not in ["Car 1", "Car 2", "Car 3", "Car 4", "Car 5"]: continue
    
    df["FuelTime"] = pd.to_datetime(df["FuelTime"])
    
    if "TimeGapMinutes" in df.columns:
        mode_gap = df["TimeGapMinutes"].mode()
        mode_val = mode_gap.iloc[0] if not mode_gap.empty else 0
        med_gap = df["TimeGapMinutes"].median()
        max_gap = df["TimeGapMinutes"].max()
        on_beat = (df["TimeGapMinutes"] == mode_val).mean() * 100 if not mode_gap.empty else 0
        long_gap = df["FlagLongGap"].sum() if "FlagLongGap" in df.columns else 0
    else:
        mode_val, med_gap, max_gap, on_beat, long_gap = 0, 0, 0, 0, 0
        
    stats.append({
        "Xe": car,
        "TimeGap phổ biến": f"{mode_val:.1f} ph",
        "TimeGap trung vị": f"{med_gap:.1f} ph",
        "Tỷ lệ đúng chu kỳ phổ biến": f"{on_beat:.1f}%",
        "Gap lớn nhất": f"{max_gap:.0f} ph",
        "Số gap > 30 phút": f"{long_gap:,}"
    })
    
    all_data.append(df)

data = pd.concat(all_data, ignore_index=True)

print("=== MARKDOWN TABLE START ===")
print("| Xe | TimeGap phổ biến | TimeGap trung vị | Tỷ lệ đúng chu kỳ phổ biến | Gap lớn nhất | Số gap > 30 phút |")
print("|---|---|---|---|---|---|")
for row in stats:
    print(f"| {row['Xe']} | {row['TimeGap phổ biến']} | {row['TimeGap trung vị']} | {row['Tỷ lệ đúng chu kỳ phổ biến']} | {row['Gap lớn nhất']} | {row['Số gap > 30 phút']} |")
print("=== MARKDOWN TABLE END ===")

# 1. Grouped Bar Chart cho các khoảng TimeGap bất thường (loại trừ nhóm 3-6 phút)
bins = [-1, 3, 6, 10, 30, float('inf')]
labels_tg = ['≤ 3 phút', '3–6 phút', '6–10 phút', '10–30 phút', '> 30 phút']
data['TimeGapCategory'] = pd.cut(data['TimeGapMinutes'].fillna(0), bins=bins, labels=labels_tg)
tg_counts = data.groupby(['VehicleID', 'TimeGapCategory'], observed=False).size().unstack(fill_value=0)

# Loại bỏ nhóm '3–6 phút' vì chiếm tỷ lệ quá lớn (94-99%)
anomalous_counts = tg_counts.drop(columns=['3–6 phút'])

fig, ax = plt.subplots(figsize=(12, 6))
# Vẽ biểu đồ cột nhóm (Grouped Bar Chart)
anomalous_counts.plot(kind='bar', ax=ax, width=0.8, color=['#2ca02c', '#ff7f0e', '#d62728', '#9467bd'])

ax.set_title("Số lượng các khoảng gián đoạn và chu kỳ bất thường (Loại trừ chu kỳ 3-6 phút)", fontsize=14)
ax.set_ylabel("Số lượng (lần)", fontsize=12)
ax.set_xlabel("")
plt.xticks(rotation=0)
ax.legend(title="Khoảng thời gian", bbox_to_anchor=(1.05, 1), loc='upper left')

# Thêm nhãn số liệu trên đầu cột
for p in ax.patches:
    height = p.get_height()
    if height > 0:
        ax.annotate(f'{int(height)}', (p.get_x() + p.get_width() / 2., height),
                    ha='center', va='bottom', fontsize=9, rotation=0, xytext=(0, 2), textcoords='offset points')

# Tăng y-limit để không bị mất nhãn
ax.set_ylim(0, anomalous_counts.max().max() * 1.1)

fig.savefig(os.path.join(out_dir, "Slide3_PhanBoTimeGap.png"), bbox_inches="tight", dpi=300)
plt.close(fig)

# 2. Histogram TimeGap for Car 5
car5 = data[data["VehicleID"] == "Car 5"].copy()
car5_filtered = car5[(car5["TimeGapMinutes"] > 0) & (car5["TimeGapMinutes"] <= 100)]

fig, ax = plt.subplots(figsize=(10, 5))
ax.hist(car5_filtered["TimeGapMinutes"].dropna(), bins=50, color='#9467bd', edgecolor='black')
ax.set_title("Phân bố tần suất gửi dữ liệu (TimeGap) của Car 5 (Zoom < 100 phút)", fontsize=14)
ax.set_xlabel("TimeGap (phút)")
ax.set_ylabel("Số lần xuất hiện")
fig.savefig(os.path.join(out_dir, "Slide3_Histogram_Car5.png"), bbox_inches="tight", dpi=300)
plt.close(fig)
