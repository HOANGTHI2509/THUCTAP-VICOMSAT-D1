import sys
sys.stdout.reconfigure(encoding='utf-8')
import pandas as pd
import matplotlib.pyplot as plt
import os
import matplotlib.dates as mdates

out_dir = r"d:\THUCTAP_VICOMSAT\slide_plots"
os.makedirs(out_dir, exist_ok=True)

print("Đọc dữ liệu Car 3 đã trích xuất đặc trưng...")
# Chọn Car 3 vì nó có nhiều dao động sóng sánh rõ rệt nhất
df = pd.read_csv(r"d:\THUCTAP_VICOMSAT\CarFuelHistory_Car3_Features.csv")
df['FuelTime'] = pd.to_datetime(df['FuelTime'])
df = df.sort_values('FuelTime')

# Lấy 1 đoạn ngắn khoảng 2 tiếng đồng hồ (13h - 15h) để zoom kỹ vào các đặc trưng
most_records_date = df['FuelTime'].dt.date.value_counts().idxmax()
df_zoom = df[(df['FuelTime'].dt.date == most_records_date) & 
             (df['FuelTime'].dt.hour >= 13) & 
             (df['FuelTime'].dt.hour <= 15)].copy()

fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

# -------------------------------------------------------------------------
# Biểu đồ 1: Sự khác biệt giữa FuelLevel Gốc và Fuel_RollingMean_5
# -------------------------------------------------------------------------
ax1.plot(df_zoom['FuelTime'], df_zoom['FuelLevel'], color='lightblue', label='Nhiên liệu gốc (Nhiễu răng cưa)', linewidth=2)
ax1.plot(df_zoom['FuelTime'], df_zoom['Fuel_RollingMean_5'], color='navy', label='Fuel_RollingMean_5 (Đường trung bình trượt)', linewidth=2)
ax1.set_title("Đặc trưng 1: Fuel_RollingMean_5 giúp làm mượt bớt các gai nhiễu tức thời", fontsize=12, fontweight='bold')
ax1.set_ylabel("Lít", fontsize=11)
ax1.legend(loc='upper right')
ax1.grid(True, linestyle='--', alpha=0.5)

# -------------------------------------------------------------------------
# Biểu đồ 2: Fuel_RollingStd_5 (Độ phân tán / Rung lắc)
# -------------------------------------------------------------------------
ax2.plot(df_zoom['FuelTime'], df_zoom['Fuel_RollingStd_5'], color='red', label='Fuel_RollingStd_5 (Độ phân tán mức dầu)', linewidth=2)
ax2.set_title("Đặc trưng 2: Fuel_RollingStd_5 giúp AI định lượng được 'Độ rung lắc/sóng sánh' của bình dầu", fontsize=12, fontweight='bold')
ax2.set_ylabel("Độ lệch chuẩn", fontsize=11)
ax2.legend(loc='upper right')
ax2.grid(True, linestyle='--', alpha=0.5)

# -------------------------------------------------------------------------
# Biểu đồ 3: Speed và Acceleration (Gia tốc)
# -------------------------------------------------------------------------
ax3.plot(df_zoom['FuelTime'], df_zoom['Speed'], color='orange', label='Speed (Vận tốc)', linewidth=2)
ax3_twin = ax3.twinx()
ax3_twin.plot(df_zoom['FuelTime'], df_zoom['Acceleration'], color='green', label='Acceleration (Gia tốc)', linewidth=1.5, linestyle='--')

ax3.set_title("Đặc trưng 3 & 4: Speed và Acceleration giúp phát hiện các cú thốc ga / phanh gấp", fontsize=12, fontweight='bold')
ax3.set_ylabel("km/h", fontsize=11)
ax3_twin.set_ylabel("Gia tốc", fontsize=11)

# Gộp legend của ax3 và ax3_twin
lines, labels = ax3.get_legend_handles_labels()
lines2, labels2 = ax3_twin.get_legend_handles_labels()
ax3_twin.legend(lines + lines2, labels + labels2, loc='upper right')
ax3.grid(True, linestyle='--', alpha=0.5)

# Format trục X
ax3.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
fig.autofmt_xdate()

plt.tight_layout()
filename = "Slide7_Feature_Visualization.png"
fig.savefig(os.path.join(out_dir, filename), dpi=300)
plt.close(fig)

print(f"Đã tạo thành công biểu đồ trực quan hóa đặc trưng tại: {filename}")
