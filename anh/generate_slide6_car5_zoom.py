import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import sys
sys.stdout.reconfigure(encoding='utf-8')
import pandas as pd
import matplotlib.pyplot as plt
import os
import matplotlib.dates as mdates

out_dir = r"d:\THUCTAP_VICOMSAT\slide_plots"
os.makedirs(out_dir, exist_ok=True)

print("Đọc dữ liệu CAR 5...")
df = pd.read_csv(r"d:\THUCTAP_VICOMSAT\CarFuelHistory_Car5_Flags.csv")
df['FuelTime'] = pd.to_datetime(df['FuelTime'])
df = df.sort_values('FuelTime')

# Lấy dữ liệu bao trọn khoảng mất tín hiệu dài nhất của CAR 5
start_date = pd.to_datetime("2026-02-10")
end_date = pd.to_datetime("2026-02-26")
df_zoom = df[(df['FuelTime'] >= start_date) & (df['FuelTime'] <= end_date)].copy()

fig, ax = plt.subplots(figsize=(12, 6))

# Vẽ đường nét đứt màu đỏ với chấm tròn (marker) giống hệt ảnh mẫu
ax.plot(df_zoom['FuelTime'], df_zoom['FuelLevel'], 
        color='#e74c3c', linestyle='--', marker='.', markersize=8, linewidth=1.5)

ax.set_title("Biểu đồ minh họa dữ liệu bị đứt gãy (Mất kết nối GPS/GPRS) - CAR 5", fontsize=14, fontweight='bold')
ax.set_xlabel("Thời gian", fontsize=12)
ax.set_ylabel("Mức nhiên liệu (Lít)", fontsize=12)

# Bật lưới nền mờ giống ảnh mẫu để dễ quan sát
ax.grid(True, linestyle='-', color='#ecf0f1', alpha=0.9)

# Format Date X-axis
ax.xaxis.set_major_locator(mdates.DayLocator(interval=2))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%d/%m'))
fig.autofmt_xdate()

fig.tight_layout()

filename = "Slide6_Car5_Massive_Gap_Zoom.png"
fig.savefig(os.path.join(out_dir, filename), dpi=300)
plt.close(fig)

print(f"Hoàn tất tạo lại biểu đồ {filename} theo chuẩn ảnh mẫu.")
