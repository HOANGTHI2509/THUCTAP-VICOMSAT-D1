import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

print("Đang đọc dữ liệu...")
df = pd.read_csv('data/processed/CarFuelHistory_Car1_Features.csv', parse_dates=['FuelTime'])

# Lọc ra các Segment có độ dài đáng kể (ví dụ > 50 điểm dữ liệu) để vẽ cho đẹp, tránh các "chấm"
segment_counts = df['SegmentID'].value_counts()
valid_segments = segment_counts[segment_counts > 50].index.sort_values()

if len(valid_segments) >= 3:
    # Chọn 3 segment liên tiếp hợp lệ
    target_segments = valid_segments[1:4]
else:
    target_segments = valid_segments[:3]

df_subset = df[df['SegmentID'].isin(target_segments)].copy()
df_subset = df_subset.sort_values('FuelTime')

plt.figure(figsize=(14, 7))
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

for i, seg_id in enumerate(target_segments):
    seg_data = df_subset[df_subset['SegmentID'] == seg_id]
    color = colors[i % len(colors)]
    
    plt.plot(seg_data['FuelTime'], seg_data['FuelLevel'], linestyle='-', linewidth=2.5, color=color, alpha=0.9, label=f'Segment {seg_id}')
    
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%d/%m/%Y %H:%M'))
plt.gcf().autofmt_xdate()

plt.title('Minh họa Chia Đoạn Dữ Liệu (SegmentID) dựa trên TimeGap (Đã lọc đoạn dài)', fontsize=16, fontweight='bold')
plt.xlabel('Thời gian (FuelTime)', fontsize=12)
plt.ylabel('Mức nhiên liệu (Lít)', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend(fontsize=12)
plt.tight_layout()

save_path = 'segment_visualization_fixed.png'
plt.savefig(save_path, dpi=150)
print(f"Đã lưu biểu đồ thành công tại {save_path}")
