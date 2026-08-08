import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import sys

sys.stdout.reconfigure(encoding='utf-8')

def create_slide9_infographic():
    print("Bắt đầu tạo Biểu đồ minh họa đặc trưng cho Slide 9...")
    
    # 1. Đọc dữ liệu (Dùng Car 1)
    df = pd.read_csv('data/processed/CarFuelHistory_Processed_Car1.csv')
    df['FuelTime'] = pd.to_datetime(df['FuelTime'])
    
    # Để có một biểu đồ hội tụ đủ 5 yếu tố tự nhiên nhất, ta lấy 150 mẫu đầu tiên của Car 1
    # Đoạn này có: Fuel=0 (mẫu 0), nhảy vọt (mẫu 1), xe chạy (dao động), và ta sẽ lấy thêm 1 điểm TimeGap
    seg = df.head(200).copy()
    
    # 2. Thiết lập biểu đồ
    plt.rcParams['figure.dpi'] = 300
    plt.rcParams['font.family'] = 'sans-serif'
    
    fig, ax = plt.subplots(figsize=(14, 7))
    fig.suptitle('MINH HỌA CÁC ĐẶC TRƯNG SAU TIỀN XỬ LÝ', fontsize=20, fontweight='bold', y=0.95, color='#1f2937')
    
    # 3. Vẽ line chính
    ax.plot(seg['FuelTime'], seg['FuelLevel'], color='#2563eb', marker='.', linestyle='-', linewidth=2, label='FuelLevel (Mức xăng)')
    
    # Tô nền trạng thái (MovementState)
    # Lấy các điểm Stopped và Moving để tô nền
    is_moving = seg['Speed'] > 0
    ax.fill_between(seg['FuelTime'], 0, 250, where=is_moving, facecolor='#dcfce7', alpha=0.5, label='Vùng Moving (Speed > 0)')
    ax.fill_between(seg['FuelTime'], 0, 250, where=~is_moving, facecolor='#f3f4f6', alpha=0.5, label='Vùng Stopped (Speed = 0)')

    # 4. Chú thích các Đặc trưng (Annotations)
    
    # A. QualityFlag (FuelLevel = 0)
    zero_row = seg.iloc[0] # Dòng đầu tiên của Car 1 có Fuel=0
    ax.annotate('Điểm A: FuelLevel = 0\n-> Đánh cờ QualityFlag', 
                xy=(zero_row['FuelTime'], zero_row['FuelLevel']),
                xytext=(30, 40), textcoords='offset points',
                arrowprops=dict(facecolor='#ef4444', shrink=0.05, width=2, headwidth=8),
                fontsize=11, color='#b91c1c', fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#ef4444", alpha=0.9))

    # B. DeltaFuel & FuelRate (Biến động mạnh)
    jump_row_prev = seg.iloc[0]
    jump_row = seg.iloc[1]
    ax.annotate(f'Điểm B: Nhảy vọt mức xăng\n-> DeltaFuel, FuelRate cực lớn', 
                xy=(jump_row['FuelTime'], jump_row['FuelLevel']),
                xytext=(30, -50), textcoords='offset points',
                arrowprops=dict(facecolor='#f59e0b', shrink=0.05, width=2, headwidth=8),
                fontsize=11, color='#b45309', fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#f59e0b", alpha=0.9))
                
    # C. TimeGap / SegmentID (Tìm một khoảng trống thời gian)
    gap_idx = seg['TimeGapMinutes'].idxmax()
    gap_row_prev = seg.loc[gap_idx - 1]
    gap_row = seg.loc[gap_idx]
    
    # Nếu gap tự nhiên trong 200 dòng đầu chưa đủ lớn, ta giả lập việc ngắt segment ở gap lớn nhất này để minh họa
    ax.axvspan(gap_row_prev['FuelTime'], gap_row['FuelTime'], color='#fca5a5', alpha=0.4)
    mid_time = gap_row_prev['FuelTime'] + (gap_row['FuelTime'] - gap_row_prev['FuelTime'])/2
    ax.annotate('Khoảng C: TimeGap lớn\n-> Bắt đầu SegmentID mới', 
                xy=(mid_time, 100),
                xytext=(0, 60), textcoords='offset points',
                arrowprops=dict(facecolor='#8b5cf6', shrink=0, width=2, headwidth=8),
                fontsize=11, ha='center', color='#5b21b6', fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#8b5cf6", alpha=0.9))

    # D. RollingStd (Vùng dao động mạnh)
    # Lấy điểm có RollingStd cao nhất (khi đang di chuyển)
    std_idx = seg['RollingStd'].idxmax()
    std_row = seg.loc[std_idx]
    ax.annotate(f'Vùng D: Dao động (Sloshing)\n-> RollingStd = {std_row["RollingStd"]:.2f} (Rất cao)', 
                xy=(std_row['FuelTime'], std_row['FuelLevel']),
                xytext=(-30, -70), textcoords='offset points', ha='center',
                arrowprops=dict(facecolor='#3b82f6', shrink=0.05, width=2, headwidth=8),
                fontsize=11, color='#1d4ed8', fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#3b82f6", alpha=0.9))

    # 5. Định dạng trục
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M\n%d/%m/%Y'))
    ax.tick_params(axis='x', rotation=0, labelsize=10)
    ax.tick_params(axis='y', labelsize=10)
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.set_ylabel("Mức Xăng (Lít)", fontsize=13, fontweight='bold')
    ax.set_xlabel("Thời gian (FuelTime)", fontsize=13, fontweight='bold')
    ax.set_ylim(0, 220)
    
    # Legend
    ax.legend(loc='lower right', fontsize=11, framealpha=0.9)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    output_path = 'slide9_features_illustration.png'
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    print(f"-> Đã lưu biểu đồ thành công: {output_path}")

if __name__ == "__main__":
    create_slide9_infographic()
