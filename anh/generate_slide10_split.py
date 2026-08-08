import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import sys

sys.stdout.reconfigure(encoding='utf-8')

def format_axis(ax, hide_x=False):
    if not hide_x:
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M\n%d/%m/%Y'))
        ax.set_xlabel("Thời gian (FuelTime)", fontsize=12, fontweight='bold')
    else:
        ax.set_xticklabels([])
        
    ax.tick_params(axis='x', rotation=0, labelsize=10)
    ax.tick_params(axis='y', labelsize=11)
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

def create_split_slide10_infographic():
    print("Bắt đầu tạo 2 biểu đồ độc lập cho Slide 10 (Thêm Metadata Xe/Ngày)...")
    plt.rcParams['figure.dpi'] = 300
    plt.rcParams['font.family'] = 'sans-serif'
    
    # 1. Đọc dữ liệu
    df = pd.read_csv('data/processed/CarFuelHistory_Processed_Car1.csv')
    df['FuelTime'] = pd.to_datetime(df['FuelTime'])
    df['AbsDeltaFuel'] = df['DeltaFuel'].abs()
    
    # 2. Tìm một đoạn Dừng -> Chạy -> Dừng (Sử dụng đoạn dữ liệu nguyên bản)
    seg = df.iloc[450:650].copy()
    
    # Lấy thông tin ngày tháng để gắn vào Title
    date_str = seg['FuelTime'].iloc[0].strftime('%d/%m/%Y')
    car_id = "CAR 1"
    
    # Tìm điểm có biến thiên tức thời mạnh nhất trong thực tế
    idx_spike = seg['AbsDeltaFuel'].idxmax()
    err_row = seg.loc[idx_spike]
    
    is_moving = seg['Speed'] > 0

    # =========================================
    # BIỂU ĐỒ 1: FuelLevel & Trạng thái xe
    # =========================================
    fig1, ax1 = plt.subplots(figsize=(10, 5))
    ax1.plot(seg['FuelTime'], seg['FuelLevel'], color='#2563eb', marker='.', linestyle='-', linewidth=1.5, label='FuelLevel')
    
    ax1.fill_between(seg['FuelTime'], seg['FuelLevel'].min() - 5, seg['FuelLevel'].max() + 5, 
                     where=is_moving, facecolor='#dcfce7', alpha=0.6, label='Moving (Di chuyển)')
    ax1.fill_between(seg['FuelTime'], seg['FuelLevel'].min() - 5, seg['FuelLevel'].max() + 5, 
                     where=~is_moving, facecolor='#f3f4f6', alpha=0.8, label='Stopped (Dừng)')
                     
    # Đánh dấu điểm nhiễu tức thời
    ax1.scatter(err_row['FuelTime'], err_row['FuelLevel'], color='red', s=100, zorder=5, marker='X')
    ax1.annotate('Điểm biến động tức thời lớn nhất\n(Dao động do xe xóc)', 
                 xy=(err_row['FuelTime'], err_row['FuelLevel']),
                 xytext=(10, 30), textcoords='offset points',
                 arrowprops=dict(facecolor='red', shrink=0.05, width=1.5, headwidth=6),
                 fontsize=11, color='red', fontweight='bold',
                 bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="red", alpha=0.9))
    
    ax1.set_title(f"Tín hiệu FuelLevel và trạng thái chuyển động\n[Dữ liệu: {car_id} | {date_str}]", fontsize=15, fontweight='bold', pad=15)
    ax1.set_ylabel("FuelLevel (Lít)", fontsize=13, fontweight='bold')
    ax1.legend(loc='lower left')
    format_axis(ax1)
    
    plt.tight_layout()
    fig1.savefig('anh/Slide10_1_fuellevel_state.png')
    plt.close(fig1)

    # =========================================
    # BIỂU ĐỒ 2: Đặc trưng dao động cục bộ
    # =========================================
    fig2, ax2 = plt.subplots(figsize=(10, 5))
    
    # Vẽ RollingStd
    ax2.plot(seg['FuelTime'], seg['RollingStd'], color='#ea580c', marker='', linestyle='-', linewidth=2, label='RollingStd (Độ dao động nền)')
    
    # Vẽ AbsDeltaFuel
    ax2.plot(seg['FuelTime'], seg['AbsDeltaFuel'], color='#8b5cf6', marker='.', linestyle='--', linewidth=1, alpha=0.7, label='AbsDeltaFuel (Biến động tức thời)')
    
    # Tô nền đồng bộ với biểu đồ 1
    max_y = max(seg['RollingStd'].max(), seg['AbsDeltaFuel'].max())
    ax2.fill_between(seg['FuelTime'], -1, max_y + 2, 
                     where=is_moving, facecolor='#dcfce7', alpha=0.6)
    ax2.fill_between(seg['FuelTime'], -1, max_y + 2, 
                     where=~is_moving, facecolor='#f3f4f6', alpha=0.8)
                     
    ax2.annotate('Thay đổi đột ngột\n-> AbsDeltaFuel tăng vọt', 
                 xy=(err_row['FuelTime'], err_row['AbsDeltaFuel']),
                 xytext=(15, 20), textcoords='offset points',
                 arrowprops=dict(facecolor='#8b5cf6', shrink=0.05, width=1.5, headwidth=6),
                 fontsize=11, color='#6d28d9', fontweight='bold',
                 bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#8b5cf6", alpha=0.9))
                 
    ax2.set_title(f"Đặc trưng dao động cục bộ theo thời gian\n[Dữ liệu: {car_id} | {date_str}]", fontsize=15, fontweight='bold', pad=15)
    ax2.set_ylabel("Mức biến động (Lít)", fontsize=13, fontweight='bold')
    ax2.legend(loc='upper left')
    format_axis(ax2)
    ax2.set_ylim(-0.5, max_y + 1)
    
    plt.tight_layout()
    fig2.savefig('anh/Slide10_2_rolling_delta.png')
    plt.close(fig2)

    print("Hoàn tất xuất 2 ảnh độc lập (Đã bổ sung Metadata)!")

if __name__ == "__main__":
    create_split_slide10_infographic()
