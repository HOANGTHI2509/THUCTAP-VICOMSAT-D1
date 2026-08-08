import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import sys

sys.stdout.reconfigure(encoding='utf-8')

def create_slide10_infographic():
    print("Bắt đầu tạo Biểu đồ minh họa Đặc trưng cho Slide 10...")
    
    # Đọc dữ liệu đã qua tiền xử lý (Dùng Car 1)
    df = pd.read_csv('data/processed/CarFuelHistory_Processed_Car1.csv')
    df['FuelTime'] = pd.to_datetime(df['FuelTime'])
    
    # Trích xuất một đoạn dữ liệu có sự chuyển giao rõ rệt giữa Chạy và Đỗ (Ví dụ từ dòng 480 đến 650)
    seg = df.iloc[480:650].copy()
    
    # Thiết lập giao diện biểu đồ
    plt.rcParams['figure.dpi'] = 300
    plt.rcParams['font.family'] = 'sans-serif'
    
    # Tạo Figure có 2 Subplots xếp dọc, dùng chung trục X (sharex=True)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True, gridspec_kw={'height_ratios': [2, 1.2]})
    
    # =========================================
    # BIỂU ĐỒ 1: FuelLevel & MovementState
    # =========================================
    ax1.plot(seg['FuelTime'], seg['FuelLevel'], color='#2563eb', marker='.', linestyle='-', linewidth=1.5, label='FuelLevel')
    ax1.set_title("Biến thiên Mức xăng & Trạng thái di chuyển (MovementState)", fontsize=14, fontweight='bold', pad=10)
    ax1.set_ylabel("Mức Xăng (Lít)", fontsize=12, fontweight='bold')
    
    # Tô nền trạng thái (MovementState)
    is_moving = seg['Speed'] > 0
    ax1.fill_between(seg['FuelTime'], seg['FuelLevel'].min() - 5, seg['FuelLevel'].max() + 5, 
                     where=is_moving, facecolor='#dcfce7', alpha=0.6, label='Moving (Đang chạy)')
    ax1.fill_between(seg['FuelTime'], seg['FuelLevel'].min() - 5, seg['FuelLevel'].max() + 5, 
                     where=~is_moving, facecolor='#f3f4f6', alpha=0.8, label='Stopped (Đang đỗ)')
    
    # Đánh dấu 1 điểm khởi đầu Segment (Minh họa)
    start_row = seg.iloc[0]
    ax1.annotate('Bắt đầu tính toán\ncửa sổ trượt (Rolling)', 
                 xy=(start_row['FuelTime'], start_row['FuelLevel']),
                 xytext=(20, 20), textcoords='offset points',
                 arrowprops=dict(facecolor='#8b5cf6', shrink=0.05, width=2, headwidth=8),
                 fontsize=11, color='#5b21b6', fontweight='bold',
                 bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#8b5cf6", alpha=0.9))
                 
    ax1.legend(loc='lower left', fontsize=10)
    ax1.grid(True, linestyle='--', alpha=0.5)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    # =========================================
    # BIỂU ĐỒ 2: RollingStd
    # =========================================
    # Đổi màu đường RollingStd sang màu cam/đỏ cho nổi bật
    ax2.plot(seg['FuelTime'], seg['RollingStd'], color='#ea580c', marker='', linestyle='-', linewidth=2, label='RollingStd')
    ax2.set_title("Biến động độ lệch chuẩn cục bộ (RollingStd)", fontsize=14, fontweight='bold', pad=10)
    ax2.set_ylabel("Độ lệch chuẩn (Lít)", fontsize=12, fontweight='bold')
    
    # Tô nền y hệt biểu đồ trên để đối chiếu
    ax2.fill_between(seg['FuelTime'], 0, seg['RollingStd'].max() + 2, 
                     where=is_moving, facecolor='#dcfce7', alpha=0.6)
    ax2.fill_between(seg['FuelTime'], 0, seg['RollingStd'].max() + 2, 
                     where=~is_moving, facecolor='#f3f4f6', alpha=0.8)
                     
    # Chú thích tương quan
    high_std_row = seg.loc[seg['RollingStd'].idxmax()]
    ax2.annotate('Khi xe chạy, xăng dao động\n-> RollingStd tăng vọt', 
                 xy=(high_std_row['FuelTime'], high_std_row['RollingStd']),
                 xytext=(-40, 30), textcoords='offset points', ha='center',
                 arrowprops=dict(facecolor='#ea580c', shrink=0.05, width=2, headwidth=8),
                 fontsize=11, color='#c2410c', fontweight='bold',
                 bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#ea580c", alpha=0.9))
                 
    # Định dạng trục X cho cả 2 biểu đồ (vì dùng chung)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax2.tick_params(axis='x', rotation=0, labelsize=11)
    ax2.set_xlabel("Thời gian (Giờ:Phút)", fontsize=12, fontweight='bold')
    ax2.grid(True, linestyle='--', alpha=0.5)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.set_ylim(0, seg['RollingStd'].max() + 3)

    plt.tight_layout()
    output_path = 'slide10_features.png'
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    print(f"-> Đã lưu ảnh thành công: {output_path}")

if __name__ == "__main__":
    create_slide10_infographic()
