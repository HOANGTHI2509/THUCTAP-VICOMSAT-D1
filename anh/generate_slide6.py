import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import sys
sys.stdout.reconfigure(encoding='utf-8')

def create_slide6_infographic():
    print("Bắt đầu tạo Biểu đồ Infographic cho Slide 6 (Chỉnh sửa tọa độ chữ)...")
    
    # Thiết lập phong cách cao cấp cho biểu đồ
    plt.rcParams['figure.dpi'] = 300
    plt.rcParams['font.family'] = 'sans-serif'
    
    fig, axs = plt.subplots(2, 2, figsize=(16, 11))
    fig.suptitle('PHÂN TÍCH CÁC BIỂU HIỆN NHIỄU QUAN SÁT ĐƯỢC TỪ CẢM BIẾN (RAW DATA)', fontsize=22, fontweight='bold', y=0.98, color='#1f2937')

    # Hàm định dạng trục thời gian
    def format_axis(ax):
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M\n%d/%m/%Y'))
        ax.tick_params(axis='x', rotation=0, labelsize=9)
        ax.tick_params(axis='y', labelsize=10)
        ax.grid(True, linestyle='--', alpha=0.5, color='#9ca3af')
        ax.set_ylabel("Mức Xăng (Lít)", fontsize=12, fontweight='bold', color='#4b5563')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    # ==========================================
    # 1. Dao động răng cưa & Sloshing (Car 5)
    # ==========================================
    df5 = pd.read_csv('data/processed/CarFuelHistory_Processed_Car5.csv')
    df5['FuelTime'] = pd.to_datetime(df5['FuelTime'])
    
    seg_5 = df5.iloc[1000:1150]
    date_5 = seg_5['FuelTime'].iloc[0].strftime('%d/%m/%Y')
    ax1 = axs[0, 0]
    ax1.plot(seg_5['FuelTime'], seg_5['FuelLevel'], color='#f87171', marker='.', linestyle='-', alpha=0.9, linewidth=1.5)
    ax1.set_title(f"1. Dao động răng cưa & Rung lắc (Sloshing)\n[Dữ liệu: CAR 5 | {date_5}]", fontsize=15, fontweight='bold', color='#111827', pad=15)
    format_axis(ax1)
    
    highest_idx = seg_5['FuelLevel'].idxmax()
    highest_row = seg_5.loc[highest_idx]
    # Đẩy text xuống dưới và sang phải để không chạm Title
    ax1.annotate('Xăng sóng sánh mạnh (Sloshing)\ntạo thành các gai nhọn (Spikes) liên tục', 
                 xy=(highest_row['FuelTime'], highest_row['FuelLevel']),
                 xytext=(30, -40), textcoords='offset points',
                 arrowprops=dict(facecolor='#ef4444', shrink=0.05, width=2, headwidth=8),
                 fontsize=11, color='#b91c1c', ha='center', bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#ef4444", alpha=0.9))

    # ==========================================
    # 2. FuelLevel = 0 & Spike ngắn (Car 1)
    # ==========================================
    df1 = pd.read_csv('data/processed/CarFuelHistory_Processed_Car1.csv')
    df1['FuelTime'] = pd.to_datetime(df1['FuelTime'])
    seg_1 = df1.head(60) 
    date_1 = seg_1['FuelTime'].iloc[0].strftime('%d/%m/%Y')
    
    ax2 = axs[0, 1]
    ax2.plot(seg_1['FuelTime'], seg_1['FuelLevel'], color='#f87171', marker='.', linestyle='-', linewidth=1.5)
    ax2.set_title(f"2. Lỗi rớt tín hiệu (FuelLevel = 0) & Spike ngắn\n[Dữ liệu: CAR 1 | {date_1}]", fontsize=15, fontweight='bold', color='#111827', pad=15)
    format_axis(ax2)
    
    zero_row = seg_1[seg_1['FuelLevel'] == 0].iloc[0]
    ax2.annotate('Bất thường: Báo 0 Lít đột ngột.\nCần phải lọc bỏ hoàn toàn!', 
                 xy=(zero_row['FuelTime'], zero_row['FuelLevel']),
                 xytext=(30, 50), textcoords='offset points',
                 arrowprops=dict(facecolor='#ef4444', shrink=0.05, width=2, headwidth=8),
                 fontsize=11, color='#b91c1c', ha='left', bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#ef4444", alpha=0.9))

    # ==========================================
    # 3. Gián đoạn dữ liệu (TimeGap) (Car 4)
    # ==========================================
    df4 = pd.read_csv('data/processed/CarFuelHistory_Processed_Car4.csv')
    df4['FuelTime'] = pd.to_datetime(df4['FuelTime'])
    
    gap_idx = df4['TimeGapMinutes'].idxmax()
    seg_gap = df4.iloc[gap_idx-15 : gap_idx+15]
    date_4 = seg_gap['FuelTime'].iloc[0].strftime('%d/%m/%Y')
    
    ax3 = axs[1, 0]
    ax3.plot(seg_gap['FuelTime'], seg_gap['FuelLevel'], color='#f87171', marker='o', linestyle='-', linewidth=1.5)
    ax3.set_title(f"3. Gián đoạn dữ liệu (Mất kết nối > 30 phút)\n[Dữ liệu: CAR 4 | {date_4}]", fontsize=15, fontweight='bold', color='#111827', pad=15)
    format_axis(ax3)
    
    p1 = df4.iloc[gap_idx-1]
    p2 = df4.iloc[gap_idx]
    ax3.axvspan(p1['FuelTime'], p2['FuelTime'], color='#e5e7eb', alpha=0.5)
    ax3.text(p1['FuelTime'] + (p2['FuelTime'] - p1['FuelTime'])/2, seg_gap['FuelLevel'].mean(), 
             'Khoảng trống không có dữ liệu\n(Do mất nguồn / Mất sóng)', ha='center', va='center', 
             bbox=dict(facecolor='white', alpha=0.9, edgecolor='#6b7280', boxstyle='round,pad=0.5'), 
             color='#4b5563', fontweight='bold', fontsize=11)

    # ==========================================
    # 4. Thay đổi mức kéo dài (Trend) (Car 5)
    # ==========================================
    seg_trend = df5.iloc[-250:-50] 
    date_trend = seg_trend['FuelTime'].iloc[0].strftime('%d/%m/%Y')
    
    ax4 = axs[1, 1]
    ax4.plot(seg_trend['FuelTime'], seg_trend['FuelLevel'], color='#f87171', marker='.', linestyle='-', alpha=0.7, linewidth=1.5)
    ax4.set_title(f"4. Thay đổi mức kéo dài (Tiêu hao thực tế)\n[Dữ liệu: CAR 5 | {date_trend}]", fontsize=15, fontweight='bold', color='#111827', pad=15)
    format_axis(ax4)
    
    mid_idx = len(seg_trend) // 2
    # Đẩy text xuống dưới để không chạm Title
    ax4.annotate('Xăng tiêu hao dần đều (Trend thật).\nBài toán: Bộ lọc tương lai không được\nvát phẳng đường này!', 
                 xy=(seg_trend.iloc[mid_idx]['FuelTime'], seg_trend.iloc[mid_idx]['FuelLevel']),
                 xytext=(20, -50), textcoords='offset points',
                 arrowprops=dict(facecolor='#3b82f6', shrink=0.05, width=2, headwidth=8),
                 fontsize=11, color='#1d4ed8', ha='left', bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#3b82f6", alpha=0.9))

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    output_path = 'slide6_noise_analysis.png'
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    print(f"-> Đã lưu ảnh thành công: {output_path}")

if __name__ == "__main__":
    create_slide6_infographic()
