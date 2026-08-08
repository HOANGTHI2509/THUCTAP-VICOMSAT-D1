import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import sys

sys.stdout.reconfigure(encoding='utf-8')

def format_axis(ax):
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M\n%d/%m'))
    ax.tick_params(axis='x', rotation=0, labelsize=12)
    ax.tick_params(axis='y', labelsize=12)
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.set_ylabel("Mức Xăng (Lít)", fontsize=14, fontweight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

def create_split_images():
    print("Bắt đầu tạo các ảnh Zoom-in riêng lẻ cho Slide 9...")
    plt.rcParams['figure.dpi'] = 300
    plt.rcParams['font.family'] = 'sans-serif'
    
    # 1. Ảnh FuelLevel = 0 & DeltaFuel (Dùng Car 1 đoạn đầu)
    df1 = pd.read_csv('data/processed/CarFuelHistory_Processed_Car1.csv')
    df1['FuelTime'] = pd.to_datetime(df1['FuelTime'])
    
    fig1, ax1 = plt.subplots(figsize=(10, 5))
    seg1 = df1.head(10) # Rất ngắn để zoom cực đại
    ax1.plot(seg1['FuelTime'], seg1['FuelLevel'], color='#2563eb', marker='o', linestyle='-', linewidth=2)
    ax1.set_title("1. Minh họa QualityFlag & DeltaFuel", fontsize=16, fontweight='bold', pad=15)
    format_axis(ax1)
    
    zero_row = seg1.iloc[0]
    jump_row = seg1.iloc[1]
    
    ax1.annotate('QualityFlag = 1\n(Do FuelLevel = 0)', 
                xy=(zero_row['FuelTime'], zero_row['FuelLevel']),
                xytext=(30, 20), textcoords='offset points',
                arrowprops=dict(facecolor='#ef4444', shrink=0.05, width=2, headwidth=8),
                fontsize=13, color='#b91c1c', fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#ef4444", alpha=0.9))
                
    ax1.annotate('DeltaFuel & FuelRate cực lớn\n(Gai đột biến)', 
                xy=(jump_row['FuelTime'], jump_row['FuelLevel']),
                xytext=(30, -50), textcoords='offset points',
                arrowprops=dict(facecolor='#f59e0b', shrink=0.05, width=2, headwidth=8),
                fontsize=13, color='#b45309', fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#f59e0b", alpha=0.9))
                
    plt.tight_layout()
    fig1.savefig('anh/Slide9_1_quality_delta.png')
    plt.close(fig1)

    # 2. Ảnh TimeGap & SegmentID (Dùng Car 4)
    df4 = pd.read_csv('data/processed/CarFuelHistory_Processed_Car4.csv')
    df4['FuelTime'] = pd.to_datetime(df4['FuelTime'])
    gap_idx = df4['TimeGapMinutes'].idxmax()
    seg_gap = df4.iloc[gap_idx-5 : gap_idx+5]
    
    fig2, ax2 = plt.subplots(figsize=(10, 5))
    ax2.plot(seg_gap['FuelTime'], seg_gap['FuelLevel'], color='#2563eb', marker='o', linestyle='-', linewidth=2)
    ax2.set_title("2. Minh họa TimeGapMinutes & SegmentID", fontsize=16, fontweight='bold', pad=15)
    format_axis(ax2)
    
    p1 = df4.iloc[gap_idx-1]
    p2 = df4.iloc[gap_idx]
    ax2.axvspan(p1['FuelTime'], p2['FuelTime'], color='#fca5a5', alpha=0.4)
    mid_time = p1['FuelTime'] + (p2['FuelTime'] - p1['FuelTime'])/2
    ax2.annotate(f"TimeGap > 30 phút\nSinh ra SegmentID mới\n(Ngắt chuỗi dữ liệu)", 
                xy=(mid_time, seg_gap['FuelLevel'].mean()),
                xytext=(0, 40), textcoords='offset points',
                arrowprops=dict(facecolor='#8b5cf6', shrink=0, width=2, headwidth=8),
                fontsize=13, ha='center', color='#5b21b6', fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#8b5cf6", alpha=0.9))
    
    plt.tight_layout()
    fig2.savefig('anh/Slide9_2_timegap_segment.png')
    plt.close(fig2)

    # 3. Ảnh RollingStd (Dùng Car 5)
    df5 = pd.read_csv('data/processed/CarFuelHistory_Processed_Car5.csv')
    df5['FuelTime'] = pd.to_datetime(df5['FuelTime'])
    seg5 = df5.iloc[1050:1100]
    
    fig3, ax3 = plt.subplots(figsize=(10, 5))
    ax3.plot(seg5['FuelTime'], seg5['FuelLevel'], color='#2563eb', marker='.', linestyle='-', linewidth=1.5)
    ax3.set_title("3. Minh họa RollingStd (Độ nhiễu/Xóc nảy)", fontsize=16, fontweight='bold', pad=15)
    format_axis(ax3)
    
    std_idx = seg5['RollingStd'].idxmax()
    std_row = seg5.loc[std_idx]
    ax3.annotate(f"RollingStd tăng vọt\nĐại diện cho vùng dao động mạnh", 
                xy=(std_row['FuelTime'], std_row['FuelLevel']),
                xytext=(0, -50), textcoords='offset points', ha='center',
                arrowprops=dict(facecolor='#3b82f6', shrink=0.05, width=2, headwidth=8),
                fontsize=13, color='#1d4ed8', fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#3b82f6", alpha=0.9))
                
    plt.tight_layout()
    fig3.savefig('anh/Slide9_3_rollingstd.png')
    plt.close(fig3)

    # 4. Ảnh MovementState (Car 1 đoạn chuyển giao)
    seg_move = df1.iloc[500:580] # Có đoạn chạy và đỗ
    fig4, ax4 = plt.subplots(figsize=(10, 5))
    ax4.plot(seg_move['FuelTime'], seg_move['FuelLevel'], color='#2563eb', marker='.', linestyle='-', linewidth=1.5)
    ax4.set_title("4. Minh họa MovementState (Phân loại trạng thái)", fontsize=16, fontweight='bold', pad=15)
    format_axis(ax4)
    
    is_moving = seg_move['Speed'] > 0
    ax4.fill_between(seg_move['FuelTime'], 0, 250, where=is_moving, facecolor='#dcfce7', alpha=0.6, label='Moving (Chạy)')
    ax4.fill_between(seg_move['FuelTime'], 0, 250, where=~is_moving, facecolor='#f3f4f6', alpha=0.8, label='Stopped (Đỗ)')
    ax4.set_ylim(170, 200)
    
    ax4.annotate("Mảng Xanh: Trạng thái Moving\n(Xăng dao động mạnh)", 
                xy=(seg_move['FuelTime'].iloc[20], 190),
                fontsize=13, color='#166534', fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#166534", alpha=0.9))
                
    ax4.annotate("Mảng Xám: Trạng thái Stopped\n(Xăng phẳng lỳ)", 
                xy=(seg_move['FuelTime'].iloc[60], 195),
                fontsize=13, color='#4b5563', fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#4b5563", alpha=0.9))
    
    plt.tight_layout()
    fig4.savefig('anh/Slide9_4_movement.png')
    plt.close(fig4)

    print("Hoàn tất xuất 4 ảnh lẻ!")

if __name__ == "__main__":
    create_split_images()
