import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import sys
sys.stdout.reconfigure(encoding='utf-8')
import pandas as pd
import numpy as np
import glob
import os

print("Bắt đầu trích xuất đặc trưng cho dự án VICOMSAT...")

csv_files = glob.glob(r"d:\THUCTAP_VICOMSAT\data\processed\CarFuelHistory_Processed_*.csv")
if not csv_files:
    print("Không tìm thấy file CarFuelHistory_Processed_*.csv nào trong data/processed.")
    sys.exit()

for f in csv_files:
    filename = os.path.basename(f)
    print(f"\nĐang xử lý {filename}...")
    
    df = pd.read_csv(f)
    
    # 1. Khôi phục định dạng thời gian và sắp xếp
    df['FuelTime'] = pd.to_datetime(df['FuelTime'])
    df = df.sort_values('FuelTime').reset_index(drop=True)
    
    # 2. Đặc trưng động học cơ bản
    df['DeltaFuel'] = df['FuelLevel'].diff().fillna(0)
    df['DeltaSpeed'] = df['Speed'].diff().fillna(0)
    df['MovementState'] = np.where(df['Speed'] > 0, 1, 0) # 1 = Chạy, 0 = Dừng
    
    # 3. Phân mảnh chuỗi thời gian (SegmentID)
    # Nếu TimeGapMinutes > 30 (tức là FlagLongGap = True), ta ngắt thành Segment mới
    if 'FlagLongGap' in df.columns:
        is_gap = df['FlagLongGap'] == True
    else:
        is_gap = df['TimeGapMinutes'] > 30.0
        
    df['SegmentID'] = is_gap.cumsum()
    
    # 4. Trích xuất đặc trưng chuỗi thời gian (Group theo Segment để tránh nối đoạn đứt gãy)
    def extract_group_features(segment):
        seg = segment.copy()
        
        # Tính gia tốc (tránh chia cho 0)
        time_gap_sec = seg['TimeGapMinutes'] * 60
        safe_time_gap = time_gap_sec.replace(0, 1)
        seg['Acceleration'] = seg['DeltaSpeed'] / safe_time_gap
        
        # Đặc trưng Rolling (Cửa sổ = 5)
        # Fuel_RollingStd giúp phát hiện nhiễu dạng sóng sánh
        if len(seg) >= 5:
            seg['Fuel_RollingMean_5'] = seg['FuelLevel'].rolling(5, min_periods=1).mean()
            seg['Fuel_RollingStd_5'] = seg['FuelLevel'].rolling(5, min_periods=1).std().fillna(0)
        else:
            seg['Fuel_RollingMean_5'] = seg['FuelLevel']
            seg['Fuel_RollingStd_5'] = 0.0
            
        return seg
        
    df = df.groupby('SegmentID', group_keys=False).apply(extract_group_features)
    
    # 5. Lưu ra file mới
    out_file = f.replace('_Processed_', '_').replace('.csv', '_Features.csv')
    df.to_csv(out_file, index=False)
    
    print(f"-> Đã lưu {len(df)} bản ghi vào {os.path.basename(out_file)}")
    
    if "Car 5" in filename:
        print("\n--- Mẫu dữ liệu trích xuất của CAR 5 ---")
        features = ['FuelTime', 'Speed', 'FuelLevel', 'TimeGapMinutes', 'SegmentID', 'DeltaFuel', 'MovementState', 'Fuel_RollingStd_5']
        print(df[features].head(5).to_string())

print("\nHoàn tất toàn bộ quy trình!")
