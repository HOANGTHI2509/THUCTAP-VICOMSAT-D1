import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import pandas as pd
import numpy as np
import sys
sys.stdout.reconfigure(encoding='utf-8')
import os

print("=== BẮT ĐẦU QUY TRÌNH TIỀN XỬ LÝ DỮ LIỆU ===")

# 1. Đầu vào là nhiều file xls
if __name__ == '__main__':
    import glob
    files = glob.glob("DU_lieu_VICOMSAT/*.xls")
    
    for input_file in files:
        print(f"\n--- Đang xử lý file: {input_file} ---")
        try:
            df = pd.read_excel(input_file, header=12)
            # Dòng 0 và 1 dưới header là (No.), (Cars) và (1), (2). Ta cần bỏ đi.
            df = df.iloc[2:].reset_index(drop=True)
        except Exception as e:
            print(f"Lỗi: {e}")
            continue
            
        # Đổi tên cột
        df.rename(columns=lambda x: str(x).strip(), inplace=True)
        col_map = {
            'Biển số': 'VehicleID',
            'Thời gian': 'FuelTime',
            'Nhiên liệu': 'FuelLevel',
            'Vận tốc': 'Speed',
            'Địa điểm': 'Address'
        }
        df.rename(columns=col_map, inplace=True)
        
        if df.empty or 'VehicleID' not in df.columns:
            print("File rỗng hoặc sai cấu trúc.")
            continue
            
        vehicle_id = str(df['VehicleID'].iloc[0]).strip()
        print(f"Nhận diện xe: {vehicle_id}")
        
        # 3. Chuẩn hóa FuelTime và kiểu số
        df['FuelTime'] = pd.to_datetime(df['FuelTime'], dayfirst=True, errors='coerce')
        for col in ['FuelLevel', 'Lat', 'Lng', 'Speed']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # 4. Sắp xếp theo thời gian
        df = df.sort_values(['FuelTime']).reset_index(drop=True)
        
        # 5. Kiểm tra hợp lệ và gắn cờ cơ bản
        df['FlagFuelZero'] = (df['FuelLevel'] == 0).astype(int)
        df['FlagDuplicateTime'] = df.duplicated(subset=['FuelTime'], keep=False).astype(int)
        
        # 6. Đặt điểm không hợp lệ thành NaN trong bản dùng tính toán
        # Điều này ngăn việc tính Delta giả khi chuyển từ 0 sang số bình thường
        df['FuelLevel_calc'] = df['FuelLevel'].where(df['FlagFuelZero'] == 0, np.nan)
        
        # 7. Tính TimeGap và chia SegmentID
        df['TimeGapMinutes'] = df['FuelTime'].diff().dt.total_seconds() / 60
        
        # Tính chu kỳ chuẩn cục bộ bằng Rolling Median (N=20), k=3
        k = 3
        baseline_local = df['TimeGapMinutes'].rolling(window=20, min_periods=1).median()
        df['DynamicGapThreshold'] = np.maximum(baseline_local * k, 5.0)
        
        df['FlagLongGap'] = (df['TimeGapMinutes'] > df['DynamicGapThreshold']).astype(int)
        
        # Tính Haversine Distance nếu có Lat/Lng
        if 'Lat' in df.columns and 'Lng' in df.columns:
            lat1 = np.radians(df['Lat'].shift(1))
            lon1 = np.radians(df['Lng'].shift(1))
            lat2 = np.radians(df['Lat'])
            lon2 = np.radians(df['Lng'])
            
            dlon = lon2 - lon1
            dlat = lat2 - lat1
            a = np.sin(dlat / 2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0)**2
            # Tránh lỗi domain của arcsin do sai số phẩy động
            a = np.clip(a, 0, 1)
            c = 2 * np.arcsin(np.sqrt(a))
            r = 6371000 # Bán kính trái đất (mét)
            df['DistanceMeters'] = c * r
        else:
            df['DistanceMeters'] = 0.0
    
        # Điều kiện 3: Kiểm tra đỗ xe thật sự (Dùng cho cờ Suspicious Gap)
        prev_stopped = df['Speed'].shift(1) <= 5
        curr_stopped = df['Speed'] <= 5
        
        is_true_parking = (df['TimeGapMinutes'] > df['DynamicGapThreshold']) & prev_stopped & curr_stopped & (df['DistanceMeters'].fillna(0) <= 50)
        
        # Nhóm sử dụng 2 giờ (120 phút) làm ngưỡng hard-cut ban đầu. Các khoảng mất 
        # tín hiệu ngắn hơn được phát hiện bằng ngưỡng động (DynamicGapThreshold)
        # dựa trên chu kỳ truyền dữ liệu của từng xe và được gắn cờ (FlagLongGap) 
        # thay vì cắt Segment. Sau đó nhóm sẽ đánh giá độ nhạy của ngưỡng hard-cut 
        # để xác định giá trị phù hợp.
        is_new_segment = (df['TimeGapMinutes'] >= 120) | df['TimeGapMinutes'].isna()
        df['SegmentID'] = is_new_segment.cumsum()
        df['IsSegmentStart'] = is_new_segment.astype(int)
        
        # Gắn cờ Suspicious Gap cho những trường hợp mất sóng khi xe đang di chuyển
        signal_loss_moving = (df['TimeGapMinutes'] > df['DynamicGapThreshold']) & (~is_true_parking)
        df['FlagSuspiciousGap'] = signal_loss_moving.astype(int)
        
        # FeatureStatus: Đánh dấu nếu bản ghi trước đó không hợp lệ (làm DeltaFuel = NaN)
        df['FeatureStatus'] = 'VALID'
        df.loc[df['FlagFuelZero'] == 1, 'FeatureStatus'] = 'FUEL_ZERO'
        
        # Nếu điểm trước đó là FUEL_ZERO (hoặc không hợp lệ), điểm hiện tại sẽ bị ảnh hưởng Delta
        prev_invalid = df['FlagFuelZero'].shift(1) == 1
        # Chỉ gán PREVIOUS_INVALID cho các điểm đang VALID nhưng bị ảnh hưởng bởi điểm trước
        df.loc[(df['FeatureStatus'] == 'VALID') & prev_invalid, 'FeatureStatus'] = 'PREVIOUS_INVALID'
        
        # 8. Tính DeltaFuel và FuelRate trong từng segment
        # Vì dùng FuelLevel_calc, điểm đầu tiên sau điểm 0 cũng sẽ có PreviousFuel = NaN -> Delta = NaN
        df['DeltaFuel'] = df.groupby('SegmentID')['FuelLevel_calc'].diff()
        
        safe_gap = df['TimeGapMinutes'].replace(0, np.nan)
        df['FuelRate'] = df['DeltaFuel'] / safe_gap
        
        # 9. Tính RollingMedian, RollingStd trên mẫu hợp lệ
        # Dùng FuelLevel_calc để không bị số 0 kéo kết quả xuống
        df['RollingMedian'] = df.groupby('SegmentID')['FuelLevel_calc'].transform(
            lambda x: x.rolling(window=5, min_periods=1).median())
        df['RollingStd'] = df.groupby('SegmentID')['FuelLevel_calc'].transform(
            lambda x: x.rolling(window=5, min_periods=1).std()) # Bỏ fillna(0) để giữ NaN tự nhiên
            
        # Thêm RollingCount và RollingWindowReady
        df['RollingCount'] = df.groupby('SegmentID')['FuelLevel_calc'].transform(
            lambda x: x.rolling(window=5, min_periods=1).count())
        df['RollingWindowReady'] = (df['RollingCount'] >= 3).astype(int) # Sẵn sàng nếu có từ 3 mẫu trở lên
            
        # 10. Tạo MovementState và các cờ riêng
        if 'Speed' in df.columns:
            df['MovementState'] = np.where(df['Speed'] > 0, 'Moving', 'Stopped')
            # Gán nhãn cho các trường hợp thiếu speed hoặc không hợp lệ
            df.loc[df['Speed'].isna(), 'MovementState'] = 'Uncertain'
        else:
            df['MovementState'] = 'Uncertain'
            
        # Xác định FlagLargeDelta dựa trên P99 của DeltaFuel (Tránh áp ngưỡng cứng)
        if not df['DeltaFuel'].isna().all():
            threshold_large = df['DeltaFuel'].abs().quantile(0.99)
            if pd.isna(threshold_large) or threshold_large < 5:
                threshold_large = 10 # Ngưỡng tối thiểu
        else:
            threshold_large = 10
        df['FlagLargeDelta'] = (df['DeltaFuel'].abs() > threshold_large).astype(int)
        
        # 11. Tổng hợp QualityFlag & QualityReason
        flags = {
            'FUEL_ZERO': df['FlagFuelZero'] == 1,
            'LONG_GAP': df['FlagLongGap'] == 1,
            'LARGE_DELTA': df['FlagLargeDelta'] == 1,
            'DUPLICATE_TIME': df['FlagDuplicateTime'] == 1,
            'SIGNAL_LOSS_MOVING': df['FlagSuspiciousGap'] == 1,
        }
        
        df['QualityFlag'] = 0
        reasons = pd.Series("", index=df.index)
        
        for reason, condition in flags.items():
            df.loc[condition, 'QualityFlag'] = 1
            # Nối chuỗi lý do
            reasons = np.where(condition, 
                               np.where(reasons == "", reason, reasons + "|" + reason), 
                               reasons)
                               
        df['QualityReason'] = np.where(reasons == "", "VALID", reasons)
        print(f"Ngưỡng FlagLargeDelta tự động (Adaptive threshold) của xe này là: {threshold_large:.2f} Lít")
        
        # 12. Lưu file Output (Chỉ giữ các cột phân tích)
        cols_to_keep = [
            'VehicleID', 'FuelTime', 'FuelLevel', 'Speed', 'Lat', 'Lng', 'Address', 'DistanceMeters', 'TimeGapMinutes', 'SegmentID', 'IsSegmentStart',
            'DeltaFuel', 'FuelRate', 'RollingMedian', 'RollingStd', 'RollingCount',
            'RollingWindowReady', 'MovementState', 'FlagFuelZero', 'FlagLongGap', 'FlagLargeDelta', 'FlagSuspiciousGap', 'QualityFlag', 'QualityReason', 'FeatureStatus'
        ]
        # Lọc những cột có tồn tại
        cols_to_keep = [col for col in cols_to_keep if col in df.columns]
        df_out = df[cols_to_keep]
        
        # Lưu kết quả
        sanitized_id = "".join(c for c in str(vehicle_id) if c.isalnum() or c in ('_', '-')).strip()
        output_file = f"data/processed/CarFuelHistory_Processed_{sanitized_id}.csv"
        print(f"Đang lưu dữ liệu đã xử lý ra {output_file}...")
        df_out.to_csv(output_file, index=False)
        
        # In dữ liệu mẫu
        print(df_out.head(3).to_string())
    
    print("\n=== HOÀN TẤT QUY TRÌNH ===")
