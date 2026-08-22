import sys
import os
import glob
import pandas as pd
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
sys.stdout.reconfigure(encoding='utf-8')

print("=== BẮT ĐẦU QUY TRÌNH TIỀN XỬ LÝ DỮ LIỆU TỪ THƯ MỤC 'dulieu' ===")

def process_file(input_file, out_dir):
    print(f"\n--- Đang xử lý file: {input_file} ---")
    try:
        # Some xls files might be read properly by read_excel, but sometimes they are html tables saved as xls.
        # Assuming standard read_excel works as in the original pipeline.
        xl = pd.ExcelFile(input_file)
    except Exception as e:
        print(f"Lỗi đọc file {input_file}: {e}")
        return
        
    for sheet_name in xl.sheet_names:
        print(f"  -> Đang xử lý Sheet: {sheet_name}")
        df = xl.parse(sheet_name, header=12)
        
        # Đổi tên cột
        df.rename(columns=lambda x: str(x).strip(), inplace=True)
        col_map = {
            'Thời gian': 'FuelTime',
            'Nhiên liệu': 'FuelLevel',
            'Vận tốc': 'Speed',
            'Địa điểm': 'Address'
        }
        df.rename(columns=col_map, inplace=True)
        
        if df.empty:
            print(f"  -> Sheet {sheet_name} rỗng.")
            continue
            
        if 'Biển số' in df.columns and not df['Biển số'].dropna().empty:
            vehicle_id = str(df['Biển số'].dropna().mode()[0]).strip()
        else:
            vehicle_id = os.path.basename(input_file).replace('.xls', '').replace('.xlsx', '')
            
        df['VehicleID'] = vehicle_id
        
        # 3. Chuẩn hóa FuelTime và kiểu số
        df['FuelTime'] = pd.to_datetime(df['FuelTime'], dayfirst=True, errors='coerce')
        for col in ['FuelLevel', 'Lat', 'Lng', 'Speed']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Lọc NaN FuelTime
        df = df.dropna(subset=['FuelTime'])
        if df.empty:
            continue
            
        # 4. Sắp xếp theo thời gian
        df = df.sort_values(['FuelTime']).reset_index(drop=True)
        
        # 5. Kiểm tra hợp lệ và gắn cờ cơ bản
        df['FlagFuelZero'] = (df['FuelLevel'] == 0).astype(int)
        df['FlagDuplicateTime'] = df.duplicated(subset=['FuelTime'], keep=False).astype(int)
        
        # 6. Đặt điểm không hợp lệ thành NaN trong bản dùng tính toán
        df['FuelLevel_calc'] = df['FuelLevel'].where(df['FlagFuelZero'] == 0, np.nan)
        
        # 7. Tính TimeGap và chia SegmentID
        df['TimeGapMinutes'] = df['FuelTime'].diff().dt.total_seconds() / 60
        
        k = 3
        baseline_local = df['TimeGapMinutes'].rolling(window=20, min_periods=1).median()
        df['DynamicGapThreshold'] = np.maximum(baseline_local * k, 5.0)
        
        df['FlagLongGap'] = (df['TimeGapMinutes'] > df['DynamicGapThreshold']).astype(int)
        
        if 'Lat' in df.columns and 'Lng' in df.columns:
            lat1 = np.radians(df['Lat'].shift(1))
            lon1 = np.radians(df['Lng'].shift(1))
            lat2 = np.radians(df['Lat'])
            lon2 = np.radians(df['Lng'])
            
            dlon = lon2 - lon1
            dlat = lat2 - lat1
            a = np.sin(dlat / 2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0)**2
            a = np.clip(a, 0, 1)
            c = 2 * np.arcsin(np.sqrt(a))
            r = 6371000
            df['DistanceMeters'] = c * r
        else:
            df['DistanceMeters'] = 0.0

        prev_stopped = df['Speed'].shift(1) <= 5
        curr_stopped = df['Speed'] <= 5
        is_true_parking = (df['TimeGapMinutes'] > df['DynamicGapThreshold']) & prev_stopped & curr_stopped & (df['DistanceMeters'].fillna(0) <= 50)
        
        is_new_segment = (df['TimeGapMinutes'] >= 120) | df['TimeGapMinutes'].isna()
        df['SegmentID'] = is_new_segment.cumsum()
        df['IsSegmentStart'] = is_new_segment.astype(int)
        
        signal_loss_moving = (df['TimeGapMinutes'] > df['DynamicGapThreshold']) & (~is_true_parking)
        df['FlagSuspiciousGap'] = signal_loss_moving.astype(int)
        
        df['FeatureStatus'] = 'VALID'
        df.loc[df['FlagFuelZero'] == 1, 'FeatureStatus'] = 'FUEL_ZERO'
        prev_invalid = df['FlagFuelZero'].shift(1) == 1
        df.loc[(df['FeatureStatus'] == 'VALID') & prev_invalid, 'FeatureStatus'] = 'PREVIOUS_INVALID'
        
        # 8. Tính DeltaFuel
        df['DeltaFuel'] = df.groupby('SegmentID')['FuelLevel_calc'].diff()
        safe_gap = df['TimeGapMinutes'].replace(0, np.nan)
        df['FuelRate'] = df['DeltaFuel'] / safe_gap
        
        # 9. Tính RollingMedian, RollingStd
        df['RollingMedian'] = df.groupby('SegmentID')['FuelLevel_calc'].transform(
            lambda x: x.rolling(window=5, min_periods=1).median())
        df['RollingStd'] = df.groupby('SegmentID')['FuelLevel_calc'].transform(
            lambda x: x.rolling(window=5, min_periods=1).std())
            
        df['RollingCount'] = df.groupby('SegmentID')['FuelLevel_calc'].transform(
            lambda x: x.rolling(window=5, min_periods=1).count())
        df['RollingWindowReady'] = (df['RollingCount'] >= 3).astype(int)
            
        # 10. Tạo MovementState và Acceleration
        if 'Speed' in df.columns:
            df['MovementState'] = np.where(df['Speed'] > 0, 'Moving', 'Stopped')
            df.loc[df['Speed'].isna(), 'MovementState'] = 'Uncertain'
            safe_time_gap = df['TimeGapMinutes'].replace(0, 1.0) * 60.0
            df['Acceleration'] = df.groupby('SegmentID')['Speed'].diff().fillna(0.0) / safe_time_gap
        else:
            df['MovementState'] = 'Uncertain'
            df['Acceleration'] = 0.0
            
        if not df['DeltaFuel'].isna().all():
            threshold_large = df['DeltaFuel'].abs().quantile(0.99)
            if pd.isna(threshold_large) or threshold_large < 5:
                threshold_large = 10
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
            reasons = np.where(condition, 
                               np.where(reasons == "", reason, reasons + "|" + reason), 
                               reasons)
                               
        df['QualityReason'] = np.where(reasons == "", "VALID", reasons)
        
        # 12. Lưu file Output
        cols_to_keep = [
            'VehicleID', 'FuelTime', 'FuelLevel', 'Speed', 'Acceleration', 'Lat', 'Lng', 'Address', 'DistanceMeters', 'TimeGapMinutes', 'SegmentID', 'IsSegmentStart',
            'DeltaFuel', 'FuelRate', 'RollingMedian', 'RollingStd', 'RollingCount',
            'RollingWindowReady', 'MovementState', 'FlagFuelZero', 'FlagLongGap', 'FlagLargeDelta', 'FlagSuspiciousGap', 'QualityFlag', 'QualityReason', 'FeatureStatus'
        ]
        cols_to_keep = [col for col in cols_to_keep if col in df.columns]
        df_out = df[cols_to_keep]
        
        sanitized_id = "".join(c for c in str(vehicle_id) if c.isalnum() or c in ('_', '-')).strip()
        output_file = os.path.join(out_dir, f"{sanitized_id}_processed.csv")
        print(f"  -> Đang lưu dữ liệu đã xử lý ra {output_file}...")
        df_out.to_csv(output_file, index=False)

if __name__ == '__main__':
    input_dir = "D:/THUCTAP_VICOMSAT/dulieu"
    output_dir = "D:/THUCTAP_VICOMSAT/TienXuLy"
    
    os.makedirs(output_dir, exist_ok=True)
    
    xls_files = glob.glob(os.path.join(input_dir, "*.xls*"))
    print(f"Tìm thấy {len(xls_files)} files trong {input_dir}")
    
    for f in xls_files:
        process_file(f, output_dir)
        
    print("\n=== HOÀN TẤT QUY TRÌNH ===")
