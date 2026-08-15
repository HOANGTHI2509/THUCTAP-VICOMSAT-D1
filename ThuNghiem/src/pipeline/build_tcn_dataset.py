import pandas as pd
import numpy as np
import glob
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.core.filters.kalman_smoother import RTSSmoother

sys.stdout.reconfigure(encoding='utf-8')

def build_dataset(window_size=30):
    input_files = glob.glob("data/processed/CarFuelHistory_Processed_*.csv")
    input_files = [f for f in input_files if "_CNN1D" not in f and "Car5" not in f]
    
    if not input_files:
        print("Không tìm thấy dữ liệu đầu vào. Vui lòng kiểm tra thư mục data/processed/")
        return
        
    X_train_list = []
    X_val_list = []
    X_test_list = []
    
    y_train_list = []
    y_val_list = []
    y_test_list = []
    
    M_train_list = []
    M_val_list = []
    M_test_list = []
    
    total_short_segments_skipped = 0
    total_windows_created = 0
    
    for file_path in input_files:
        print(f"Đang xử lý: {file_path}")
        df = pd.read_csv(file_path)
        df['FuelTime'] = pd.to_datetime(df['FuelTime'])
        df = df.sort_values(['SegmentID', 'FuelTime'])
        
        # Tiền xử lý các cột
        if 'MovementState' in df.columns:
            move_map = {'Stopped': 0.0, 'Moving': 1.0, 'Uncertain': 0.5}
            df['MovementState_Num'] = df['MovementState'].map(move_map).fillna(0.5)
        else:
            df['MovementState_Num'] = 0.5
            
        # Đảm bảo các cột không có NaN
        for col in ['Speed', 'Acceleration', 'RollingStd', 'FuelLevel']:
            if col in df.columns:
                df[col] = df[col].fillna(0)
                
        # Time-based split cho TỪNG XE theo SEGMENT, KHÔNG CẮT SEGMENT LÀM ĐÔI
        df = df.sort_values(['FuelTime'])
        
        # Lấy thông tin thống kê của các Segment
        seg_stats = df.groupby('SegmentID').size().reset_index(name='count')
        seg_times = df.groupby('SegmentID')['FuelTime'].min().reset_index(name='start_time')
        seg_stats = seg_stats.merge(seg_times, on='SegmentID').sort_values('start_time')
        
        segment_ids = seg_stats['SegmentID'].tolist()
        segment_lengths = seg_stats['count'].tolist()
        k = len(segment_ids)
        total_points = sum(segment_lengths)
        
        target_train = 0.7 * total_points
        target_val = 0.15 * total_points
        target_test = 0.15 * total_points
        
        best_loss = float('inf')
        best_i, best_j = 0, 0
        
        # Duyệt qua tất cả các cặp (i, j) với 0 <= i <= j <= k
        for i in range(k + 1):
            for j in range(i, k + 1):
                train_pts = sum(segment_lengths[0:i])
                val_pts = sum(segment_lengths[i:j])
                test_pts = sum(segment_lengths[j:k])
                
                loss = abs(train_pts - target_train) + abs(val_pts - target_val) + abs(test_pts - target_test)
                if val_pts == 0:
                    loss += total_points * 10
                if test_pts == 0:
                    loss += total_points * 10
                    
                if loss < best_loss:
                    best_loss = loss
                    best_i = i
                    best_j = j
                    
        seg_split_map = {}
        for idx, seg_id in enumerate(segment_ids):
            if idx < best_i:
                seg_split_map[seg_id] = 'Train'
            elif idx < best_j:
                seg_split_map[seg_id] = 'Val'
            else:
                seg_split_map[seg_id] = 'Test'
                
        df['Split'] = df['SegmentID'].map(seg_split_map)
        
        # Duyệt qua từng Segment (Split đã đảm bảo 1 Segment chỉ thuộc 1 Split)
        for seg_id, group in df.groupby('SegmentID'):
            group = group.sort_values('FuelTime').reset_index(drop=True)
            split_type = group['Split'].iloc[0]
            n_points = len(group)
            
            # Short Segment hoặc Chunk bị cắt quá nhỏ: Không tạo window
            if n_points < window_size:
                total_short_segments_skipped += 1
                continue
            
            
            raw_fuel = group['FuelLevel'].values
            speed = group['Speed'].values
            accel = group['Acceleration'].values
            movement = group['MovementState_Num'].values
            roll_std = group['RollingStd'].values
            fuel_times = pd.to_datetime(group['FuelTime']).values
            
            # Extract VehicleID from filename
            vehicle_id = os.path.basename(file_path).split('_')[-1].replace('.csv', '')
            
            # Trích xuất dữ liệu từ các cột đã được sinh ở Phase 5.5-C (generate_y_ep.py)
            y_std_signal = group['y_std'].values
            y_ep_signal = group['y_ep'].values
            event_weight_signal = group.get('EventWeight', np.ones(n_points)).values
            
            for i in range(window_size - 1, n_points):
                # Anchor: Trailing Median (Không bao gồm điểm hiện tại t)
                past_fuel = raw_fuel[i - window_size + 1 : i]
                if len(past_fuel) > 0:
                    anchor = np.median(past_fuel)
                else:
                    anchor = raw_fuel[i]
                    
                window_fuel = raw_fuel[i - window_size + 1 : i + 1]
                window_speed = speed[i - window_size + 1 : i + 1]
                window_accel = accel[i - window_size + 1 : i + 1]
                window_move = movement[i - window_size + 1 : i + 1]
                window_std = roll_std[i - window_size + 1 : i + 1]
                
                # Fuel Residual
                window_residual = window_fuel - anchor
                
                X_window = np.column_stack([
                    window_residual,
                    window_speed,
                    window_accel,
                    window_move,
                    window_std
                ])
                
                # Targets y
                y_std_target = y_std_signal[i] - anchor
                y_ep_target = y_ep_signal[i] - anchor
                
                # Weight
                weight = event_weight_signal[i]
                
                # Tích hợp vào Target Y dạng mảng 3 chiều: [y_std, y_ep, weight]
                y_target_array = np.array([y_std_target, y_ep_target, weight], dtype=np.float32)
                
                # METADATA
                start_time = fuel_times[i - window_size + 1]
                end_time = fuel_times[i]
                duration_mins = (end_time - start_time) / np.timedelta64(1, 'm')
                
                meta_row = {
                    'VehicleID': vehicle_id,
                    'SegmentID': seg_id,
                    'Split': split_type,
                    'StartTime': start_time,
                    'EndTime': end_time,
                    'TimeDurationMins': duration_mins,
                    'Anchor': anchor,
                    'RawFuel_t': raw_fuel[i],
                    'y_std_t': y_std_signal[i],
                    'y_ep_t': y_ep_signal[i],
                    'Weight_t': weight
                }
                
                if split_type == 'Train':
                    X_train_list.append(X_window)
                    y_train_list.append(y_target_array)
                    M_train_list.append(meta_row)
                elif split_type == 'Val':
                    X_val_list.append(X_window)
                    y_val_list.append(y_target_array)
                    M_val_list.append(meta_row)
                else:
                    X_test_list.append(X_window)
                    y_test_list.append(y_target_array)
                    M_test_list.append(meta_row)
                    
                total_windows_created += 1

    X_train = np.array(X_train_list) if X_train_list else np.empty((0, window_size, 5))
    X_val = np.array(X_val_list) if X_val_list else np.empty((0, window_size, 5))
    X_test = np.array(X_test_list) if X_test_list else np.empty((0, window_size, 5))
    
    y_train = np.array(y_train_list)
    y_val = np.array(y_val_list)
    y_test = np.array(y_test_list)
    
    M_train = pd.DataFrame(M_train_list)
    M_val = pd.DataFrame(M_val_list)
    M_test = pd.DataFrame(M_test_list)
    
    print("\n" + "="*50)
    print("BÁO CÁO KẾT QUẢ PHASE 3: SINH DATASET & TARGET Y")
    print("="*50)
    print(f"Tổng số Short Segments (< {window_size} điểm): {total_short_segments_skipped}")
    print(f"Tổng số Windows (X) được tạo ra: {total_windows_created}")
    print(f" - Train Shape: X={X_train.shape}, y={y_train.shape}, M={M_train.shape}")
    print(f" - Val Shape  : X={X_val.shape}, y={y_val.shape}, M={M_val.shape}")
    print(f" - Test Shape : X={X_test.shape}, y={y_test.shape}, M={M_test.shape}")
    print("="*50)
    
    os.makedirs('data/tcn_dataset', exist_ok=True)
    np.save('data/tcn_dataset/X_train.npy', X_train)
    np.save('data/tcn_dataset/X_val.npy', X_val)
    np.save('data/tcn_dataset/X_test.npy', X_test)
    np.save('data/tcn_dataset/y_train.npy', y_train)
    np.save('data/tcn_dataset/y_val.npy', y_val)
    np.save('data/tcn_dataset/y_test.npy', y_test)
    
    M_train.to_pickle('data/tcn_dataset/M_train.pkl')
    M_val.to_pickle('data/tcn_dataset/M_val.pkl')
    M_test.to_pickle('data/tcn_dataset/M_test.pkl')
    
    print("Hoàn tất lưu trữ X, y và Metadata vào thư mục data/tcn_dataset/")

if __name__ == "__main__":
    build_dataset(window_size=30)
