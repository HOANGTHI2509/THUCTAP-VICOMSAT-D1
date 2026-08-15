import pandas as pd
import numpy as np
import glob
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.core.filters.kalman_smoother import RTSSmoother

sys.stdout.reconfigure(encoding='utf-8')

def build_zeroshot_dataset(window_size=30):
    input_files = glob.glob("data/processed/CarFuelHistory_Processed_*.csv")
    input_files = [f for f in input_files if "_CNN1D" not in f]
    
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
            
        for col in ['Speed', 'Acceleration', 'RollingStd', 'FuelLevel']:
            if col in df.columns:
                df[col] = df[col].fillna(0)
                
        df = df.sort_values(['FuelTime'])
        
        vehicle_id = os.path.basename(file_path).split('_')[-1].replace('.csv', '')
        
        # CHIA SPLIT THEO ZERO-SHOT
        seg_stats = df.groupby('SegmentID').size().reset_index(name='count')
        seg_times = df.groupby('SegmentID')['FuelTime'].min().reset_index(name='start_time')
        seg_stats = seg_stats.merge(seg_times, on='SegmentID').sort_values('start_time')
        segment_ids = seg_stats['SegmentID'].tolist()
        
        seg_split_map = {}
        if vehicle_id == 'Car5':
            # Car 5 hoàn toàn vào Test
            for seg_id in segment_ids:
                seg_split_map[seg_id] = 'Test'
        else:
            # Car 1-4 chia Train (85%) / Val (15%) theo segment
            segment_lengths = seg_stats['count'].tolist()
            k = len(segment_ids)
            total_points = sum(segment_lengths)
            
            target_train = 0.85 * total_points
            
            best_loss = float('inf')
            best_i = 0
            
            for i in range(k + 1):
                train_pts = sum(segment_lengths[0:i])
                loss = abs(train_pts - target_train)
                
                if loss < best_loss:
                    best_loss = loss
                    best_i = i
                    
            for idx, seg_id in enumerate(segment_ids):
                if idx < best_i:
                    seg_split_map[seg_id] = 'Train'
                else:
                    seg_split_map[seg_id] = 'Val'
                
        df['Split'] = df['SegmentID'].map(seg_split_map)
        
        # Windowing
        for seg_id, group in df.groupby('SegmentID'):
            group = group.sort_values('FuelTime').reset_index(drop=True)
            split_type = group['Split'].iloc[0]
            n_points = len(group)
            
            if n_points < window_size:
                total_short_segments_skipped += 1
                continue
            
            raw_fuel = group['FuelLevel'].values
            speed = group['Speed'].values
            accel = group['Acceleration'].values
            movement = group['MovementState_Num'].values
            roll_std = group['RollingStd'].values
            fuel_times = pd.to_datetime(group['FuelTime']).values
            
            # RTS Offline Reference for Target Y
            smoother = RTSSmoother(process_noise_q=0.1, measurement_noise_r=10.0)
            reference_signal = smoother.smooth(raw_fuel)
            
            for i in range(window_size - 1, n_points):
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
                
                window_residual = window_fuel - anchor
                
                X_window = np.column_stack([
                    window_residual,
                    window_speed,
                    window_accel,
                    window_move,
                    window_std
                ])
                
                y_target = reference_signal[i] - anchor
                
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
                    'ReferenceSignal_t': reference_signal[i]
                }
                
                if split_type == 'Train':
                    X_train_list.append(X_window)
                    y_train_list.append(y_target)
                    M_train_list.append(meta_row)
                elif split_type == 'Val':
                    X_val_list.append(X_window)
                    y_val_list.append(y_target)
                    M_val_list.append(meta_row)
                else:
                    X_test_list.append(X_window)
                    y_test_list.append(y_target)
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
    print("BÁO CÁO KẾT QUẢ PHASE 5.4: ZERO-SHOT DATASET")
    print("="*50)
    print(f"Tổng số Short Segments bị skip: {total_short_segments_skipped}")
    print(f"Tổng số Windows: {total_windows_created}")
    print(f" - Train Shape (Car1-4): X={X_train.shape}, y={y_train.shape}, M={M_train.shape}")
    print(f" - Val Shape   (Car1-4): X={X_val.shape}, y={y_val.shape}, M={M_val.shape}")
    print(f" - Test Shape  (Car5): X={X_test.shape}, y={y_test.shape}, M={M_test.shape}")
    print("="*50)
    
    os.makedirs('data/zeroshot_dataset', exist_ok=True)
    np.save('data/zeroshot_dataset/X_train.npy', X_train)
    np.save('data/zeroshot_dataset/X_val.npy', X_val)
    np.save('data/zeroshot_dataset/X_test.npy', X_test)
    np.save('data/zeroshot_dataset/y_train.npy', y_train)
    np.save('data/zeroshot_dataset/y_val.npy', y_val)
    np.save('data/zeroshot_dataset/y_test.npy', y_test)
    
    M_train.to_pickle('data/zeroshot_dataset/M_train.pkl')
    M_val.to_pickle('data/zeroshot_dataset/M_val.pkl')
    M_test.to_pickle('data/zeroshot_dataset/M_test.pkl')
    
    print("Hoàn tất lưu trữ X, y và Metadata vào thư mục data/zeroshot_dataset/")

if __name__ == "__main__":
    build_zeroshot_dataset(window_size=30)
