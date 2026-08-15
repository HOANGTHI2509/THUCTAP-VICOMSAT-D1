import pandas as pd
import numpy as np
import os
from tqdm import tqdm

def process_timegap(tg):
    # Clip large timegaps to a reasonable threshold (e.g., 15 mins)
    return np.clip(tg, 0.0, 15.0)

def build_windows(df, window_size, is_synthetic=True):
    # Features: FuelResidual (NoisyFuel - Anchor), Speed, Acceleration, MovementState, RollingStd, TimeGap
    # Output: y = CleanFuel(t) - Anchor
    
    # We group by SegmentID so windows do NOT cross large time gaps or different segments
    X_list = []
    y_list = []
    
    # Pre-encode MovementState
    df['MovEncoded'] = (df['MovementState'] == 'Moving').astype(float)
    df['TimeGapClipped'] = process_timegap(df['TimeGapMinutes'])
    
    for segment_id, group in tqdm(df.groupby('SegmentID')):
        if len(group) < window_size:
            continue
            
        # Extract columns as numpy arrays for speed
        noisy_fuel = group['NoisyFuel'].values
        speed = group['Speed'].values
        accel = group['Acceleration'].values
        mov = group['MovEncoded'].values
        rstd = group['RollingStd'].values
        tg = group['TimeGapClipped'].values
        
        if is_synthetic:
            clean_fuel = group['CleanFuel'].values
            event_labels = group['EventLabel'].values
            
        n_points = len(group)
        for i in range(window_size - 1, n_points):
            # Window slice
            w_noisy = noisy_fuel[i - window_size + 1 : i + 1]
            w_speed = speed[i - window_size + 1 : i + 1]
            w_accel = accel[i - window_size + 1 : i + 1]
            w_mov = mov[i - window_size + 1 : i + 1]
            w_rstd = rstd[i - window_size + 1 : i + 1]
            w_tg = tg[i - window_size + 1 : i + 1]
            
            anchor = w_noisy[0]
            w_fuel_res = w_noisy - anchor
            
            # Combine into (WindowSize, 6)
            window_features = np.column_stack((w_fuel_res, w_speed, w_accel, w_mov, w_rstd, w_tg))
            X_list.append(window_features)
            
            if is_synthetic:
                target_residual = clean_fuel[i] - anchor
                y_list.append(target_residual)
                
                # Check event label at the target timestep (i)
                label = event_labels[i]
                mask_val = 0.0 if label in ['Refuel', 'Drain'] else 1.0
                if 'mask_list' not in locals(): mask_list = []
                mask_list.append(mask_val)
                
    X = np.array(X_list, dtype=np.float32)
    if is_synthetic:
        y = np.array(y_list, dtype=np.float32).reshape(-1, 1)
        mask = np.array(mask_list, dtype=np.float32).reshape(-1, 1)
        return X, y, mask
    return X

def generate_datasets_for_all_N():
    print("Loading synthetic dataset...")
    df = pd.read_csv('data/gru_dataset/synthetic_dataset.csv')
    
    # We will split train/val/test by SegmentID
    # Since they are synthetic, we can just take the last 20% segments as test
    segments = df['SegmentID'].unique()
    np.random.seed(42)
    np.random.shuffle(segments)
    
    n_train = int(len(segments) * 0.7)
    n_val = int(len(segments) * 0.15)
    
    train_segs = set(segments[:n_train])
    val_segs = set(segments[n_train:n_train+n_val])
    test_segs = set(segments[n_train+n_val:])
    
    df_train = df[df['SegmentID'].isin(train_segs)]
    df_val = df[df['SegmentID'].isin(val_segs)]
    df_test = df[df['SegmentID'].isin(test_segs)]
    
    windows_to_test = [10, 20, 30, 45, 60]
    out_dir = 'data/gru_dataset/windows'
    os.makedirs(out_dir, exist_ok=True)
    
    for N in windows_to_test:
        print(f"\nBuilding dataset for N={N}...")
        X_train, y_train, mask_train = build_windows(df_train, N)
        X_val, y_val, mask_val = build_windows(df_val, N)
        X_test, y_test, mask_test = build_windows(df_test, N)
        
        np.save(os.path.join(out_dir, f'X_train_N{N}.npy'), X_train)
        np.save(os.path.join(out_dir, f'y_train_N{N}.npy'), y_train)
        np.save(os.path.join(out_dir, f'mask_train_N{N}.npy'), mask_train)
        np.save(os.path.join(out_dir, f'X_val_N{N}.npy'), X_val)
        np.save(os.path.join(out_dir, f'y_val_N{N}.npy'), y_val)
        np.save(os.path.join(out_dir, f'mask_val_N{N}.npy'), mask_val)
        np.save(os.path.join(out_dir, f'X_test_N{N}.npy'), X_test)
        np.save(os.path.join(out_dir, f'y_test_N{N}.npy'), y_test)
        np.save(os.path.join(out_dir, f'mask_test_N{N}.npy'), mask_test)
        
        print(f"N={N} -> Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

if __name__ == '__main__':
    generate_datasets_for_all_N()
