import pandas as pd
import numpy as np
import glob
import os

def calculate_empirical_delay():
    files = glob.glob('data/processed/CarFuelHistory_Processed_*_CNN_Realtime.csv')
    if not files:
        print("Waiting for CNN prediction to finish...")
        return
        
    lags = []
    
    for f in files:
        df = pd.read_csv(f)
        if 'CNN_Realtime' not in df.columns or 'CNN_EMA03' not in df.columns:
            continue
            
        # We find events where CNN_Realtime drops or jumps by more than 5 liters
        diff_cnn = df['CNN_Realtime'].diff().fillna(0)
        
        # Cross correlation method for delay
        for segment, group in df.groupby('SegmentID'):
            if len(group) < 20: continue
            
            cnn_vals = group['CNN_Realtime'].values
            ema_vals = group['CNN_EMA03'].values
            
            # Identify peaks in CNN_Realtime (sudden changes)
            diffs = np.abs(np.diff(cnn_vals))
            event_indices = np.where(diffs > 10.0)[0]
            
            for idx in event_indices:
                # Target value after the jump
                target_val = cnn_vals[idx + 1]
                start_val = cnn_vals[idx]
                jump = target_val - start_val
                
                # Look forward to see when EMA reaches 80% of the jump
                threshold = start_val + 0.8 * jump
                
                for k in range(1, min(10, len(ema_vals) - idx - 1)):
                    current_ema = ema_vals[idx + k]
                    if (jump > 0 and current_ema >= threshold) or (jump < 0 and current_ema <= threshold):
                        # Calculate time diff
                        t1 = pd.to_datetime(group.iloc[idx]['FuelTime'])
                        t2 = pd.to_datetime(group.iloc[idx + k]['FuelTime'])
                        mins = (t2 - t1).total_seconds() / 60.0
                        lags.append((k, mins))
                        break
                        
    if not lags:
        print("No valid events found to calculate delay.")
        return
        
    steps = [x[0] for x in lags]
    mins = [x[1] for x in lags]
    
    print(f"=== ĐO LƯỜNG ĐỘ TRỄ THỰC TẾ TRÊN DỮ LIỆU ===")
    print(f"Số lượng sự kiện đột biến (>10L) được phân tích: {len(lags)}")
    print(f"Thời gian trung bình để EMA đạt 80% giá trị sự kiện:")
    print(f"-> {np.mean(steps):.1f} bước (steps)")
    print(f"-> {np.mean(mins):.1f} phút (minutes)")
    
    print("\n=== MÔ PHỎNG LÝ THUYẾT ===")
    alpha = 0.3
    ema = 0.0
    for t in range(1, 10):
        ema = alpha * 100 + (1 - alpha) * ema
        print(f"Bước {t}: {ema:.1f}%")

if __name__ == '__main__':
    calculate_empirical_delay()
