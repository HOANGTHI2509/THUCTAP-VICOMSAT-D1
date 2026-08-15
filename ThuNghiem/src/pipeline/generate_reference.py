import pandas as pd
import numpy as np
import glob
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.core.filters.kalman_smoother import RTSSmoother

def find_events_plateau(raw, window=10, min_jump=10):
    events = []
    n = len(raw)
    diffs = np.diff(raw)
    candidate_idxs = np.where(np.abs(diffs) > 3)[0]
    processed_idxs = set()
    
    for idx in candidate_idxs:
        if idx in processed_idxs: continue
        if idx < window or idx + 10 + window > n: continue
            
        end_idx = idx
        for j in range(idx + 1, min(idx + 15, n - window)):
            if np.abs(diffs[j-1]) < 1.0: 
                end_idx = j
                break
        else:
            end_idx = min(idx + 15, n - window)
            
        pre_plateau = raw[idx - window : idx]
        post_plateau = raw[end_idx : end_idx + window]
        
        pre_med = np.median(pre_plateau)
        post_med = np.median(post_plateau)
        true_jump = post_med - pre_med
        
        if np.abs(true_jump) > min_jump:
            ev_type = 'refuel' if true_jump > 0 else 'drain'
            
            future_window = min(n, end_idx + 30)
            if future_window > end_idx + window:
                far_med = np.median(raw[end_idx + 15 : future_window])
                if np.abs(far_med - pre_med) < min_jump / 2:
                    ev_type = 'false_positive_spike'
            
            events.append((idx, end_idx, ev_type, true_jump))
            for p in range(idx - 5, end_idx + window + 5):
                processed_idxs.add(p)
                
    return events

def generate_target_RB(raw, events):
    """
    Generate Reference Signal R-B (Split RTS + Step)
    """
    n = len(raw)
    smoother = RTSSmoother(process_noise_q=0.01, measurement_noise_r=100.0)
    target_B = np.zeros(n)
    
    last_idx = 0
    valid_events = [ev for ev in events if ev[2] != 'false_positive_spike']
    
    # Smooth normal zones independently
    for (s, e, ev_type, true_jump) in valid_events:
        if s > last_idx:
            chunk = raw[last_idx:s]
            if len(chunk) > 0:
                target_B[last_idx:s] = smoother.smooth(chunk)
        last_idx = e
        
    if last_idx < n:
        chunk = raw[last_idx:]
        if len(chunk) > 0:
            target_B[last_idx:] = smoother.smooth(chunk)
            
    # Fill event gaps with Step Function
    for (s, e, ev_type, true_jump) in valid_events:
        pre_val = target_B[s-1] if s > 0 else raw[s]
        post_val = target_B[e] if e < n else raw[-1]
        
        mid = (s + e) // 2
        for i in range(s, e):
            target_B[i] = pre_val if i < mid else post_val
                
    return target_B

def run():
    input_files = glob.glob("data/processed/CarFuelHistory_Processed_*.csv")
    input_files = [f for f in input_files if "_CNN1D" not in f and "Car5" not in f]
    
    smoother = RTSSmoother(process_noise_q=0.1, measurement_noise_r=10.0)
    
    total_events = 0
    false_positives = 0
    
    for f in input_files:
        print(f"Processing {f}...")
        df = pd.read_csv(f)
        df['FuelTime'] = pd.to_datetime(df['FuelTime'])
        df = df.sort_values(['SegmentID', 'FuelTime'])
        
        df['y_std'] = np.nan
        df['y_ep'] = np.nan
        df['EventMask'] = 0
        df['EventWeight'] = 1.0
        
        segments = []
        for seg_id, group in df.groupby('SegmentID'):
            raw_fuel = group['FuelLevel'].values
            if len(raw_fuel) < 100:
                group['y_std'] = raw_fuel
                group['y_ep'] = raw_fuel
                segments.append(group)
                continue
                
            # 1. Standard RTS (y_std)
            y_std = smoother.smooth(raw_fuel)
            
            # 2. Find events
            events = find_events_plateau(raw_fuel)
            
            # 3. New Target (y_ep = Target R-B)
            y_ep = generate_target_RB(raw_fuel, events)
            
            event_mask = np.zeros(len(raw_fuel), dtype=int)
            event_weight = np.ones(len(raw_fuel), dtype=float)
            
            for (start_idx, end_idx, ev_type, _) in events:
                if ev_type == 'false_positive_spike':
                    false_positives += 1
                    continue
                    
                total_events += 1
                
                # Assign weights
                w_start = max(0, start_idx - 10)
                event_weight[start_idx : end_idx + 1] = 5.0
                event_weight[end_idx + 1 : min(len(raw_fuel), end_idx + 11)] = 3.0
                
                event_mask[w_start : end_idx + 15] = 1
                
            group['y_std'] = y_std
            group['y_ep'] = y_ep
            group['EventMask'] = event_mask
            group['EventWeight'] = event_weight
            segments.append(group)
            
        df_out = pd.concat(segments)
        df_out.to_csv(f, index=False)
        
    print(f"Total True Events (Refuel/Drain): {total_events}")
    print(f"Total False Positive Spikes Rejected: {false_positives}")
    print("Successfully generated Reference Signal R-B (stored in 'y_ep' column).")
    
if __name__ == "__main__":
    run()
