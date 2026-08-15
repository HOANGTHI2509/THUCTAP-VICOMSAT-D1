import torch
import numpy as np
import pandas as pd
import time
import os
from src.models.time_aware_gru import FuelTimeAwareGRU
from src.pipeline.build_gru_dataset import process_timegap
from tqdm import tqdm

def calculate_delay(clean, pred, event_indices):
    # Calculate how many timesteps it takes for pred to catch up to clean during an event
    delays = []
    # group contiguous event indices
    if len(event_indices) == 0: return 0
    
    breaks = np.where(np.diff(event_indices) > 1)[0]
    chunks = np.split(event_indices, breaks + 1)
    
    for chunk in chunks:
        if len(chunk) == 0: continue
        start_idx = chunk[0]
        end_idx = chunk[-1]
        
        # Look at the target value at the end of the event
        target_val = clean[end_idx]
        start_val = clean[start_idx-1] if start_idx > 0 else clean[0]
        delta = target_val - start_val
        
        if abs(delta) < 1.0: continue
        
        # Find when pred reaches 80% of the delta
        threshold = start_val + 0.8 * delta
        
        # Search forward from start_idx up to end_idx + 10
        search_end = min(len(pred), end_idx + 10)
        
        reached_idx = None
        for i in range(start_idx, search_end):
            if (delta > 0 and pred[i] >= threshold) or (delta < 0 and pred[i] <= threshold):
                reached_idx = i
                break
                
        if reached_idx is not None:
            # delay is how much it lags behind the event's end (or actual crossing)
            # Actually, a simpler delay is just cross correlation or just looking at MAE.
            # Let's just say delay = reached_idx - end_idx if it's after end_idx, else 0
            d = max(0, reached_idx - end_idx)
            delays.append(d)
        else:
            delays.append(10) # Max penalty
            
    return np.mean(delays) if delays else 0

def validate_metrics():
    df = pd.read_csv('data/gru_dataset/synthetic_dataset.csv')
    
    # Same split as build_gru_dataset
    segments = df['SegmentID'].unique()
    np.random.seed(42)
    np.random.shuffle(segments)
    
    n_train = int(len(segments) * 0.7)
    n_val = int(len(segments) * 0.15)
    val_segs = set(segments[n_train:n_train+n_val])
    df_val = df[df['SegmentID'].isin(val_segs)].copy()
    
    df_val['MovEncoded'] = (df_val['MovementState'] == 'Moving').astype(float)
    df_val['TimeGapClipped'] = process_timegap(df_val['TimeGapMinutes'])
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    windows = [10, 20, 30, 45, 60]
    
    results_summary = []
    
    for N in windows:
        model = FuelTimeAwareGRU(input_dim=6, hidden_dim=64).to(device)
        model.load_state_dict(torch.load(f'models/gru/best_gru_N{N}.pth'))
        model.eval()
        
        metrics = {
            'SpikeError': [],
            'SloshSTD': [],
            'RefuelError': [],
            'DrainError': [],
            'ConsumptionError': [],
            'EventDelay': []
        }
        
        total_latency = 0
        total_points = 0
        
        for sid, group in tqdm(df_val.groupby('SegmentID'), desc=f'Evaluating N={N}'):
            n_points = len(group)
            if n_points < N: continue
                
            noisy_fuel = group['NoisyFuel'].values
            clean_fuel = group['CleanFuel'].values
            speed = group['Speed'].values
            accel = group['Acceleration'].values
            mov = group['MovEncoded'].values
            rstd = group['RollingStd'].values
            tg = group['TimeGapClipped'].values
            labels = group['EventLabel'].values
            
            X_infer = []
            for i in range(N - 1, n_points):
                w_noisy = noisy_fuel[i - N + 1 : i + 1]
                w_speed = speed[i - N + 1 : i + 1]
                w_accel = accel[i - N + 1 : i + 1]
                w_mov = mov[i - N + 1 : i + 1]
                w_rstd = rstd[i - N + 1 : i + 1]
                w_tg = tg[i - N + 1 : i + 1]
                anchor = w_noisy[0]
                w_fuel_res = w_noisy - anchor
                X_infer.append(np.column_stack((w_fuel_res, w_speed, w_accel, w_mov, w_rstd, w_tg)))
                
            X_tensor = torch.tensor(np.array(X_infer, dtype=np.float32)).to(device)
            
            start_time = time.time()
            with torch.no_grad():
                y_pred = model(X_tensor).cpu().numpy().flatten()
            total_latency += (time.time() - start_time)
            total_points += len(X_tensor)
            
            preds = np.zeros(n_points)
            preds[:N-1] = noisy_fuel[:N-1]
            for idx, i in enumerate(range(N - 1, n_points)):
                preds[i] = y_pred[idx] + noisy_fuel[i - N + 1]
                
            # Compute Errors
            valid_mask = np.arange(N, n_points) # skip initial window
            
            spike_idx = np.where(labels[valid_mask] == 'Spike')[0] + N
            if len(spike_idx) > 0:
                metrics['SpikeError'].append(np.mean(np.abs(preds[spike_idx] - clean_fuel[spike_idx])))
                
            slosh_idx = np.where(labels[valid_mask] == 'Sloshing')[0] + N
            if len(slosh_idx) > 0:
                metrics['SloshSTD'].append(np.std(preds[slosh_idx] - clean_fuel[slosh_idx]))
                
            refuel_idx = np.where(labels[valid_mask] == 'Refuel')[0] + N
            if len(refuel_idx) > 0:
                metrics['RefuelError'].append(np.mean(np.abs(preds[refuel_idx] - clean_fuel[refuel_idx])))
                metrics['EventDelay'].append(calculate_delay(clean_fuel, preds, refuel_idx))
                
            drain_idx = np.where(labels[valid_mask] == 'Drain')[0] + N
            if len(drain_idx) > 0:
                metrics['DrainError'].append(np.mean(np.abs(preds[drain_idx] - clean_fuel[drain_idx])))
                metrics['EventDelay'].append(calculate_delay(clean_fuel, preds, drain_idx))
                
            norm_idx = np.where((labels[valid_mask] == 'Normal') & (mov[valid_mask] == 1))[0] + N
            if len(norm_idx) > 0:
                metrics['ConsumptionError'].append(np.mean(np.abs(preds[norm_idx] - clean_fuel[norm_idx])))
                
        # Aggregate
        avg_metrics = {k: np.nanmean(v) if len(v) > 0 else 0 for k, v in metrics.items()}
        avg_metrics['Latency (ms)'] = (total_latency / total_points) * 1000
        avg_metrics['N'] = N
        results_summary.append(avg_metrics)
        
    res_df = pd.DataFrame(results_summary)
    res_df = res_df.set_index('N')
    res_df = res_df[['SpikeError', 'SloshSTD', 'RefuelError', 'DrainError', 'ConsumptionError', 'EventDelay', 'Latency (ms)']]
    
    print("\n" + "="*80)
    print("PHASE 3: QUANTITATIVE WINDOW SEARCH RESULTS")
    print("="*80)
    print(res_df.round(3).to_markdown())
    print("="*80)

if __name__ == '__main__':
    validate_metrics()
