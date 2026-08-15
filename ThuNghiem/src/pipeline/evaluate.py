import os
import sys
import time
import numpy as np
import pandas as pd
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.core.filters.kalman_adaptive import BoLocKalmanThichNghi1D
from src.core.filters.kalman_smoother import RTSSmoother

from src.pipeline.evaluate_event_aware import find_spikes
from src.models.tcn_model import FuelCausalTCN_31

def run_adaptive_kalman(raw):
    kf = BoLocKalmanThichNghi1D(trang_thai_ban_dau=raw[0])
    out = np.zeros_like(raw)
    for i, z in enumerate(raw):
        out[i] = kf.cap_nhat(z)
    return out

def run_standard_kalman(raw):
    kf = RTSSmoother(process_noise_q=0.1, measurement_noise_r=10.0)
    return kf.filter(raw) # Forward pass only

def load_tcn_model(model_name, num_features):
    device = torch.device("cpu")
    model = FuelCausalTCN_31(in_channels=num_features).to(device)
    model.load_state_dict(torch.load(f"models/tcn_weights/{model_name}.pth", map_location=device))
    model.eval()
    return model

def run_tcn(raw_features, model, num_features):
    # raw_features: (N, 5)
    # We need to simulate sliding window inference
    N = len(raw_features)
    window_size = 30
    out = np.zeros(N)
    out[:window_size-1] = raw_features[:window_size-1, 0] # fallback for start
    
    device = torch.device("cpu")
    
    # We can do batch inference for speed in evaluation, but to measure latency we should do 1 by 1
    # For now, batch inference to get outputs
    windows = []
    anchors = []
    for i in range(window_size - 1, N):
        past_fuel = raw_features[i - window_size + 1 : i, 0]
        anchor = np.median(past_fuel)
        
        win = raw_features[i - window_size + 1 : i + 1].copy()
        win[:, 0] -= anchor
        windows.append(win[:, :num_features])
        anchors.append(anchor)
        
    X_tensor = torch.tensor(np.array(windows), dtype=torch.float32).to(device)
    with torch.no_grad():
        preds = model(X_tensor).numpy()
        
    for i, (pred, anchor) in enumerate(zip(preds, anchors)):
        out[i + window_size - 1] = pred + anchor
        
    return out

def evaluate_metrics(models_dict, df):
    results = {m: {'Noise': [], 'Spike': [], 'Preservation': [], 'Delay': [], 'Trend': []} for m in models_dict.keys()}
    
    for seg_id, group in df.groupby('SegmentID'):
        raw = group['FuelLevel'].values
        if len(raw) < 100: continue
        
        events = find_events_plateau(raw)
        spikes = find_spikes(raw, events, min_spike=3.0)
        
        # Determine sloshing zones (regions far from events and spikes, with high variance)
        # For simplicity, let's just use the chunks between events
        last_idx = 0
        normal_chunks = []
        for (s, e, ev_type, _) in events:
            if s - last_idx > 50:
                normal_chunks.append((last_idx + 10, s - 10))
            last_idx = e
        if len(raw) - last_idx > 50:
            normal_chunks.append((last_idx + 10, len(raw) - 10))
            
        for m_name, output in models_dict.items():
            # 1. Noise Reduction: average standard deviation of point-to-point diffs in normal zones
            noise_list = []
            for (ns, ne) in normal_chunks:
                # To remove global trend, we diff
                diffs = np.diff(output[ns:ne])
                noise_list.append(np.std(diffs))
            if noise_list:
                results[m_name]['Noise'].append(np.mean(noise_list))
                
            # 2. Spike Suppression
            spike_errors = []
            for (s, jump) in spikes:
                pre_val = np.median(output[max(0, s-10):s])
                max_dev = np.max(np.abs(output[s:min(len(output), s+5)] - pre_val))
                spike_errors.append(max_dev)
            if spike_errors:
                results[m_name]['Spike'].extend(spike_errors)
                
            # 3 & 4: Event Preservation & Delay
            pres_errors = []
            delays = []
            for (s, e, ev_type, true_jump) in events:
                if ev_type == 'false_positive_spike': continue
                pre_val = np.median(output[max(0, s-10):s]) if s > 0 else output[0]
                post_val = np.median(output[e:min(len(output), e+10)])
                model_jump = post_val - pre_val
                pres_errors.append(np.abs(model_jump - true_jump))
                
                # Delay: time to 90% of model_jump
                target_val = pre_val + 0.9 * true_jump
                delay = 30 # default max delay
                for t in range(s, min(len(output), s+30)):
                    if (true_jump > 0 and output[t] >= target_val) or (true_jump < 0 and output[t] <= target_val):
                        delay = t - s
                        break
                delays.append(delay)
                
            if pres_errors:
                results[m_name]['Preservation'].extend(pres_errors)
            if delays:
                results[m_name]['Delay'].extend(delays)
                
    # Aggregate
    agg = {}
    for m in results:
        agg[m] = {
            'Noise (L)': np.mean(results[m]['Noise']) if results[m]['Noise'] else 0,
            'Spike (L)': np.mean(results[m]['Spike']) if results[m]['Spike'] else 0,
            'Preservation (L)': np.mean(results[m]['Preservation']) if results[m]['Preservation'] else 0,
            'Delay (steps)': np.mean(results[m]['Delay']) if results[m]['Delay'] else 0
        }
    return pd.DataFrame(agg).T

def main():
    print("Loading Test Dataset...")
    df = pd.read_csv("data/processed/CarFuelHistory_Processed_Car1.csv") # Using Car1 for evaluation as representative
    
    # Extract features for TCN
    raw_fuel = df['FuelLevel'].values
    speed = df['Speed'].values if 'Speed' in df.columns else np.zeros_like(raw_fuel)
    accel = df['Acceleration'].values if 'Acceleration' in df.columns else np.zeros_like(raw_fuel)
    movement = df['MovementState_Num'].values if 'MovementState_Num' in df.columns else np.zeros_like(raw_fuel)
    rstd = df['RollingStd'].values if 'RollingStd' in df.columns else np.zeros_like(raw_fuel)
    
    features = np.column_stack([raw_fuel, speed, accel, movement, rstd]).astype(np.float32)
    features = np.nan_to_num(features, nan=0.0)
    
    models_dict = {}
    models_dict['Standard Kalman'] = run_standard_kalman(raw_fuel)
    models_dict['Adaptive Kalman'] = run_adaptive_kalman(raw_fuel)
    
    ablation = {
        'M1 (Fuel Only)': 1,
        'M2 (+Speed)': 2,
        'M3 (+Accel)': 3,
        'M4 (All)': 5
    }
    
    for m_name, n_feat in ablation.items():
        base_name = m_name.split()[0]
        if os.path.exists(f"models/tcn_weights/{base_name}.pth"):
            model = load_tcn_model(base_name, n_feat)
            
            # Measure latency
            start_time = time.time()
            out = run_tcn(features, model, n_feat)
            latency = (time.time() - start_time) / len(raw_fuel) * 1000 # ms per step
            
            models_dict[m_name] = out
            print(f"{m_name} loaded. Latency: {latency:.2f} ms/step")
        else:
            print(f"Warning: {base_name} not found.")
            
    print("\nEvaluating Independent Metrics...")
    results_df = evaluate_metrics(models_dict, df)
    print(results_df.round(3))
    
    results_df.to_markdown(os.path.join("artifacts", "Independent_Evaluation_Report.md"))
    
if __name__ == "__main__":
    main()
