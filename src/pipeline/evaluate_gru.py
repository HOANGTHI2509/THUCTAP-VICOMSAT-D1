import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import time

from src.models.time_aware_gru import FuelTimeAwareGRU
from src.pipeline.build_gru_dataset import process_timegap

def evaluate_models():
    os.makedirs('artifacts', exist_ok=True)
    df = pd.read_csv('data/gru_dataset/synthetic_dataset.csv')
    
    # Pre-encode
    df['MovEncoded'] = (df['MovementState'] == 'Moving').astype(float)
    df['TimeGapClipped'] = process_timegap(df['TimeGapMinutes'])
    
    # Pick a segment that has multiple event types (Refuel, Drain, Spike, Sloshing)
    segment_id = None
    for sid, group in df.groupby('SegmentID'):
        events = group['EventLabel'].unique()
        if 'Refuel' in events and 'Drain' in events and 'Spike' in events:
            segment_id = sid
            break
            
    if segment_id is None:
        # Fallback
        segment_id = df['SegmentID'].unique()[0]
        
    segment = df[df['SegmentID'] == segment_id].copy()
    n_points = len(segment)
    
    windows = [10, 20, 30, 45, 60]
    results = {}
    latencies = {}
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    noisy_fuel = segment['NoisyFuel'].values
    speed = segment['Speed'].values
    accel = segment['Acceleration'].values
    mov = segment['MovEncoded'].values
    rstd = segment['RollingStd'].values
    tg = segment['TimeGapClipped'].values
    
    for N in windows:
        model = FuelTimeAwareGRU(input_dim=6, hidden_dim=64).to(device)
        model.load_state_dict(torch.load(f'models/gru/best_gru_N{N}.pth'))
        model.eval()
        
        preds = np.zeros(n_points)
        preds[:N-1] = noisy_fuel[:N-1] # First N points just copy Noisy
        
        # Build features for inference
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
            
            w_feats = np.column_stack((w_fuel_res, w_speed, w_accel, w_mov, w_rstd, w_tg))
            X_infer.append(w_feats)
            
        X_infer = torch.tensor(np.array(X_infer, dtype=np.float32)).to(device)
        
        start_time = time.time()
        with torch.no_grad():
            y_pred = model(X_infer).cpu().numpy().flatten()
        inference_time = (time.time() - start_time) / len(X_infer) * 1000 # ms per point
        latencies[N] = inference_time
        
        # Reconstruct Absolute Fuel
        for idx, i in enumerate(range(N - 1, n_points)):
            anchor = noisy_fuel[i - N + 1]
            preds[i] = y_pred[idx] + anchor
            
        results[N] = preds
        
    # Plotting
    fig, axes = plt.subplots(len(windows), 1, figsize=(12, 4 * len(windows)), sharex=True)
    clean_fuel = segment['CleanFuel'].values
    
    for i, N in enumerate(windows):
        ax = axes[i]
        ax.plot(noisy_fuel, color='red', alpha=0.3, label='Noisy Input')
        ax.plot(clean_fuel, color='blue', alpha=0.7, linewidth=2, label='Clean Target')
        ax.plot(results[N], color='green', linewidth=2, label=f'GRU (N={N})')
        ax.set_title(f'N = {N} (Latency: {latencies[N]:.3f} ms/point)')
        ax.legend()
        ax.grid(True)
        
    plt.tight_layout()
    plt.savefig('artifacts/Window_Search_Results.png')
    plt.close()
    
    print("Inference completed. Results saved to artifacts/Window_Search_Results.png")
    
if __name__ == '__main__':
    evaluate_models()
