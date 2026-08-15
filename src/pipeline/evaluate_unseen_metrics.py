import os
import glob
import time
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from tqdm import tqdm

from src.models.time_aware_gru import FuelTimeAwareGRU
from src.core.filters.kalman_traditional import BoLocKalmanTieuChuan1D
from src.core.filters.kalman_adaptive import BoLocKalmanThichNghi1D
from src.pipeline.build_gru_dataset import process_timegap
from src.pipeline.evaluate_real_vcomsat import run_kalman, run_gru

def get_local_trend(raw_series, window=15):
    """Tính Local Trend bằng cách sử dụng rolling median để bỏ qua các spike/sloshing ngắn hạn."""
    return pd.Series(raw_series).rolling(window=window, center=True, min_periods=1).median().values

def detect_events(raw_fuel):
    """
    Dò tìm các sự kiện trong chuỗi Raw Fuel.
    Trả về các dict chứa danh sách các indices hoặc start/end cho từng loại sự kiện.
    """
    n = len(raw_fuel)
    events = {'Refuel': [], 'Drain': [], 'Spike': [], 'Sloshing': [], 'Normal': []}
    
    # 1. Detect Refuel & Drain (biến động > 10L và kéo dài >= 2 timestep)
    smooth_raw = pd.Series(raw_fuel).rolling(window=3, center=True, min_periods=1).median().values
    
    i = 0
    while i < n - 1:
        start = i
        curr_val = smooth_raw[i]
        j = i + 1
        
        while j < n:
            diff = smooth_raw[j] - smooth_raw[j-1]
            if abs(diff) < 0.5: # Plateau
                break
            # Nếu đổi chiều
            if (smooth_raw[j] - curr_val) * diff < 0 and abs(smooth_raw[j] - smooth_raw[j-1]) > 2:
                break
            j += 1
            
        delta = smooth_raw[j-1] - smooth_raw[start]
        duration = j - 1 - start
        
        if abs(delta) > 10.0 and duration >= 1: # Allow duration >= 1 if it's a huge jump
            if delta > 0:
                events['Refuel'].append((start, j-1))
            else:
                events['Drain'].append((start, j-1))
            i = j
        else:
            i += 1
            
    # 2. Detect Spikes (điểm dị thường giật lên/xuống rồi quay về)
    local_trend = get_local_trend(raw_fuel, window=11)
    for i in range(2, n - 2):
        in_event = any(s <= i <= e for s, e in events['Refuel'] + events['Drain'])
        if in_event: continue
        
        if abs(raw_fuel[i] - local_trend[i]) > 10.0:
            events['Spike'].append(i)
            
    # 3. Detect Sloshing (dao động mạnh nhưng không hao hụt thực)
    window = 15
    rolling_std = pd.Series(raw_fuel).rolling(window=window, center=True, min_periods=1).std().values
    i = 0
    while i < n - window:
        in_event = any(s <= i+j <= e for s, e in events['Refuel'] + events['Drain'] for j in range(window))
        if not in_event and np.mean(rolling_std[i:i+window]) > 3.0:
            if abs(local_trend[i+window-1] - local_trend[i]) < 5.0:
                events['Sloshing'].append((i, i+window-1))
                i += window
                continue
        i += 1
        
    # 4. Normal
    normal_mask = np.ones(n, dtype=bool)
    for s, e in events['Refuel'] + events['Drain'] + events['Sloshing']:
        normal_mask[s:e+1] = False
    for sp in events['Spike']:
        normal_mask[sp] = False
        
    events['Normal'] = np.where(normal_mask)[0]
    
    return events, local_trend

def calculate_metrics(raw, preds_dict, events, local_trend):
    results = {}
    
    for model_name, pred in preds_dict.items():
        m = {}
        
        # Event Preservation Error & Event Delay
        refuel_errors, drain_errors = [], []
        delays = []
        
        for s, e in events['Refuel'] + events['Drain']:
            if s == e: continue
            raw_delta = raw[e] - raw[s]
            pred_delta = pred[e] - pred[s]
            error = abs(raw_delta - pred_delta)
            
            if raw_delta > 0: refuel_errors.append(error)
            else: drain_errors.append(error)
            
            # Delay (Timesteps to reach 80% of Raw Delta)
            target = raw[s] + 0.8 * raw_delta
            delay = 0
            found = False
            for i in range(s, min(e + 15, len(pred))):
                if (raw_delta > 0 and pred[i] >= target) or (raw_delta < 0 and pred[i] <= target):
                    delay = i - s
                    found = True
                    break
            if found: delays.append(delay)
            else: delays.append(15) # Penalty
            
        m['RefuelError'] = np.mean(refuel_errors) if refuel_errors else np.nan
        m['DrainError'] = np.mean(drain_errors) if drain_errors else np.nan
        m['EventDelay'] = np.mean(delays) if delays else np.nan
        
        # Spike Suppression
        spike_ratios = []
        for sp in events['Spike']:
            baseline = local_trend[sp]
            raw_spike_amp = abs(raw[sp] - baseline)
            if raw_spike_amp == 0: continue
            pred_spike_amp = abs(pred[sp] - baseline)
            
            ratio = max(0, min(100, 100 * (1.0 - pred_spike_amp / raw_spike_amp)))
            spike_ratios.append(ratio)
        m['SpikeSuppression'] = np.mean(spike_ratios) if spike_ratios else np.nan
        
        # Sloshing Noise
        slosh_stds = []
        # Calculate own trend to see how much pred wobbles around its own baseline
        own_trend = pd.Series(pred).rolling(window=31, center=True, min_periods=1).median().values
        for s, e in events['Sloshing']:
            residual = pred[s:e+1] - own_trend[s:e+1]
            slosh_stds.append(np.std(residual))
        m['SloshingSTD'] = np.mean(slosh_stds) if slosh_stds else np.nan
        
        # Trend Error (Robust Slope difference on Normal segments)
        if len(events['Normal']) > 50:
            breaks = np.where(np.diff(events['Normal']) > 1)[0]
            chunks = np.split(events['Normal'], breaks + 1)
            
            trend_errors = []
            for chunk in chunks:
                if len(chunk) < 20: continue
                # calculate theilslopes
                slope_raw, _, _, _ = stats.theilslopes(raw[chunk], np.arange(len(chunk)))
                slope_pred, _, _, _ = stats.theilslopes(pred[chunk], np.arange(len(chunk)))
                trend_errors.append(abs(slope_raw - slope_pred))
            m['TrendError'] = np.mean(trend_errors) if trend_errors else np.nan
        else:
            m['TrendError'] = np.nan
            
        results[model_name] = m
    return results

def plot_zooms(df, raw, preds_dict, events, vehicle_id):
    os.makedirs('artifacts', exist_ok=True)
    # Pick one event for each category
    cases = {
        'Spike': None,
        'Sloshing': None,
        'Refuel': None,
        'Drain': None
    }
    
    if events['Spike']: cases['Spike'] = (max(0, events['Spike'][0] - 25), min(len(raw), events['Spike'][0] + 25))
    if events['Sloshing']: cases['Sloshing'] = (max(0, events['Sloshing'][0][0] - 10), min(len(raw), events['Sloshing'][0][1] + 10))
    if events['Refuel']: cases['Refuel'] = (max(0, events['Refuel'][0][0] - 15), min(len(raw), events['Refuel'][0][1] + 15))
    if events['Drain']: cases['Drain'] = (max(0, events['Drain'][0][0] - 15), min(len(raw), events['Drain'][0][1] + 15))
    
    colors = {'Raw': 'black', 'Standard Kalman': 'blue', 'Adaptive Kalman': 'orange', 'Time-aware GRU (N=30)': 'red'}
    alphas = {'Raw': 0.3, 'Standard Kalman': 0.6, 'Adaptive Kalman': 0.8, 'Time-aware GRU (N=30)': 0.9}
    markers = {'Raw': '.', 'Standard Kalman': '', 'Adaptive Kalman': '', 'Time-aware GRU (N=30)': ''}
    
    for name, span in cases.items():
        if not span: continue
        s, e = span
        idx = range(s, e)
        plt.figure(figsize=(8, 5))
        for m_name, pred in preds_dict.items():
            plt.plot(pd.Series(pred, index=df.index)[idx], 
                     color=colors[m_name], alpha=alphas[m_name], marker=markers[m_name], 
                     linewidth=1 if m_name == 'Raw' else 2, label=m_name)
        plt.title(f"Zoom {name} - Vehicle {vehicle_id}")
        plt.xlabel("Time Step")
        plt.ylabel("Fuel Level (L)")
        plt.legend()
        plt.grid(True)
        plt.savefig(f"artifacts/Zoom_{name}.png")
        plt.close()

def main():
    print("Khởi chạy Kế hoạch Đánh giá Metrics Độc lập trên Unseen Vehicles...")
    os.makedirs('artifacts', exist_ok=True)
    
    # Locate all processed cars
    all_files = glob.glob('data/processed/CarFuelHistory_Processed_*.csv')
    
    # Exclude Car1 to Car5 (Seen)
    seen_cars = ['Car1.csv', 'Car2.csv', 'Car3.csv', 'Car4.csv', 'Car5.csv']
    unseen_files = [f for f in all_files if not any(sc in f for sc in seen_cars)]
    
    # Load GRU Final Model (which is Model D)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    gru_model = FuelTimeAwareGRU(input_dim=6, hidden_dim=64).to(device)
    gru_model.load_state_dict(torch.load('models/gru/best_gru_final.pth', map_location=device))
    gru_model.eval()

    all_results = []
    
    for file in tqdm(unseen_files, desc="Đánh giá Unseen"):
        vehicle_id = os.path.basename(file).split('_')[-1].replace('.csv', '')
        df = pd.read_csv(file)
        if len(df) < 100: continue
        
        # 1. Predict Kalmans
        t0 = time.time()
        std_kalman = run_kalman(df, BoLocKalmanTieuChuan1D, is_adaptive=False)
        std_lat = (time.time() - t0) / len(df) * 1000
        
        t0 = time.time()
        adp_kalman = run_kalman(df, BoLocKalmanThichNghi1D, is_adaptive=True)
        adp_lat = (time.time() - t0) / len(df) * 1000
        
        raw_fuel = df['FuelLevel'].values
        events, local_trend = detect_events(raw_fuel)
        
        preds_dict = {
            'Raw': raw_fuel,
            'Standard Kalman': std_kalman,
            'Adaptive Kalman': adp_kalman,
        }
        
        # Predict GRU Final
        t0 = time.time()
        gru_preds = run_gru(df, gru_model, N=30)
        gru_lat = (time.time() - t0) / max(1, len(df)) * 1000
        preds_dict['Time-aware GRU Final'] = gru_preds
            
        metrics = calculate_metrics(raw_fuel, preds_dict, events, local_trend)
        
        for m_name, m_vals in metrics.items():
            m_vals['VehicleID'] = vehicle_id
            m_vals['Model'] = m_name
            if m_name == 'Raw': m_vals['Latency'] = 0.0
            elif 'Standard' in m_name: m_vals['Latency'] = std_lat
            elif 'Adaptive' in m_name: m_vals['Latency'] = adp_lat
            else: m_vals['Latency'] = gru_lat
            
            all_results.append(m_vals)
            
    res_df = pd.DataFrame(all_results)
    
    # Tạo thư mục final_evaluation
    os.makedirs('final_evaluation', exist_ok=True)
    
    # Lưu per_vehicle.csv
    res_df.to_csv('final_evaluation/per_vehicle.csv', index=False)
    
    # 2. Báo cáo Thống kê
    summary = []
    model_names = ['Raw', 'Standard Kalman', 'Adaptive Kalman', 'Time-aware GRU Final']
    
    for model in model_names:
        m_df = res_df[res_df['Model'] == model]
        if len(m_df) == 0: continue
        
        summary.append({
            'Model': model,
            'Spike Suppression (%) ↑': f"{m_df['SpikeSuppression'].mean():.1f} ± {m_df['SpikeSuppression'].std():.1f}",
            'Sloshing Noise (L) ↓': f"{m_df['SloshingSTD'].mean():.2f} ± {m_df['SloshingSTD'].std():.2f}",
            'Refuel Amp Distortion (L) ↓': f"{m_df['RefuelError'].mean():.2f} ± {m_df['RefuelError'].std():.2f}",
            'Drain Amp Distortion (L) ↓': f"{m_df['DrainError'].mean():.2f} ± {m_df['DrainError'].std():.2f}",
            'Event Delay (steps) ↓': round(m_df['EventDelay'].mean(), 1),
            'Trend Error (L/step) ↓': round(m_df['TrendError'].mean(), 4),
            'Latency (ms/pt) ↓': round(m_df['Latency'].mean(), 3)
        })
        
    sum_df = pd.DataFrame(summary)
    sum_df.to_csv('final_evaluation/summary.csv', index=False)
    
    # Xuất ra summary.md
    md_content = "# BẢNG TỔNG HỢP METRICS ĐỊNH LƯỢNG TRÊN UNSEEN VEHICLES (MEAN ± STD)\n\n"
    md_content += sum_df.to_markdown(index=False)
    with open('final_evaluation/summary.md', 'w', encoding='utf-8') as f:
        f.write(md_content)
        
    print("\n" + "="*90)
    print("BẢNG TỔNG HỢP METRICS ĐỊNH LƯỢNG TRÊN UNSEEN VEHICLES (MEAN ± STD)")
    print("="*90)
    print(sum_df.to_markdown(index=False))
    print("="*90)
    print("Kết quả đã được lưu tại thư mục final_evaluation/")

if __name__ == '__main__':
    main()
