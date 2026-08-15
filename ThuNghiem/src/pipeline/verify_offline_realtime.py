import os
import sys
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.pipeline.realtime_inference import FuelTCNRealtime
from src.models.tcn_model import FuelCausalTCN_31

def predict_tcn_offline(X, device):
    model = FuelCausalTCN_31().to(device)
    model.load_state_dict(torch.load("src/models/best_Model_3_EventPreserving.pth", map_location=device))
    model.eval()
    
    batch_size = 1024
    all_preds = []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            batch_X = torch.tensor(X[i:i+batch_size], dtype=torch.float32).to(device)
            preds = model(batch_X)
            all_preds.append(preds.cpu().numpy())
    return np.concatenate(all_preds).squeeze()

def test_equivalence():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    X_test = np.load("data/zeroshot_dataset/X_test.npy")
    M_test = pd.read_pickle("data/zeroshot_dataset/M_test.pkl")
    
    seg_counts = M_test['SegmentID'].value_counts()
    target_seg = seg_counts[seg_counts > 100].index[0]
    
    idx = M_test[M_test['SegmentID'] == target_seg].index
    X_offline = X_test[idx]
    M_offline = M_test.loc[idx].reset_index(drop=True)
    
    preds_offline_res = predict_tcn_offline(X_offline, device)
    preds_offline = preds_offline_res + M_offline['Anchor'].values
    
    tcn_rt = FuelTCNRealtime("src/models/best_Model_3_EventPreserving.pth")
    
    df_raw = pd.read_csv("data/processed/CarFuelHistory_Processed_Car5.csv")
    df_seg = df_raw[df_raw['SegmentID'] == target_seg].sort_values('FuelTime').reset_index(drop=True)
    
    for col in ['Speed', 'Acceleration', 'RollingStd', 'FuelLevel']:
        df_seg[col] = df_seg[col].fillna(0)
    
    move_map = {'Stopped': 0.0, 'Moving': 1.0, 'Uncertain': 0.5}
    
    preds_realtime = []
    tcn_rt.reset()
    for i, row in df_seg.iterrows():
        fuel = row['FuelLevel']
        speed = row['Speed']
        accel = row['Acceleration']
        move = move_map.get(row['MovementState'], 0.5)
        std = row['RollingStd']
        
        pred = tcn_rt.cap_nhat(fuel, speed, accel, move, std)
        preds_realtime.append(pred)
        
    preds_realtime = np.array(preds_realtime)
    preds_rt_aligned = preds_realtime[29:]
    
    diff = np.abs(preds_offline - preds_rt_aligned)
    max_diff = np.max(diff)
    mean_diff = np.mean(diff)
    
    print(f"\nEquivalence Test Results:")
    print(f"Max Difference: {max_diff:.6f} L")
    print(f"Mean Difference: {mean_diff:.6f} L")
    
    if max_diff < 1e-4:
        print("SUCCESS: Offline and Realtime are EXACTLY identical!")
    else:
        print("FAILURE: Realtime output differs from Offline output!")
        print("\nTop 5 differences:")
        diff_idx = np.argsort(diff)[::-1][:5]
        for i in diff_idx:
            print(f"Idx {i}: Offline = {preds_offline[i]:.4f}, Realtime = {preds_rt_aligned[i]:.4f}, Diff = {diff[i]:.4f}")

if __name__ == "__main__":
    test_equivalence()
