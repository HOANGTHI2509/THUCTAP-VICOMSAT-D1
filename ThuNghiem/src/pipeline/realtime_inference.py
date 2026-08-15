import os
import numpy as np
import torch
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.models.tcn_model import FuelCausalTCN_31

class FuelTCNRealtime:
    def __init__(self, model_path="models/tcn_weights/M4.pth", window_size=30):
        self.window_size = window_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.model = FuelCausalTCN_31().to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        
        # State buffers
        self.raw_fuel_buffer = []
        self.speed_buffer = []
        self.accel_buffer = []
        self.move_buffer = []
        self.std_buffer = []
        
    def reset(self):
        self.raw_fuel_buffer = []
        self.speed_buffer = []
        self.accel_buffer = []
        self.move_buffer = []
        self.std_buffer = []
        
    def cap_nhat(self, fuel, speed=0.0, accel=0.0, movement_state=1, rolling_std=0.0):
        self.raw_fuel_buffer.append(fuel)
        
        # Hậu kiểm: Lọc gai (Spike) đơn lẻ để tránh làm hỏng cửa sổ TCN (tương tự Innovation Threshold)
        if len(self.raw_fuel_buffer) >= 3:
            p1, p2, p3 = self.raw_fuel_buffer[-3], self.raw_fuel_buffer[-2], self.raw_fuel_buffer[-1]
            if abs(p2 - p1) > 10.0 and abs(p2 - p3) > 10.0 and (p2 - p1) * (p2 - p3) > 0:
                # p2 là gai đơn (vọt lên hoặc tụt xuống 1 điểm rồi quay lại)
                self.raw_fuel_buffer[-2] = (p1 + p3) / 2.0
                
        self.speed_buffer.append(speed)
        self.accel_buffer.append(accel)
        self.move_buffer.append(movement_state)
        self.std_buffer.append(rolling_std)
        
        if len(self.raw_fuel_buffer) > self.window_size:
            self.raw_fuel_buffer.pop(0)
            self.speed_buffer.pop(0)
            self.accel_buffer.pop(0)
            self.move_buffer.pop(0)
            self.std_buffer.pop(0)
            
        if len(self.raw_fuel_buffer) < self.window_size:
            # Not enough data for full window inference, just return raw or median
            return np.median(self.raw_fuel_buffer)
            
        anchor = np.median(self.raw_fuel_buffer[:-1]) # Anchor is median of past 29 points
        
        # Build features
        residual = np.array(self.raw_fuel_buffer) - anchor
        speed_arr = np.array(self.speed_buffer)
        accel_arr = np.array(self.accel_buffer)
        move_arr = np.array(self.move_buffer)
        std_arr = np.array(self.std_buffer)
        
        X = np.column_stack([residual, speed_arr, accel_arr, move_arr, std_arr])
        X_tensor = torch.tensor(X, dtype=torch.float32).unsqueeze(0).to(self.device) # Shape [1, 30, 5]
        
        with torch.no_grad():
            pred = self.model(X_tensor).item()
            
        return pred + anchor

