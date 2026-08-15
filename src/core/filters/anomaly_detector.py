import pandas as pd
import numpy as np

class FuelAnomalyDetector:
    def __init__(self, capacity: float = 200.0, look_ahead_hours: float = 6.0, min_low_minutes: float = 30.0, spike_threshold: float = 10.0, spike_lookahead_mins: float = 60.0):
        self.capacity = capacity
        self.look_ahead_hours = look_ahead_hours
        self.min_low_minutes = min_low_minutes
        self.spike_threshold = spike_threshold
        self.spike_lookahead_mins = spike_lookahead_mins
        
    def detect_and_clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Quét và làm sạch dữ liệu nhiễu chữ U.
        Input: DataFrame có các cột: FuelTime, FuelLevel (Raw)
        Output: Cập nhật DataFrame với các cột mới:
                - FuelAnomalyType
                - CleanedFuel
        """
        # Đảm bảo được sắp xếp theo thời gian
        df = df.sort_values('FuelTime').copy()
        
        # Mặc định
        df['FuelAnomalyType'] = 'NORMAL'
        df['CleanedFuel'] = df['FuelLevel'].copy()
        
        # Khởi tạo kích thước cửa sổ cho robust baseline
        window_size = 10
        
        n = len(df)
        i = 0
        
        while i < n:
            current_fuel = df['FuelLevel'].iloc[i]
            
            # Tính baseline động dựa trên CleanedFuel (những điểm đã được làm sạch)
            if i == 0:
                baseline = current_fuel
            else:
                start_b = max(0, i - window_size)
                baseline = df['CleanedFuel'].iloc[start_b:i].median()
                if pd.isna(baseline):
                    baseline = current_fuel
            
            # Điều kiện A: Rơi cực mạnh
            drop_threshold = max(30.0, 0.15 * self.capacity)
            
            if baseline - current_fuel > drop_threshold:
                start_idx = i
                start_time = df['FuelTime'].iloc[start_idx]
                
                # Quét tương lai tìm Recovery
                recovery_idx = -1
                max_lookahead_time = start_time + pd.Timedelta(hours=self.look_ahead_hours)
                
                k = i + 1
                valley_min_fuel = current_fuel
                
                while k < n and df['FuelTime'].iloc[k] <= max_lookahead_time:
                    val = df['FuelLevel'].iloc[k]
                    valley_min_fuel = min(valley_min_fuel, val)
                    
                    epsilon = max(10.0, 0.05 * baseline)
                    if abs(val - baseline) <= epsilon:
                        recovery_idx = k
                        break
                    k += 1
                
                if recovery_idx != -1:
                    # Tìm thấy điểm Recovery
                    recovery_time = df['FuelTime'].iloc[recovery_idx]
                    duration_mins = (recovery_time - start_time).total_seconds() / 60.0
                    
                    # Kiểm tra Điều kiện B (chạm đáy rất sâu) và C (kéo dài đủ lâu)
                    if valley_min_fuel < 0.2 * baseline and duration_mins >= self.min_low_minutes:
                        # ĐÂY LÀ SENSOR DROPOUT!
                        df.iloc[start_idx:recovery_idx, df.columns.get_loc('FuelAnomalyType')] = 'SENSOR_DROPOUT'
                        df.iloc[recovery_idx, df.columns.get_loc('FuelAnomalyType')] = 'RECOVERY'
                        
                        # Nội suy Trend-preserving (nối 2 baseline)
                        # Tính baseline_after nhưng loại trừ các điểm bị rớt (nếu có dropout liên tiếp)
                        after_window_end = min(n, recovery_idx + window_size)
                        future_vals = df['FuelLevel'].iloc[recovery_idx:after_window_end]
                        recovery_val = df['FuelLevel'].iloc[recovery_idx]
                        
                        # Chỉ lấy những điểm không bị rớt quá 30L so với điểm recovery
                        valid_future_vals = future_vals[abs(future_vals - recovery_val) < max(30.0, 0.15 * self.capacity)]
                        
                        if len(valid_future_vals) > 0:
                            baseline_after = valid_future_vals.median()
                        else:
                            baseline_after = recovery_val
                            
                        # Thay thế dữ liệu bằng nội suy
                        total_time_diff = (recovery_time - start_time).total_seconds()
                        if total_time_diff == 0: total_time_diff = 1 # Tránh chia 0
                        
                        for step in range(recovery_idx - start_idx):
                            step_time = df['FuelTime'].iloc[start_idx + step]
                            ratio = (step_time - start_time).total_seconds() / total_time_diff
                            interpolated_val = baseline + ratio * (baseline_after - baseline)
                            df.iloc[start_idx + step, df.columns.get_loc('CleanedFuel')] = interpolated_val
                            
                        # Nhảy đến điểm recovery để tiếp tục quét
                        i = recovery_idx
                        continue
                
                # Nếu không phải dropout hợp lệ (không đủ sâu, không đủ lâu, hoặc không có recovery)
                pass # Chuyển sang check Spike bên dưới
                
            # Điều kiện E: Phát hiện SPIKE (Nhiễu lồi/lõm ngắn hạn)
            # Nếu chênh lệch > spike_threshold (e.g. 10L) và quay về baseline trong vòng 1 giờ
            if abs(current_fuel - baseline) > self.spike_threshold:
                start_idx = i
                start_time = df['FuelTime'].iloc[start_idx]
                
                recovery_idx = -1
                max_spike_time = start_time + pd.Timedelta(minutes=self.spike_lookahead_mins)
                k = i + 1
                
                while k < n and df['FuelTime'].iloc[k] <= max_spike_time:
                    val = df['FuelLevel'].iloc[k]
                    epsilon = max(3.0, 0.01 * baseline) # Ngưỡng phục hồi khắt khe hơn cho spike (phải về sát baseline)
                    if abs(val - baseline) <= epsilon:
                        recovery_idx = k
                        break
                    k += 1
                
                if recovery_idx != -1:
                    # Là SPIKE!
                    df.iloc[start_idx:recovery_idx, df.columns.get_loc('FuelAnomalyType')] = 'SPIKE'
                    
                    # Nội suy thẳng qua Spike
                    after_window_end = min(n, recovery_idx + window_size)
                    future_vals = df['FuelLevel'].iloc[recovery_idx:after_window_end]
                    recovery_val = df['FuelLevel'].iloc[recovery_idx]
                    valid_future_vals = future_vals[abs(future_vals - recovery_val) < self.spike_threshold]
                    
                    if len(valid_future_vals) > 0:
                        baseline_after = valid_future_vals.median()
                    else:
                        baseline_after = recovery_val
                        
                    total_time_diff = (df['FuelTime'].iloc[recovery_idx] - start_time).total_seconds()
                    if total_time_diff == 0: total_time_diff = 1
                    
                    for step in range(recovery_idx - start_idx):
                        step_time = df['FuelTime'].iloc[start_idx + step]
                        ratio = (step_time - start_time).total_seconds() / total_time_diff
                        interpolated_val = baseline + ratio * (baseline_after - baseline)
                        df.iloc[start_idx + step, df.columns.get_loc('CleanedFuel')] = interpolated_val
                        
                    i = recovery_idx
                    continue
                    
            i += 1
                
        return df
