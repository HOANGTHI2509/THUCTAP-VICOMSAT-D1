import pandas as pd
import numpy as np

class FuelAnomalyDetector:
    def __init__(self, capacity: float = 200.0, look_ahead_hours: float = 6.0, min_low_minutes: float = 10.0, spike_threshold: float = 10.0, spike_lookahead_mins: float = 60.0):
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
        if df.empty or len(df) < 3:
            res = df.copy()
            res['FuelAnomalyType'] = 'NORMAL'
            res['CleanedFuel'] = res['FuelLevel'].copy() if 'FuelLevel' in res.columns else np.nan
            return res

        df = df.sort_values('FuelTime', kind='stable').copy()
        fuels = pd.to_numeric(df['FuelLevel'], errors='coerce').to_numpy(dtype=float)
        times = pd.to_datetime(df['FuelTime']).to_numpy()
        n = len(df)

        anomaly_type = np.full(n, 'NORMAL', dtype=object)
        cleaned = fuels.copy()

        drop_thresh = max(5.0, 0.025 * self.capacity)

        i = 0
        while i < n - 2:
            start_b = max(0, i - 7)
            baseline_before = float(np.nanmedian(cleaned[start_b:i])) if i > 0 else float(fuels[i])
            if np.isnan(baseline_before):
                baseline_before = float(fuels[i])

            # Kiểm tra sụt giảm mất tín hiệu cảm biến (rơi về sát 0L hoặc NaN đột ngột từ mức cao)
            drop_start_idx = -1
            for look_idx in range(i, min(n, i + 5)):
                val_look = fuels[look_idx]
                if np.isnan(val_look) or (val_look <= 1.0 and baseline_before >= drop_thresh):
                    drop_start_idx = i
                    break

            if drop_start_idx != -1:
                t_start = times[drop_start_idx]
                max_t = t_start + np.timedelta64(int(self.look_ahead_hours * 3600), 's')

                recovery_idx = -1
                k = drop_start_idx + 1
                while k < n and times[k] <= max_t:
                    val = fuels[k]
                    # Nếu gặp sự kiện bơm xăng lớn thì không nối đè qua
                    if not np.isnan(val) and val > baseline_before + 10.0:
                        break
                    # Phục hồi sụt cảm biến CHỈ KHI mức nhiên liệu nẩy về đúng sát mức trước sụt (trong khoảng ±5.0L)
                    epsilon = min(max(4.0, 0.03 * baseline_before), 5.0)
                    if not np.isnan(val) and abs(val - baseline_before) <= epsilon:
                        recovery_idx = k
                        break
                    k += 1

                if recovery_idx != -1:
                    duration_mins = float((times[recovery_idx] - t_start) / np.timedelta64(1, 'm'))
                    if duration_mins >= self.min_low_minutes:
                        t0 = times[drop_start_idx - 1] if drop_start_idx > 0 else times[drop_start_idx]
                        v0 = cleaned[drop_start_idx - 1] if drop_start_idx > 0 else baseline_before
                        t1 = times[recovery_idx]
                        v1 = fuels[recovery_idx]

                        dt_total = float((t1 - t0) / np.timedelta64(1, 's'))
                        if dt_total <= 0:
                            dt_total = 1.0

                        for idx_sub in range(drop_start_idx, recovery_idx):
                            ratio = float((times[idx_sub] - t0) / np.timedelta64(1, 's')) / dt_total
                            cleaned[idx_sub] = v0 + ratio * (v1 - v0)
                            anomaly_type[idx_sub] = 'SENSOR_DROPOUT'

                        anomaly_type[recovery_idx] = 'RECOVERY'
                        i = recovery_idx
                        continue
            
            # Điều kiện E: Phát hiện SPIKE (Nhiễu lồi/lõm ngắn hạn)
            # Nếu chênh lệch > spike_threshold (e.g. 10L) và quay về baseline trong vòng 1 giờ
            if abs(fuels[i] - baseline_before) > self.spike_threshold:
                start_idx = i
                start_time = times[start_idx]
                
                recovery_idx = -1
                max_spike_time = start_time + np.timedelta64(int(self.spike_lookahead_mins * 60), 's')
                k = i + 1
                
                while k < n and times[k] <= max_spike_time:
                    val = fuels[k]
                    epsilon = max(3.0, 0.01 * baseline_before) # Ngưỡng phục hồi khắt khe hơn cho spike (phải về sát baseline)
                    if not np.isnan(val) and abs(val - baseline_before) <= epsilon:
                        recovery_idx = k
                        break
                    k += 1
                
                if recovery_idx != -1:
                    # Là SPIKE!
                    window_size = 5
                    after_window_end = min(n, recovery_idx + window_size)
                    
                    valid_future = []
                    recovery_val = fuels[recovery_idx]
                    for future_k in range(recovery_idx, after_window_end):
                        if not np.isnan(fuels[future_k]) and abs(fuels[future_k] - recovery_val) < self.spike_threshold:
                            valid_future.append(fuels[future_k])
                    
                    if len(valid_future) > 0:
                        baseline_after = np.median(valid_future)
                    else:
                        baseline_after = recovery_val
                        
                    total_time_diff = float((times[recovery_idx] - start_time) / np.timedelta64(1, 's'))
                    if total_time_diff <= 0: 
                        total_time_diff = 1.0
                    
                    for step in range(start_idx, recovery_idx):
                        ratio = float((times[step] - start_time) / np.timedelta64(1, 's')) / total_time_diff
                        interpolated_val = baseline_before + ratio * (baseline_after - baseline_before)
                        cleaned[step] = interpolated_val
                        anomaly_type[step] = 'SPIKE'
                        
                    anomaly_type[recovery_idx] = 'RECOVERY'
                    i = recovery_idx
                    continue
                    
            i += 1
                
        df['FuelAnomalyType'] = anomaly_type
        df['CleanedFuel'] = cleaned
        return df
