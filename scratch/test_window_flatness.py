import os
import sys
sys.path.insert(0, os.path.abspath("."))
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import numpy as np

def check_flatness(samples, times, jitter=0.8):
    """
    Kết hợp độ dốc theo thời gian và chênh lệch mức đại diện giữa hai phần cửa sổ.
    Ngưỡng chênh lệch dùng lít; ngưỡng độ dốc dùng lít/phút.
    """
    k = len(samples)
    if k < 4:
        return False, 0.0, 0.0
    
    mid = k // 2
    h1 = samples[:mid]
    h2 = samples[mid:]
    
    med1 = float(np.median(h1))
    med2 = float(np.median(h2))
    diff_liters = abs(med2 - med1)
    
    # Thời lượng giữa tâm 2 nửa cửa sổ
    dur_min = (times[-1] - times[0]).total_seconds() / 60.0
    if dur_min <= 0:
        return False, diff_liters, 0.0
        
    slope_l_per_min = diff_liters / max(dur_min * 0.5, 1.0)
    
    # Ngưỡng: không còn xu hướng đáng kể so với độ nhiễu
    max_diff_liters = max(jitter * 1.0, 1.8)
    max_slope = max(jitter * 0.15, 0.25)
    
    spread = max(samples) - min(samples)
    max_spread = max(jitter * 1.5, 2.5)
    
    is_flat = (diff_liters <= max_diff_liters) and (slope_l_per_min <= max_slope) and (spread <= max_spread)
    return is_flat, diff_liters, slope_l_per_min

print("Helper defined.")
