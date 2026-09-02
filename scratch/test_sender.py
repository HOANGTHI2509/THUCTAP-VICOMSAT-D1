import time
import requests
import random
from datetime import datetime, timedelta

API_URL = "http://localhost:8000/api/push"

def simulate_data_stream():
    print("Bat dau ban du lieu gia lap sang he thong Realtime...")
    print("Nhan Ctrl+C de dung.\n")
    
    current_time = datetime.now() - timedelta(days=2) # Giả lập bắt đầu từ 2 ngày trước để có lịch sử
    base_fuel = 200.0
    
    while True:
        # Nhảy thời gian 5 phút mỗi nhịp bắn giả lập
        current_time += timedelta(minutes=5) 
        time_str = current_time.strftime("%Y-%m-%d %H:%M:%S")
        
        if random.random() < 0.02:
            base_fuel += random.choice([-20, 30])
        
        raw_fuel = base_fuel + random.uniform(-5.0, 5.0) 
        kalman = base_fuel + random.uniform(-2.0, 2.0)
        adaptive = base_fuel + random.uniform(-0.5, 0.5)
        ai_enhanced = base_fuel
        
        payload = {
            "time": time_str,
            "raw": round(raw_fuel, 2),
            "kalman": round(kalman, 2),
            "adaptive": round(adaptive, 2),
            "ai_enhanced": round(ai_enhanced, 2)
        }
        
        try:
            response = requests.post(API_URL, json=payload)
            print(f"[{time_str}] Bắn dữ liệu Raw={raw_fuel:.1f}L -> {response.status_code}")
        except Exception as e:
            print(f"[{time_str}] Không thể kết nối tới API: {e}")
            
        time.sleep(0.5)

if __name__ == "__main__":
    simulate_data_stream()
