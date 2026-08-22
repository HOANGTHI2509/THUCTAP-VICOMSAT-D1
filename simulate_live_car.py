import pandas as pd
import requests
import time
import glob
import os
import sqlite3
from datetime import datetime, timedelta
import sys

# Import trực tiếp các bộ lọc thật từ source code
sys.path.append(r"D:\THUCTAP_VICOMSAT")
from src.core.filters.kalman_traditional import BoLocKalmanTieuChuan1D
from src.core.filters.kalman_adaptive import BoLocKalmanThichNghi1D

API_URL = "http://localhost:8000/api/push"
FULLTT_DIR = r"D:\THUCTAP_VICOMSAT\fulltt"

def simulate():
    print("=== BẮT ĐẦU BẮN DỮ LIỆU TỪ FULLTT ===")
    
    # Xóa dữ liệu cũ trong DB để làm mới biểu đồ Live Dashboard
    try:
        conn = sqlite3.connect("fuel_data.db")
        c = conn.cursor()
        c.execute('DELETE FROM fuel_records')
        conn.commit()
        conn.close()
        print("Đã xóa trắng lịch sử dữ liệu cũ trên Database.")
    except Exception as e:
        print(f"Không thể xóa DB cũ: {e}")
        
    # Lấy danh sách các file excel
    excel_files = glob.glob(os.path.join(FULLTT_DIR, "*_da_gop.xlsx"))
    
    if not excel_files:
        print(f"Không tìm thấy file nào trong {FULLTT_DIR}")
        return

    # Để demo nhanh, ta chọn file đầu tiên
    target_file = excel_files[0]
    
    # Lấy tên xe từ tên file (Ví dụ: 24H-04650_da_gop.xlsx -> 24H-04650)
    filename = os.path.basename(target_file)
    vehicle_id = filename.replace("_da_gop.xlsx", "")
    print(f"Đang đọc file: {filename} - Nhận diện xe: {vehicle_id}")
    
    df = pd.read_excel(target_file)
    
    print(f"Tổng số bản ghi: {len(df)}. Bắt đầu bắn lên Server (10 điểm / giây)...")
    
    # KHỞI TẠO BỘ LỌC THẬT (REAL FILTERS)
    first_val = float(str(df.iloc[0]['FuelLevel']).replace(',', '.')) if pd.notnull(df.iloc[0]['FuelLevel']) else 0.0
    
    kf_traditional = BoLocKalmanTieuChuan1D(trang_thai_ban_dau=first_val, nhieu_do_luong=9.0, nhieu_qua_trinh=1.0)
    kf_adaptive = BoLocKalmanThichNghi1D(trang_thai_ban_dau=first_val, capacity=200.0, r_co_ban=9.0, nhieu_qua_trinh=0.2)
    
    for idx, row in df.iterrows():
        # Lấy dữ liệu từ file Excel
        # Lấy thời gian gốc từ file thay vì thời gian hiện tại
        time_str = str(row['FuelTime'])
        
        raw_fuel = float(str(row['FuelLevel']).replace(',', '.')) if pd.notnull(row['FuelLevel']) else 0.0
        speed = float(str(row['Speed']).replace(',', '.')) if pd.notnull(row['Speed']) else 0.0
        lat = float(str(row['Lat']).replace(',', '.')) if pd.notnull(row['Lat']) else 0.0
        lng = float(str(row['Lng']).replace(',', '.')) if pd.notnull(row['Lng']) else 0.0
        address = str(row['Address']) if pd.notnull(row['Address']) else ""
        
        # SỬ DỤNG BỘ LỌC THỰC TẾ (REAL KALMAN)
        # 1. Cập nhật Traditional Kalman (Random Forest line tạm mượn)
        kalman = kf_traditional.cap_nhat(gia_tri_do=raw_fuel, ty_le_dt=1.0)
        
        # 2. Cập nhật Adaptive Kalman
        # (Ở thực tế, nếu có Speed và gia tốc thì truyền vào để adaptive chuẩn hơn, ở đây giả lập tốc độ và noise)
        adaptive = kf_adaptive.cap_nhat(
            gia_tri_do=raw_fuel, 
            ty_le_dt=1.0, 
            trang_thai_chuyen_dong=1 if speed > 0 else 0,
            van_toc=speed,
            gia_toc=0.0,
            rolling_std=1.0 # Giá trị giả định để lọc hoạt động
        )
        
        # AI Enhanced hiện tại dùng chung Adaptive nếu không có buffer DataFrame
        ai_enhanced = adaptive 
        
        payload = {
            "time": time_str,
            "vehicle_id": vehicle_id,
            "raw": raw_fuel,
            "kalman": kalman,
            "adaptive": adaptive,
            "ai_enhanced": ai_enhanced,
            "speed": speed,
            "lat": lat,
            "lng": lng,
            "address": address
        }
        
        try:
            response = requests.post(API_URL, json=payload)
            if idx % 50 == 0:
                print(f"[{idx}/{len(df)}] Đã bắn: {time_str} - Tọa độ: ({lat}, {lng}) -> OK")
        except Exception as e:
            print(f"Lỗi khi gửi API: {e}")
            break
            
        # Nghỉ 0.1s để giả lập chạy thật, bạn có thể chỉnh nhanh hơn
        time.sleep(0.1)
        
    print("=== HOÀN TẤT ===")

if __name__ == "__main__":
    simulate()
