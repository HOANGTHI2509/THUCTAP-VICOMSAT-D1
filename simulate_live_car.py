import os
import sys
import time
import requests
import argparse
import pandas as pd
from datetime import datetime

sys.path.append(os.path.abspath(os.path.dirname(__file__)))
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if sys.stderr and hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from src.core.filters.kalman_traditional import BoLocKalmanTieuChuan1D

API_URL = "http://localhost:8000/api/push"

AVAILABLE_DATASETS = [
    {
        "id": "1",
        "vehicle_id": "21H-03221",
        "name": "Xe bồn 21H-03221 (Lào Cai - TienXuLy)",
        "file": os.path.join("TienXuLy", "21H-03221_processed.csv"),
        "capacity": 850.0,
    },
    {
        "id": "2",
        "vehicle_id": "24H-04650",
        "name": "Xe tải 24H-04650 (Dữ liệu TienXuLy)",
        "file": os.path.join("TienXuLy", "24H-04650_processed.csv"),
        "capacity": 500.0,
    },
    {
        "id": "3",
        "vehicle_id": "29E-45520",
        "name": "Xe 29E-45520 (Dữ liệu TienXuLy)",
        "file": os.path.join("TienXuLy", "29E-45520_processed.csv"),
        "capacity": 200.0,
    },
    {
        "id": "4",
        "vehicle_id": "90H-03494",
        "name": "Xe 90H-03494 (Dữ liệu TienXuLy)",
        "file": os.path.join("TienXuLy", "90H-03494_processed.csv"),
        "capacity": 250.0,
    },
]


def load_dataset(file_path: str) -> pd.DataFrame:
    """Đọc dữ liệu từ file Excel hoặc CSV."""
    if file_path.endswith(".xlsx") or file_path.endswith(".xls"):
        return pd.read_excel(file_path)
    elif file_path.endswith(".csv"):
        # Đọc CSV hỗ trợ cả dấu phẩy thập phân
        df = pd.read_csv(file_path)
        return df
    else:
        raise ValueError(f"Không hỗ trợ định dạng file: {file_path}")


def _stream_all_cars_concurrent():
    import threading
    print("\n" + "=" * 75)
    print(f"🚀 KHỞI ĐỘNG PHÁT LUỒNG SONG SONG (MULTI-THREADING) CHO {len(AVAILABLE_DATASETS)} XE ĐỒNG THỜI")
    print("=" * 75)
    threads = []
    for ds in AVAILABLE_DATASETS:
        t = threading.Thread(target=_stream_single_car, args=(ds,), name=f"Thread-{ds['vehicle_id']}")
        threads.append(t)
        t.start()
    for t in threads:
        t.join()


def simulate(choice_id: str = None):
    print("=" * 75)
    print("🚗 HỆ THỐNG PHÁT LUỒNG STREAMING TELEMETRY LÊN API VÀ WEB APP")
    print("=" * 75)

    if choice_id in ["0", "all", "--all"]:
        _stream_all_cars_concurrent()
        return

    selected = None
    if choice_id:
        for ds in AVAILABLE_DATASETS:
            if ds["id"] == choice_id or ds["vehicle_id"] == choice_id:
                selected = ds
                break

    if not selected:
        print("\nVui lòng chọn xe để phát dữ liệu hành trình:")
        for ds in AVAILABLE_DATASETS:
            print(f"  [{ds['id']}] {ds['name']} -> Biển số: {ds['vehicle_id']}")
        print("  [0] PHÁT ĐỒNG THỜI TẤT CẢ CÁC XE (Multi-threading song song)")
        
        try:
            val = input("\n👉 Nhập lựa chọn của bạn (1/2/3 hoặc 0 để chạy đồng thời tất cả): ").strip()
            if not val:
                val = "1"
        except Exception:
            val = "1"

        if val == "0" or val.lower() in ["all", "--all"]:
            _stream_all_cars_concurrent()
            return

        for ds in AVAILABLE_DATASETS:
            if ds["id"] == val:
                selected = ds
                break
        if not selected:
            selected = AVAILABLE_DATASETS[0]

    _stream_single_car(selected)


def _stream_single_car(selected: dict):
    vehicle_id = selected["vehicle_id"]
    file_path = selected["file"]
    capacity = selected["capacity"]

    if not os.path.exists(file_path):
        print(f"❌ Không tìm thấy file: {file_path}")
        return

    print("\n" + "-" * 75)
    print(f"📡 ĐANG KẾT NỐI & PHÁT DỮ LIỆU XE: {vehicle_id}")
    print(f"📁 Tệp nguồn: {os.path.basename(file_path)}")
    print("-" * 75)

    # 1. Reset state của xe trên Server
    try:
        requests.post(f"http://localhost:8000/api/v1/vehicles/{vehicle_id}/reset-state", timeout=2.0)
    except Exception:
        pass

    df = load_dataset(file_path)
    print(f"📊 Tổng số điểm đo: {len(df):,} điểm.")
    print("🟢 Đang truyền dữ liệu qua AI & Lọc Kalman...")

    first_val = 100.0
    for col in ["FuelLevel", "fuel_level", "RawFuel"]:
        if col in df.columns and pd.notnull(df.iloc[0][col]):
            first_val = float(str(df.iloc[0][col]).replace(",", "."))
            break

    kf_traditional = BoLocKalmanTieuChuan1D(trang_thai_ban_dau=first_val, nhieu_do_luong=9.0, nhieu_qua_trinh=1.0)

    for idx, row in df.iterrows():
        # Trích xuất thời gian
        time_str = datetime.now().isoformat()
        for col in ["FuelTime", "fuel_time", "Time", "timestamp"]:
            if col in row and pd.notnull(row[col]):
                time_str = str(row[col])
                break

        # Trích xuất mức xăng thô
        raw_fuel = 0.0
        for col in ["FuelLevel", "fuel_level", "RawFuel"]:
            if col in row and pd.notnull(row[col]):
                raw_fuel = float(str(row[col]).replace(",", "."))
                break

        # Trích xuất vận tốc
        speed = 0.0
        for col in ["Speed", "speed", "Vận tốc"]:
            if col in row and pd.notnull(row[col]):
                speed = float(str(row[col]).replace(",", "."))
                break

        # Tọa độ GPS & Địa chỉ
        lat = 21.0285
        lng = 105.8542
        for col in ["Lat", "lat", "Latitude"]:
            if col in row and pd.notnull(row[col]):
                lat = float(str(row[col]).replace(",", "."))
                break
        for col in ["Lng", "lng", "Longitude"]:
            if col in row and pd.notnull(row[col]):
                lng = float(str(row[col]).replace(",", "."))
                break

        # Tự động sửa lỗi đảo ngược Lat/Lng trong các file CSV
        if lat > 50 and lng < 50:
            lat, lng = lng, lat

        address = str(row.get("Address", str(row.get("address", "Hà Nội"))))

        # Tính toán đối chứng Kalman truyền thống
        kalman_std = kf_traditional.cap_nhat(gia_tri_do=raw_fuel, ty_le_dt=1.0)

        payload = {
            "time": time_str,
            "vehicle_id": vehicle_id,
            "raw": raw_fuel,
            "kalman": round(kalman_std, 2),
            "speed": speed,
            "lat": lat,
            "lng": lng,
            "address": address,
            "capacity_est": capacity,
        }

        try:
            res = requests.post(API_URL, json=payload, timeout=2.0)
            if res.status_code == 200:
                res_data = res.json()
                if idx % 15 == 0 or idx < 3:
                    q_len = res_data.get("queue_size", 0)
                    print(f"[{time_str}] Xe: {vehicle_id} | Raw: {raw_fuel:>6.1f}L | Std-Kalman: {kalman_std:>6.1f}L | Queue: {q_len} -> Enqueued OK")
            else:
                print(f"⚠️ API trả về mã lỗi: {res.status_code}")
        except Exception as e:
            print(f"❌ Mất kết nối tới API: {e}")
            break

        time.sleep(0.12)

    print("-" * 75)
    print(f"✅ HOÀN TẤT TRUYỀN DỮ LIỆU CHO XE {vehicle_id}!\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phát luồng streaming dữ liệu xe lên Web App")
    parser.add_argument("--car", type=str, default=None, help="Chọn xe (1: 24H-04650, 2: 29E-45520, 3: 90H-03494)")
    args = parser.parse_args()
    simulate(args.car)
