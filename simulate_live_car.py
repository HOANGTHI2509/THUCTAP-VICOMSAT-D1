import os
import sys
import time
import requests
import argparse
import pandas as pd
from datetime import datetime

sys.path.append(r"D:\THUCTAP_VICOMSAT")
from src.core.filters.kalman_traditional import BoLocKalmanTieuChuan1D
from src.core.filters.kalman_adaptive import BoLocKalmanThichNghi1D

API_URL = "http://localhost:8000/api/push"

AVAILABLE_DATASETS = [
    {
        "id": "1",
        "vehicle_id": "24H-04650",
        "name": "Xe tải 24H-04650 (Dữ liệu tổng hợp fulltt)",
        "file": r"D:\THUCTAP_VICOMSAT\fulltt\24H-04650_da_gop.xlsx",
        "capacity": 500.0,
    },
    {
        "id": "2",
        "vehicle_id": "29C-92841",
        "name": "Xe cao tốc Ninh Bình (2026-08-27T00-48_export.csv)",
        "file": r"D:\THUCTAP_VICOMSAT\2026-08-27T00-48_export.csv",
        "capacity": 500.0,
    },
    {
        "id": "3",
        "vehicle_id": "29H-77123",
        "name": "Xe thử nghiệm leo dốc (TEST DO DOC.csv)",
        "file": r"D:\THUCTAP_VICOMSAT\TEST DO DOC.csv",
        "capacity": 500.0,
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


def simulate(choice_id: str = None):
    print("=" * 75)
    print("🚗 HỆ THỐNG PHÁT LUỒNG STREAMING TELEMETRY LÊN API VÀ WEB APP")
    print("=" * 75)

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
        print("  [0] Tự động chạy lần lượt các xe")
        
        try:
            val = input("\n👉 Nhập lựa chọn của bạn (1/2/3 hoặc nhấn Enter để chọn 1): ").strip()
            if not val:
                val = "1"
        except Exception:
            val = "1"

        if val == "0":
            for ds in AVAILABLE_DATASETS:
                _stream_single_car(ds)
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
    kf_adaptive = BoLocKalmanThichNghi1D(trang_thai_ban_dau=first_val, capacity=capacity, r_co_ban=9.0, nhieu_qua_trinh=0.2)

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

        # Tính toán đối chứng
        kalman_std = kf_traditional.cap_nhat(gia_tri_do=raw_fuel, ty_le_dt=1.0)
        kalman_adapt = kf_adaptive.cap_nhat(
            gia_tri_do=raw_fuel,
            ty_le_dt=1.0,
            trang_thai_chuyen_dong=1 if speed > 0 else 0,
            van_toc=speed,
            gia_toc=0.0,
            rolling_std=1.0,
        )

        payload = {
            "time": time_str,
            "vehicle_id": vehicle_id,
            "raw": raw_fuel,
            "kalman": round(kalman_std, 2),
            "adaptive": round(kalman_adapt, 2),
            "speed": speed,
            "lat": lat,
            "lng": lng,
            "address": address,
        }

        try:
            res = requests.post(API_URL, json=payload, timeout=2.0)
            if res.status_code == 200:
                res_data = res.json()
                if idx % 15 == 0 or idx < 3:
                    clean_val = res_data.get("clean", 0.0)
                    ai_state = res_data.get("state", "STABLE_JITTER")
                    print(f"[{time_str}] Xe: {vehicle_id} | Raw: {raw_fuel:>6.1f}L | AI-Kalman: {clean_val:>6.1f}L | State: {ai_state:<18} -> Web App OK")
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
    parser.add_argument("--car", type=str, default=None, help="Chọn xe (1: 24H-04650, 2: 29C-92841, 3: 29H-77123)")
    args = parser.parse_args()
    simulate(args.car)
