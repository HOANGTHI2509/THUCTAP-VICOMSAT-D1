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

API_URL = "http://localhost:8000/api/push"

AVAILABLE_DATASETS = [
    {
        "id": "1",
        "vehicle_id": "24H-04650",
        "name": "Xe tải 24H-04650 (Dữ liệu THÔ nguyên bản fulltt - Chưa tiền xử lý)",
        "file": os.path.join("fulltt", "24H-04650_da_gop.xlsx"),
        "capacity": 500.0,
        "start_mov": 166,  # Bắt đầu từ lúc xe nổ máy lăn bánh > 20 km/h
    },
    {
        "id": "2",
        "vehicle_id": "29E-45520",
        "name": "Xe tải 29E-45520 (Dữ liệu THÔ nguyên bản fulltt - Chưa tiền xử lý)",
        "file": os.path.join("fulltt", "29E-45520_da_gop.xlsx"),
        "capacity": 200.0,
        "start_mov": 99,   # Bắt đầu từ lúc xe nổ máy lăn bánh
    },
    {
        "id": "3",
        "vehicle_id": "90H-03494",
        "name": "Xe tải 90H-03494 (Dữ liệu THÔ nguyên bản fulltt - Chưa tiền xử lý)",
        "file": os.path.join("fulltt", "90H-03494_da_gop.xlsx"),
        "capacity": 250.0,
        "start_mov": 184,  # Bắt đầu từ lúc xe nổ máy lăn bánh
    },
    {
        "id": "4",
        "vehicle_id": "35H-09245",
        "name": "Xe tải 35H-09245 (Dữ liệu THÔ nguyên bản fulltt - Chưa tiền xử lý)",
        "file": os.path.join("fulltt", "35H-09245_da_gop.xlsx"),
        "capacity": 300.0,
        "start_mov": 100,
    },
    {
        "id": "5",
        "vehicle_id": "21H-03221",
        "name": "Xe bồn 21H-03221 (Dữ liệu đối chứng TienXuLy)",
        "file": os.path.join("TienXuLy", "21H-03221_processed.csv"),
        "capacity": 850.0,
        "start_mov": 195,
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


def simulate(choice_id: str = None, start_index: int = None, speed_factor: float = 1.0):
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

    _stream_single_car(selected, start_index=start_index, speed_factor=speed_factor)


def _stream_single_car(selected: dict, start_index: int = None, speed_factor: float = 1.0):
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

    # 1. Reset state của xe và dọn sạch buffer trên Server
    try:
        requests.post(f"http://localhost:8000/api/v1/vehicles/{vehicle_id}/reset-state", timeout=2.0)
    except Exception:
        pass

    df = load_dataset(file_path)
    total_pts = len(df)
    
    start_row = start_index if start_index is not None else selected.get("start_mov", 0)
    if start_row > 0 and start_row < total_pts:
        df = df.iloc[start_row:].reset_index(drop=True)
        print(f"📊 Bắt đầu phát từ điểm đo {start_row}/{total_pts} (đoạn xe hoạt động thực tế)")
    else:
        print(f"📊 Tổng số điểm đo: {total_pts:,} điểm.")
        
    print("🟢 Đang truyền dữ liệu xe vào Bộ lọc AI...")

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

        payload = {
            "time": time_str,
            "vehicle_id": vehicle_id,
            "raw": raw_fuel,
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
                clean_fuel = float(res_data.get("clean_fuel", raw_fuel))
                ai_st = str(res_data.get("ai_state", ""))
                if idx % 15 == 0 or idx < 3:
                    state_info = f" | TT: {ai_st}" if ai_st else ""
                    print(f"[{time_str}] Xe: {vehicle_id} | Vận tốc: {speed:>4.1f}km/h | Raw: {raw_fuel:>6.1f}L | Lọc AI: {clean_fuel:>6.1f}L{state_info} -> OK")
            else:
                print(f"⚠️ API trả về mã lỗi: {res.status_code}")
        except Exception as e:
            print(f"❌ Mất kết nối tới API: {e}")
            break

        time.sleep(max(0.01, 0.12 / speed_factor))

    print("-" * 75)
    print(f"✅ HOÀN TẤT TRUYỀN DỮ LIỆU CHO XE {vehicle_id}!\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phát luồng streaming dữ liệu xe lên Web App")
    parser.add_argument("--car", type=str, default=None, help="Chọn xe (1: 24H-04650 [Thô fulltt], 2: 29E-45520 [Thô fulltt], 3: 90H-03494, 4: 35H-09245, 5: 21H-03221)")
    parser.add_argument("--start", type=int, default=None, help="Chỉ số dòng bắt đầu phát (mặc định tự nhảy đến đoạn xe hoạt động)")
    parser.add_argument("--speed", type=float, default=1.0, help="Tốc độ phát mô phỏng (1.0 = chuẩn ~8 điểm/giây, 2.0 = nhanh gấp đôi)")
    args = parser.parse_args()
    simulate(args.car, start_index=args.start, speed_factor=args.speed)
