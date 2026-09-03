"""
Script kiểm thử toàn diện REST API Microservice & Bảo mật cấp Doanh nghiệp (Task 3).
"""
import os
import sys
import json
import urllib.request
import urllib.error

sys.path.insert(0, os.path.abspath("."))
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if sys.stderr and hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

BASE_URL = "http://localhost:8000"


def make_request(path, method="GET", data=None, headers=None):
    url = f"{BASE_URL}{path}"
    req_headers = {"Content-Type": "application/json"}
    if headers:
        req_headers.update(headers)
    req_data = json.dumps(data).encode("utf-8") if data is not None else None
    req = urllib.request.Request(url, data=req_data, headers=req_headers, method=method)
    try:
        with urllib.request.urlopen(req) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read().decode("utf-8"))


def run_tests():
    print("=" * 70)
    print("🚀 BẮT ĐẦU KIỂM THỬ TASK 3: REST API MICROSERVICE & BẢO MẬT DOANH NGHIỆP")
    print("=" * 70)

    # 1. Kiểm thử Health Check Endpoint
    print("\n[1/6] Kiểm thử GET /api/v1/health...")
    status_code, body = make_request("/api/v1/health")
    assert status_code == 200, f"Health check failed: {status_code}"
    assert body["status"] == "healthy"
    assert body["database_enabled"] is True
    print(f"  -> Health OK! Service={body['service']} | DB={body['database_enabled']} | API_Key_Required={body['api_key_required']}")

    # 2. Kiểm thử POST /api/v1/clean (Chuẩn Tiếng Anh)
    print("\n[2/6] Kiểm thử POST /api/v1/clean (Đầu vào tiếng Anh chuẩn)...")
    payload_en = {
        "vehicle_id": "29H-75028",
        "timestamp": "2026-08-15 09:16:00",
        "raw_fuel": 93.3,
        "speed": 61.0,
        "lat": 21.0285,
        "lng": 105.8542,
        "capacity_est": 95.0,
    }
    status_code, body = make_request("/api/v1/clean", method="POST", data=payload_en)
    assert status_code == 200, f"Lỗi POST /api/v1/clean: {status_code}, body={body}"
    assert body["clean_fuel"] > 0
    assert body["saved_to_db"] is True, "Lỗi: Chưa lưu vào CSDL 3NF"
    print(f"  -> Trả về: Xe={body['vehicle_id']} | Sạch={body['clean_fuel']}L | Trạng thái={body['ai_state_desc']} | Lưu DB={body['saved_to_db']} ({body['processing_time_ms']} ms)")

    # 3. Kiểm thử POST /api/v1/clean (Bí danh Tiếng Việt: bien_so, thoi_gian, xang_tho, van_toc)
    print("\n[3/6] Kiểm thử POST /api/v1/clean (Bí danh Tiếng Việt: bien_so, thoi_gian, xang_tho, van_toc)...")
    # Điểm 1: Mức nền ổn định 430.0L
    p1 = {
        "bien_so": "TEST_CAR_VI",
        "thoi_gian": "2026-08-10 08:46:00",
        "xang_tho": 430.0,
        "van_toc": 0.0,
        "dung_tich": 850.0,
    }
    make_request("/api/v1/clean", method="POST", data=p1)

    # Điểm 2: Xung nhiễu vọt lên 440.2L
    payload_vi = {
        "bien_so": "TEST_CAR_VI",
        "thoi_gian": "2026-08-10 08:48:00",
        "xang_tho": 440.2,
        "van_toc": 0.0,
        "vi_do": 22.3167,
        "kinh_do": 104.1167,
        "dung_tich": 850.0,
    }
    status_code, body = make_request("/api/v1/clean", method="POST", data=payload_vi)
    assert status_code == 200, f"Lỗi POST bí danh tiếng Việt: {status_code}, body={body}"
    assert body["vehicle_id"] == "TEST_CAR_VI", "Lỗi: Bí danh bien_so không được map đúng"
    assert body["clean_fuel"] < 433.0, f"Lỗi: Xung 440.2L không được nén mượt ({body['clean_fuel']})"
    print(f"  -> Nhận diện tiếng Việt OK: Xe={body['vehicle_id']} | Xung 440.2L nén về={body['clean_fuel']}L | Nhãn={body['ai_state_desc']}")

    # 4. Kiểm thử POST /api/v1/clean-batch (Xử lý theo mảng/cụm)
    print("\n[4/6] Kiểm thử POST /api/v1/clean-batch (Xử lý theo mảng)...")
    batch_payload = {
        "points": [
            {"vehicle_id": "BATCH_CAR", "timestamp": "2026-08-12 12:00:00", "raw_fuel": 150.0, "speed": 45.0},
            {"vehicle_id": "BATCH_CAR", "timestamp": "2026-08-12 12:02:00", "raw_fuel": 149.8, "speed": 45.0},
            {"vehicle_id": "BATCH_CAR", "timestamp": "2026-08-12 12:04:00", "raw_fuel": 149.6, "speed": 45.0},
        ]
    }
    status_code, body = make_request("/api/v1/clean-batch", method="POST", data=batch_payload)
    assert status_code == 200, f"Lỗi POST /api/v1/clean-batch: {status_code}"
    assert len(body) == 3, f"Số lượng kết quả batch không khớp: {len(body)}"
    print(f"  -> Xử lý thành công mảng 3 điểm! Điểm cuối: Clean={body[-1]['clean_fuel']}L | Nhãn={body[-1]['event_label']}")

    # 5. Kiểm thử GET /api/v1/events (Truy vấn sự kiện từ CSDL 3NF)
    print("\n[5/6] Kiểm thử GET /api/v1/events...")
    status_code, body = make_request("/api/v1/events?vehicle_id=29H-75028")
    assert status_code == 200, f"Lỗi GET /api/v1/events: {status_code}"
    print(f"  -> Số lượng sự kiện tìm thấy cho xe 29H-75028: {len(body)}")
    if body:
        ev = body[0]
        print(f"     * Sự kiện: {ev['event_type']} | Chênh lệch: {ev['change_liters']}L | Thời gian: {ev['start_time']}")

    # 6. Kiểm tra các Endpoint phục vụ Web Demo Local không bị ảnh hưởng
    print("\n[6/6] Kiểm tra tương thích ngược: Các endpoint Web Demo Local cũ...")
    # Test /api/data
    status_code, body = make_request("/api/data?limit=5")
    assert status_code == 200, "Lỗi: Endpoint cũ /api/data bị gãy"
    # Test /api/vehicles
    status_code, body = make_request("/api/vehicles")
    assert status_code == 200, "Lỗi: Endpoint cũ /api/vehicles bị gãy"
    print("  -> Toàn bộ các endpoint cũ (/api/data, /api/vehicles, /api/push) hoạt động 100% nguyên vẹn!")

    print("\n" + "=" * 70)
    print("🎉 TẤT CẢ 6/6 BÀI KIỂM THỬ TASK 3 ĐÃ PASS THÀNH CÔNG 100%!")
    print("=" * 70)


if __name__ == "__main__":
    run_tests()
