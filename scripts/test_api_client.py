import time
from datetime import datetime, timedelta
from fastapi.testclient import TestClient

from src.service.api import app

client = TestClient(app)

def run_demo():
    print("=" * 70)
    print("1. KIỂM TRA SỨC KHỎE DỊCH VỤ (GET /api/v1/health)")
    print("=" * 70)
    health = client.get("/api/v1/health").json()
    print(f"Health Status: {health}")
    print()

    print("=" * 70)
    print("2. MÔ PHỎNG STREAMING TỪNG ĐIỂM (POST /api/v1/fuel/clean-point)")
    print("   Kịch bản: Xe 29E-45520 xuất phát -> tăng tốc -> tiêu hao xăng trên đường ")
    print("=" * 70)
    
    mock_data = [
        {'fuel_level': 105.0, 'speed': 45.0, 'distance_meters': 350.0},
        {'fuel_level': 105.4, 'speed': 60.0, 'distance_meters': 2000.0},
        {'fuel_level': 104.8, 'speed': 58.0, 'distance_meters': 1800.0},
        {'fuel_level': 103.9, 'speed': 62.0, 'distance_meters': 2100.0},
        {'fuel_level': 0.0,   'speed': 55.0, 'distance_meters': 2000.0},
        {'fuel_level': 103.1, 'speed': 65.0, 'distance_meters': 2300.0},
        {'fuel_level': 102.5, 'speed': 63.0, 'distance_meters': 2200.0},
    ]

    base_time = datetime(2026, 8, 27, 8, 0, 0)
    for i, pt in enumerate(mock_data):
        t = base_time + timedelta(minutes=i * 5)
        payload = {
            "VehicleID": "29E-45520",
            "FuelTime": t.isoformat(),
            "FuelLevel": pt["fuel_level"],
            "Speed": pt["speed"],
            "DistanceMeters": pt["distance_meters"],
            "CapacityEst": 200.0,
            "NoiseSigmaLiters": 0.8,
        }
        res = client.post("/api/v1/fuel/clean-point", json=payload).json()
        print(f"[{res['FuelTime']}] Raw: {res['RawFuel']:>6.2f}L | Clean: {res['CleanFuel']:>6.2f}L | State: {res['AI_State']:<18} | Conf: {res['Confidence']:>5.1f}% | Latency: {res['LatencyMs']}ms")

    print()
    print("=" * 70)
    print("3. DANH SÁCH XE ĐANG HOẠT ĐỘNG TRONG BỘ NHỚ (GET /api/v1/vehicles)")
    print("=" * 70)
    vehicles = client.get("/api/v1/vehicles").json()
    for v in vehicles:
        print(f"-> Xe: {v['vehicle_id']} | Dung tích: {v['capacity_est']}L | Tổng điểm đã lọc: {v['total_points']} | Mức sạch cuối: {v['last_clean_fuel']}L | Trạng thái cuối: {v['last_state']}")

if __name__ == "__main__":
    run_demo()
