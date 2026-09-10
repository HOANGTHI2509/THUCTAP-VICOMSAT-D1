"""
Bo kiem thu toan dien cho tat ca cac Endpoint API (Vicomsat Fuel Denoising Service).
Chay duoc ca hai che do:
  1. Qua pytest: python -m pytest tests/test_all_apis_comprehensive.py -v
  2. Truc tiep qua terminal (kiem tra in-memory hoac live Docker container):
     python tests/test_all_apis_comprehensive.py [http://localhost:8000]
"""

import os
import sys
import time
from datetime import datetime, timedelta
import pytest

# Đảm bảo đường dẫn gốc repository luôn nằm trong sys.path khi chạy trực tiếp
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from fastapi.testclient import TestClient
from src.service.api import app

client = TestClient(app)


# ============================================================================
# 1. TEST SUITE: GET /api/v1/health
# ============================================================================
class TestHealthAPI:
    def test_health_check_returns_200_and_healthy(self):
        """Case 1.1: Health check tra ve HTTP 200 va status='healthy'."""
        resp = client.get("/api/v1/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "healthy"
        assert data["service"] == "vicomsat-fuel-cleaning-service"

    def test_health_check_payload_contract_fields(self):
        """Case 1.2: Kiem tra day du cac truong thong tin he thong."""
        resp = client.get("/api/v1/health")
        assert resp.status_code == 200
        data = resp.json()
        expected_keys = [
            "status",
            "service",
            "version",
            "model_loaded",
            "model_type",
            "active_vehicles_in_memory",
            "state_store",
            "database_enabled",
            "timestamp",
        ]
        for key in expected_keys:
            assert key in data, f"Thieu truong {key} trong phan hoi health check"
        assert isinstance(data["model_loaded"], bool)
        assert data["state_store"]["status"] == "healthy"


# ============================================================================
# 2. TEST SUITE: POST /api/v1/fuel/clean-point (Official Topic 1)
# ============================================================================
class TestCleanPointAPI:
    def test_clean_point_nominal_full_fields(self):
        """Case 2.1: Diem hop le voi day du cac truong tieu chuan."""
        payload = {
            "VehicleID": "29E-TEST-01",
            "FuelTime": "2026-08-27T10:00:00",
            "FuelLevel": 150.5,
            "Speed": 45.0,
            "Lat": 21.0285,
            "Lng": 105.8542,
            "CapacityEst": 200.0,
            "Address": "Hanoi, Vietnam",
        }
        resp = client.post("/api/v1/fuel/clean-point", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert data["VehicleID"] == "29E-TEST-01"
        assert data["RawFuel"] == 150.5
        assert isinstance(data["CleanFuel"], (int, float))
        assert "SignalState" in data
        assert "QualityFlag" in data
        assert "MotionState" in data
        assert data["LatencyMs"] >= 0

    def test_clean_point_minimal_required_fields(self):
        """Case 2.2: Diem chi truyen 3 truong bat buoc (VehicleID, FuelTime, FuelLevel)."""
        payload = {
            "VehicleID": "29E-TEST-02",
            "FuelTime": "2026-08-27T10:02:00",
            "FuelLevel": 140.0,
        }
        resp = client.post("/api/v1/fuel/clean-point", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert data["VehicleID"] == "29E-TEST-02"
        assert data["RawFuel"] == 140.0
        assert data["CleanFuel"] > 0

    def test_clean_point_zero_dropout_behavior(self):
        """Case 2.3: Truong hop cam bien rot ve 0.0 L -> kiem tra giu muc sach."""
        vid = "29E-TEST-ZERO"
        # Diem 1: Nguong binh thuong
        p1 = {"VehicleID": vid, "FuelTime": "2026-08-27T10:00:00", "FuelLevel": 180.0, "Speed": 30.0}
        resp1 = client.post("/api/v1/fuel/clean-point", json=p1)
        assert resp1.status_code == 200
        clean1 = resp1.json()["CleanFuel"]

        # Diem 2: Rot ve 0 L
        p2 = {"VehicleID": vid, "FuelTime": "2026-08-27T10:02:00", "FuelLevel": 0.0, "Speed": 30.0}
        resp2 = client.post("/api/v1/fuel/clean-point", json=p2)
        assert resp2.status_code == 200
        data2 = resp2.json()
        # CleanFuel khong duoc phep tut ve 0 theo raw ma phai duoc giu gan muc clean1
        assert data2["CleanFuel"] > 100.0
        assert "DROPOUT" in data2["QualityFlag"] or "HOLD" in data2["QualityFlag"] or data2["CleanFuel"] == clean1

    def test_clean_point_motion_state_moving(self):
        """Case 2.4: Xe di chuyển liên tục (Speed >= 5 km/h + GPS dịch chuyển >= 25m) -> MotionState = MOVING."""
        vid = "29E-TEST-MOVING-SEQ"
        for i in range(3):
            t = (datetime(2026, 8, 27, 10, 5, 0) + timedelta(minutes=2 * i)).isoformat()
            # Mỗi điểm dịch chuyển vĩ độ ~0.001 độ (~111m)
            payload = {
                "VehicleID": vid,
                "FuelTime": t,
                "FuelLevel": 160.0 - i * 0.2,
                "Speed": 55.0,
                "Lat": 21.0285 + i * 0.001,
                "Lng": 105.8542,
            }
            resp = client.post("/api/v1/fuel/clean-point", json=payload)
            assert resp.status_code == 200
        # Điểm thứ 3 đã đủ 3 điểm (enough_points) và GPS dịch chuyển -> MOVING
        assert resp.json()["MotionState"] == "MOVING"

    def test_clean_point_motion_state_low_motion(self):
        """Case 2.5: Xe dung yen (Speed <= 0.5) trong ban kinh nho -> LOW_MOTION."""
        vid = "29E-TEST-PARK"
        for i in range(4):
            t = (datetime(2026, 8, 27, 10, 0, 0) + timedelta(minutes=2 * i)).isoformat()
            p = {"VehicleID": vid, "FuelTime": t, "FuelLevel": 120.0, "Speed": 0.0, "Lat": 21.02, "Lng": 105.85}
            r = client.post("/api/v1/fuel/clean-point", json=p)
            assert r.status_code == 200
        assert r.json()["MotionState"] in ["LOW_MOTION", "UNCERTAIN"]

    def test_clean_point_validation_missing_vehicle_id(self):
        """Case 2.6: Loi validation khi thieu VehicleID -> HTTP 422."""
        payload = {
            "FuelTime": "2026-08-27T10:00:00",
            "FuelLevel": 150.0,
        }
        resp = client.post("/api/v1/fuel/clean-point", json=payload)
        assert resp.status_code == 422

    def test_clean_point_validation_invalid_fuel_time(self):
        """Case 2.7: Loi validation khi thoi gian sai dinh dang -> HTTP 422."""
        payload = {
            "VehicleID": "29E-ERR-TIME",
            "FuelTime": "invalid-datetime-string",
            "FuelLevel": 150.0,
        }
        resp = client.post("/api/v1/fuel/clean-point", json=payload)
        assert resp.status_code == 422

    def test_clean_point_validation_non_numeric_fuel(self):
        """Case 2.8: Loi validation khi muc nhien lieu la chuoi -> HTTP 422."""
        payload = {
            "VehicleID": "29E-ERR-FUEL",
            "FuelTime": "2026-08-27T10:00:00",
            "FuelLevel": "mot_tram_lit",
        }
        resp = client.post("/api/v1/fuel/clean-point", json=payload)
        assert resp.status_code == 422


# ============================================================================
# 3. TEST SUITE: POST /api/v1/fuel/clean-batch (Official Batch)
# ============================================================================
class TestCleanBatchAPI:
    def test_clean_batch_nominal_sequence(self):
        """Case 3.1: Goi batch voi chuoi 5 diem lien tiep theo thoi gian."""
        vid = "29E-BATCH-01"
        pts = []
        for i in range(5):
            t = (datetime(2026, 8, 27, 11, 0, 0) + timedelta(minutes=2 * i)).isoformat()
            pts.append({
                "VehicleID": vid,
                "FuelTime": t,
                "FuelLevel": 150.0 - i * 0.4,
                "Speed": 40.0,
            })
        payload = {"vehicle_id": vid, "points": pts}
        resp = client.post("/api/v1/fuel/clean-batch", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert data["vehicle_id"] == vid
        assert data["total_points"] == 5
        assert len(data["results"]) == 5
        assert data["total_latency_ms"] >= 0

    def test_clean_batch_out_of_order_auto_sorting(self):
        """Case 3.2: Batch bi dao nguoc thu tu thoi gian -> API tu dong sap xep."""
        vid = "29E-BATCH-SORT"
        t1 = "2026-08-27T12:00:00"
        t2 = "2026-08-27T12:02:00"
        t3 = "2026-08-27T12:04:00"
        # Gui thu tu t3 -> t1 -> t2
        pts = [
            {"VehicleID": vid, "FuelTime": t3, "FuelLevel": 140.0},
            {"VehicleID": vid, "FuelTime": t1, "FuelLevel": 142.0},
            {"VehicleID": vid, "FuelTime": t2, "FuelLevel": 141.0},
        ]
        payload = {"vehicle_id": vid, "points": pts}
        resp = client.post("/api/v1/fuel/clean-batch", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        results = data["results"]
        # Kiem tra ket qua da duoc sort theo thu tu t1 -> t2 -> t3
        assert results[0]["FuelTime"] == t1
        assert results[1]["FuelTime"] == t2
        assert results[2]["FuelTime"] == t3

    def test_clean_batch_empty_points(self):
        """Case 3.3: Batch rong khong co diem nao."""
        payload = {"vehicle_id": "29E-BATCH-EMPTY", "points": []}
        resp = client.post("/api/v1/fuel/clean-batch", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_points"] == 0
        assert data["results"] == []


# ============================================================================
# 4. TEST SUITE: POST /api/v1/clean & /api/v1/clean-batch (Enterprise Aliases)
# ============================================================================
class TestEnterpriseAPI:
    def test_enterprise_clean_english_aliases(self):
        """Case 4.1: Goi endpoint /api/v1/clean dung cac truong tieng Anh."""
        payload = {
            "vehicle_id": "29E-ENT-EN",
            "timestamp": "2026-08-27 10:00:00",
            "raw_fuel": 170.5,
            "speed": 35.0,
            "distance_m": 1200.0,
            "capacity_est": 250.0,
        }
        resp = client.post("/api/v1/clean", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert data["VehicleID"] == "29E-ENT-EN"
        assert data["RawFuel"] == 170.5
        assert "CleanFuel" in data
        assert "SignalState" in data

    def test_enterprise_clean_vietnamese_aliases(self):
        """Case 4.2: Goi endpoint /api/v1/clean dung bi danh tieng Viet (bien_so, thoi_gian, xang_tho)."""
        payload = {
            "bien_so": "29E-ENT-VN",
            "thoi_gian": "2026-08-27 10:15:00",
            "xang_tho": 165.2,
            "van_toc": 42.0,
            "dung_tich": 200.0,
        }
        resp = client.post("/api/v1/clean", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert data["VehicleID"] == "29E-ENT-VN"
        assert data["RawFuel"] == 165.2

    def test_enterprise_clean_batch(self):
        """Case 4.3: Goi /api/v1/clean-batch hang loat."""
        payload = {
            "points": [
                {"vehicle_id": "29E-EB-1", "timestamp": "2026-08-27 10:00:00", "raw_fuel": 100.0},
                {"vehicle_id": "29E-EB-2", "timestamp": "2026-08-27 10:00:00", "raw_fuel": 120.0},
            ]
        }
        resp = client.post("/api/v1/clean-batch", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 2
        assert data[0]["VehicleID"] == "29E-EB-1"
        assert data[1]["VehicleID"] == "29E-EB-2"


# ============================================================================
# 5. TEST SUITE: Vehicle State Management (/api/v1/vehicles)
# ============================================================================
class TestVehicleStateManagementAPI:
    def test_list_active_vehicles(self):
        """Case 5.1: Lay danh sach xe dang co context."""
        vid = "29E-ACTIVE-LIST"
        client.post("/api/v1/fuel/clean-point", json={
            "VehicleID": vid,
            "FuelTime": "2026-08-27T10:00:00",
            "FuelLevel": 110.0,
        })
        resp = client.get("/api/v1/vehicles")
        assert resp.status_code == 200
        vehicles = resp.json()
        assert isinstance(vehicles, list)
        vids = [v.get("vehicle_id") for v in vehicles]
        assert vid in vids

    def test_reset_vehicle_state_existing(self):
        """Case 5.2: Reset state cua mot xe thanh cong."""
        vid = "29E-RESET-01"
        client.post("/api/v1/fuel/clean-point", json={
            "VehicleID": vid,
            "FuelTime": "2026-08-27T10:00:00",
            "FuelLevel": 110.0,
        })
        resp = client.post(f"/api/v1/vehicles/{vid}/reset-state")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "success"
        assert data["vehicle_id"] == vid

    def test_reset_vehicle_state_unknown(self):
        """Case 5.3: Reset state xe chua ton tai van tra ve thanh cong em thuan."""
        resp = client.post("/api/v1/vehicles/UNKNOWN-VEHICLE/reset-state")
        assert resp.status_code == 200
        assert resp.json()["status"] == "success"


# ============================================================================
# 6. TEST SUITE: Live Demo & Streaming APIs (/api/push, /api/data, /api/fleet)
# ============================================================================
class TestLiveStreamingDemoAPI:
    def test_push_live_point_and_query_data(self):
        """Case 6.1: Day diem qua /api/push va truy van lai qua /api/data."""
        vid = "29E-PUSH-DEMO"
        push_payload = {
            "vehicle_id": vid,
            "time": "2026-08-27T13:00:00",
            "raw_fuel": 135.0,
            "speed": 35.0,
        }
        p_resp = client.post("/api/push", json=push_payload)
        assert p_resp.status_code == 200

        d_resp = client.get(f"/api/data?vehicle_id={vid}&limit=10")
        assert d_resp.status_code == 200
        data = d_resp.json()
        assert isinstance(data, list)
        assert len(data) >= 1
        assert data[-1]["raw"] == 135.0

    def test_fleet_status_api(self):
        """Case 6.2: Kiem tra trang thai doi xe /api/fleet/status."""
        resp = client.get("/api/fleet/status")
        assert resp.status_code == 200
        fleet = resp.json()
        assert isinstance(fleet, list)

    def test_switch_vehicle_api(self):
        """Case 6.3: Chuyen doi xe hien thi qua /api/switch-vehicle."""
        resp = client.post("/api/switch-vehicle", json={"vehicle_id": "29E-SWITCH-TARGET"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "switched"
        assert data["vehicle_id"] == "29E-SWITCH-TARGET"

    def test_web_vehicles_list_and_current(self):
        """Case 6.4: Lay danh sach xe cho web app (/api/vehicles va /api/current-vehicle)."""
        r1 = client.get("/api/vehicles")
        assert r1.status_code == 200
        assert isinstance(r1.json(), list)

        r2 = client.get("/api/current-vehicle")
        assert r2.status_code == 200
        assert "vehicle_id" in r2.json()


# ============================================================================
# 7. TEST SUITE: History & Excel Export (/api/history, /api/export_history)
# ============================================================================
class TestHistoryAndExportAPI:
    def test_history_query_and_date_filter(self):
        """Case 7.1: Tra cuu lich su co loc theo ngay."""
        vid = "29E-HIST-01"
        # Push 2 diem o 2 ngay khac nhau
        client.post("/api/push", json={"vehicle_id": vid, "time": "2026-08-10T10:00:00", "fuel": 100.0})
        client.post("/api/push", json={"vehicle_id": vid, "time": "2026-08-15T10:00:00", "fuel": 95.0})

        # Khong loc ngay -> ca 2
        r_all = client.get(f"/api/history?vehicle_id={vid}")
        assert r_all.status_code == 200
        assert len(r_all.json()) >= 2

        # Loc chi lay ngay 2026-08-15
        r_filtered = client.get(f"/api/history?vehicle_id={vid}&start_date=2026-08-15")
        assert r_filtered.status_code == 200
        for pt in r_filtered.json():
            assert "2026-08-15" in pt.get("time", "")

    def test_export_history_excel_success(self):
        """Case 7.2: Xuat file Excel bao cao nhien lieu (.xlsx)."""
        vid = "29E-EXCEL-01"
        client.post("/api/push", json={"vehicle_id": vid, "time": "2026-08-10T10:00:00", "fuel": 100.0, "speed": 40.0})
        resp = client.get(f"/api/export_history?vehicle_id={vid}")
        assert resp.status_code == 200
        assert resp.headers["content-type"] == "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        assert len(resp.content) > 1000  # File Excel hop le co kich thuoc > 1KB

    def test_export_history_not_found(self):
        """Case 7.3: Xuat Excel cho xe hoan toan khong co du lieu -> HTTP 404."""
        resp = client.get("/api/export_history?vehicle_id=NON-EXISTENT-CAR-999")
        assert resp.status_code == 404


# ============================================================================
# 8. TEST SUITE: Security & API Key Authentication Middleware
# ============================================================================
class TestSecurityMiddleware:
    def test_security_middleware_when_enabled(self, monkeypatch):
        """Kiem tra middleware chan toan bo endpoint ghi/loc du lieu khi bat REQUIRE_API_KEY=true."""
        import src.service.api as api_mod

        monkeypatch.setattr(api_mod, "REQUIRE_API_KEY", True)
        monkeypatch.setattr(api_mod, "API_KEY", "test_secret_key_123")

        # 1. Endpoint public (health) phai van truy cap duoc khong can key
        r_health = client.get("/api/v1/health")
        assert r_health.status_code == 200

        # 2. Endpoint /api/v1/clean bi chan neu khong co key, cho qua neu dung key
        r_clean_blocked = client.post("/api/v1/clean", json={
            "vehicle_id": "29E-SEC-01",
            "timestamp": "2026-08-27 10:00:00",
            "raw_fuel": 100.0,
        })
        assert r_clean_blocked.status_code == 401
        assert r_clean_blocked.json()["error_code"] == "UNAUTHORIZED"

        r_clean_ok = client.post(
            "/api/v1/clean",
            json={"vehicle_id": "29E-SEC-01", "timestamp": "2026-08-27 10:00:00", "raw_fuel": 100.0},
            headers={"X-API-Key": "test_secret_key_123"},
        )
        assert r_clean_ok.status_code == 200

        # 3. Endpoint /api/v1/fuel/clean-point bi chan neu khong co key
        r_fuel_blocked = client.post("/api/v1/fuel/clean-point", json={
            "VehicleID": "29E-SEC-01",
            "FuelTime": "2026-08-27T10:00:00",
            "FuelLevel": 100.0,
        })
        assert r_fuel_blocked.status_code == 401

        # 4. Endpoint /api/push bi chan neu khong co key, cho qua neu dung Bearer token
        r_push_blocked = client.post("/api/push", json={"vehicle_id": "29E-SEC-01", "raw": 100.0})
        assert r_push_blocked.status_code == 401

        r_push_ok = client.post(
            "/api/push",
            json={"vehicle_id": "29E-SEC-01", "raw": 100.0},
            headers={"Authorization": "Bearer test_secret_key_123"},
        )
        assert r_push_ok.status_code == 200

        # 5. Endpoint /api/switch-vehicle bi chan neu khong co key
        r_switch_blocked = client.post("/api/switch-vehicle", json={"vehicle_id": "29E-SEC-01"})
        assert r_switch_blocked.status_code == 401

        # 6. Endpoint /api/export_history bi chan neu khong co key
        r_export_blocked = client.get("/api/export_history?vehicle_id=29E-SEC-01")
        assert r_export_blocked.status_code == 401


# ============================================================================
# 9. STANDALONE RUNNER (Ho tro chay truc tiep kiem tra Live Server / Container)
# ============================================================================
def run_standalone_test(base_url: str = None):
    """Chay toan bo test suite va in bao cao mau sac truc quan ra terminal."""
    import requests

    is_remote = base_url is not None
    target_name = base_url if is_remote else "FastAPI In-Memory TestClient"
    print("=" * 75)
    print(f"VICOMSAT API TEST RUNNER — MUC TIEU: {target_name}")
    print("=" * 75)

    def request(method, path, **kwargs):
        if is_remote:
            url = base_url.rstrip("/") + path
            return requests.request(method, url, timeout=10, **kwargs)
        else:
            return getattr(client, method.lower())(path, **kwargs)

    test_cases = [
        ("GET", "/api/v1/health", {}, 200, "1.1 Health Check Status"),
        ("POST", "/api/v1/fuel/clean-point", {"VehicleID": "29E-RUNNER", "FuelTime": "2026-08-27T10:00:00", "FuelLevel": 150.0, "Speed": 35.0}, 200, "2.1 Clean Point Standard"),
        ("POST", "/api/v1/fuel/clean-point", {"VehicleID": "29E-RUNNER", "FuelTime": "2026-08-27T10:02:00", "FuelLevel": 0.0}, 200, "2.2 Clean Point Zero Dropout"),
        ("POST", "/api/v1/fuel/clean-point", {"FuelLevel": 150.0}, 422, "2.3 Clean Point Missing VehicleID (Expect 422)"),
        ("POST", "/api/v1/fuel/clean-batch", {"vehicle_id": "29E-RUNNER", "points": [{"VehicleID": "29E-RUNNER", "FuelTime": "2026-08-27T10:04:00", "FuelLevel": 149.0}]}, 200, "3.1 Clean Batch Standard"),
        ("POST", "/api/v1/clean", {"vehicle_id": "29E-RUNNER", "timestamp": "2026-08-27 10:06:00", "raw_fuel": 148.5, "speed": 40.0}, 200, "4.1 Enterprise Point English"),
        ("POST", "/api/v1/clean", {"bien_so": "29E-RUNNER", "thoi_gian": "2026-08-27 10:08:00", "xang_tho": 148.0, "van_toc": 40.0}, 200, "4.2 Enterprise Point Vietnamese"),
        ("GET", "/api/v1/vehicles", {}, 200, "5.1 List Active Vehicles"),
        ("POST", "/api/v1/vehicles/29E-RUNNER/reset-state", {}, 200, "5.2 Reset Vehicle State"),
        ("POST", "/api/push", {"vehicle_id": "29E-RUNNER", "time": "2026-08-27T10:10:00", "raw_fuel": 147.0}, 200, "6.1 Push Live Point"),
        ("GET", "/api/data?vehicle_id=29E-RUNNER&limit=5", {}, 200, "6.2 Get Live Data"),
        ("GET", "/api/fleet/status", {}, 200, "6.3 Fleet Status"),
        ("POST", "/api/switch-vehicle", {"vehicle_id": "29E-RUNNER"}, 200, "6.4 Switch Live Vehicle"),
        ("GET", "/api/vehicles", {}, 200, "6.5 Web Vehicles List"),
        ("GET", "/api/current-vehicle", {}, 200, "6.6 Get Current Vehicle"),
        ("GET", "/api/history?vehicle_id=29E-RUNNER", {}, 200, "7.1 Get History"),
        ("GET", "/api/export_history?vehicle_id=29E-RUNNER", {}, 200, "7.2 Export Excel History"),
        ("GET", "/api/export_history?vehicle_id=CAR-NOT-EXIST-404", {}, 404, "7.3 Export Excel Not Found (Expect 404)"),
    ]

    passed = 0
    failed = 0

    for method, path, body, exp_status, title in test_cases:
        t0 = time.perf_counter()
        try:
            if method == "GET":
                res = request("GET", path)
            else:
                res = request("POST", path, json=body)
            dt = (time.perf_counter() - t0) * 1000.0

            if res.status_code == exp_status:
                passed += 1
                status_str = f"[PASS] (HTTP {res.status_code} in {dt:.1f}ms)"
                print(f"  OK   {title:<45} {status_str}")
            else:
                failed += 1
                status_str = f"[FAIL] (Got {res.status_code}, Expected {exp_status})"
                print(f"  ERR  {title:<45} {status_str}")
        except Exception as e:
            failed += 1
            print(f"  ERR  {title:<45} [EXCEPTION]: {e}")

    print("-" * 75)
    print(f"TONG KET: {passed}/{len(test_cases)} cases PASSED ({passed*100//len(test_cases)}%) | FAILED: {failed}")
    print("=" * 75)
    return failed == 0


if __name__ == "__main__":
    url_arg = sys.argv[1] if len(sys.argv) > 1 else None
    success = run_standalone_test(url_arg)
    sys.exit(0 if success else 1)
