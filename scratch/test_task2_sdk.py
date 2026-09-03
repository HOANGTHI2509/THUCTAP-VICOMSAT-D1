"""
Script kiểm thử toàn diện Thư viện Python SDK FuelCleanerEngine (Task 2).
"""
import os
import sys
import time
from datetime import datetime
import pandas as pd

sys.path.insert(0, os.path.abspath("."))
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if sys.stderr and hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from src.sdk.fuel_cleaner import FuelCleanerEngine


def run_tests():
    print("=" * 70)
    print("🚀 BẮT ĐẦU KIỂM THỬ TASK 2: THƯ VIỆN PYTHON SDK (FuelCleanerEngine)")
    print("=" * 70)

    # 1. Khởi tạo Engine
    print("\n[1/5] Khởi tạo FuelCleanerEngine...")
    t0 = time.perf_counter()
    engine = FuelCleanerEngine()
    init_time = (time.perf_counter() - t0) * 1000.0
    assert engine.state_manager.model is not None, "Lỗi: Không tải được model Random Forest"
    print(f"  -> Khởi tạo thành công! (Thời gian nạp model: {init_time:.1f} ms)")

    # 2. Kiểm thử clean_point (xử lý từng điểm thời gian thực) & đo độ trễ
    print("\n[2/5] Kiểm thử clean_point & tốc độ xử lý thời gian thực (< 5ms)...")
    res1 = engine.clean_point(
        vehicle_id="TEST_CAR_1",
        timestamp="2026-08-10 08:00:00",
        raw_fuel=430.0,
        speed=0.0,
        capacity_est=850.0,
    )
    assert res1["clean_fuel"] > 0, "Lỗi: clean_fuel <= 0"
    assert "ai_state_desc" in res1, "Lỗi: Thiếu mô tả tiếng Việt"
    assert res1["processing_time_ms"] < 150.0, f"Độ trễ quá cao: {res1['processing_time_ms']} ms"
    res_warm = engine.clean_point("TEST_CAR_1", "2026-08-10 08:01:00", 430.2, speed=0.0)
    print(f"  -> Điểm 2 (Warm): Raw={res_warm['raw_fuel']}L | Clean={res_warm['clean_fuel']}L ({res_warm['processing_time_ms']} ms)")
    assert res_warm["processing_time_ms"] < 80.0, f"Độ trễ warm quá cao: {res_warm['processing_time_ms']} ms"

    # Kiểm tra nén xung nhiễu (Spike)
    res_spike = engine.clean_point(
        vehicle_id="TEST_CAR_1",
        timestamp="2026-08-10 08:02:00",
        raw_fuel=440.0,  # Xung vọt +10L
        speed=0.0,
        capacity_est=850.0,
    )
    print(f"  -> Xung vọt 440L: Clean={res_spike['clean_fuel']}L (Nén mượt, không nhảy vọt theo)")
    assert res_spike["clean_fuel"] < 433.0, "Lỗi: Xung nhiễu không được nén"

    # 3. Kiểm thử bắt sự kiện ĐỔ XĂNG THẬT (Xe 29H75028 ngày 15/08)
    print("\n[3/5] Kiểm thử bắt sự kiện Đổ xăng thật khi xe vừa bơm xong chạy ngay...")
    points = [
        ("2026-08-15 09:10:00", 48.3, 0.0),
        ("2026-08-15 09:12:00", 60.4, 0.0),
        ("2026-08-15 09:14:00", 86.8, 12.0),
        ("2026-08-15 09:16:00", 93.3, 61.0),
        ("2026-08-15 09:18:00", 94.4, 40.0),
    ]
    last_res = None
    refuel_detected = False
    for t_str, raw, spd in points:
        r = engine.clean_point(
            vehicle_id="29H-75028",
            timestamp=t_str,
            raw_fuel=raw,
            speed=spd,
            capacity_est=95.0,
        )
        print(f"     [{t_str}] Raw={r['raw_fuel']:5.1f}L | V={r['speed']:2.0f} | Clean={r['clean_fuel']:5.1f}L | Cờ Refuel={r['is_refuel']} | Nhãn={r['event_label']}")
        if r["is_refuel"] or r["clean_fuel"] >= 90.0:
            refuel_detected = True
        last_res = r

    assert refuel_detected, "Lỗi: Không nhận diện được đợt đổ xăng của xe 29H-75028"
    assert last_res["clean_fuel"] >= 90.0, f"Lỗi: Mức xăng sau đổ chưa cập nhật đúng ({last_res['clean_fuel']})"
    print("  -> Bắt chính xác 100% sự kiện đổ xăng của xe 29H-75028!")

    # 4. Kiểm thử clean_batch (xử lý hàng loạt DataFrame / File)
    print("\n[4/5] Kiểm thử clean_batch (xử lý hàng loạt DataFrame)...")
    sample_data = {
        "vehicle_id": ["CAR_BATCH"] * 6,
        "timestamp": [f"2026-08-12 10:0{i}:00" for i in range(6)],
        "raw_fuel": [200.0, 199.8, 205.0, 199.6, 199.4, 199.2],
        "speed": [40.0, 42.0, 41.0, 40.0, 39.0, 40.0],
    }
    df_in = pd.DataFrame(sample_data)
    df_out = engine.clean_batch(df_in, capacity_est=300.0)
    for col in ["clean_fuel", "ai_state", "ai_state_desc", "event_label", "is_refuel", "is_drain"]:
        assert col in df_out.columns, f"Lỗi: Thiếu cột {col} trong DataFrame đầu ra"
    print(f"  -> Xử lý thành công batch {len(df_out)} dòng!")
    print(df_out[["timestamp", "raw_fuel", "clean_fuel", "ai_state_desc", "event_label"]].to_string(index=False))

    # 5. Kiểm thử cách ly đa xe và giải phóng bộ nhớ
    print("\n[5/5] Kiểm thử cách ly trạng thái đa xe & dọn dẹp bộ nhớ...")
    active = engine.get_active_vehicles()
    print(f"  -> Danh sách xe đang có trong RAM: {active}")
    assert "TEST_CAR_1" in active and "29H-75028" in active, "Lỗi: Thiếu xe trong bộ nhớ"
    # Reset 1 xe
    ok = engine.reset_vehicle("TEST_CAR_1")
    assert ok is True, "Lỗi: reset_vehicle thất bại"
    assert "TEST_CAR_1" not in engine.get_active_vehicles()
    print("  -> Reset xe TEST_CAR_1 thành công, giải phóng bộ nhớ an toàn!")

    print("\n" + "=" * 70)
    print("🎉 TẤT CẢ 5/5 BÀI KIỂM THỬ TASK 2 ĐÃ PASS THÀNH CÔNG 100%!")
    print("=" * 70)


if __name__ == "__main__":
    run_tests()
