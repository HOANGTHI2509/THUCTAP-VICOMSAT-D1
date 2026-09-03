"""
Script kiểm thử toàn diện Module Cơ sở dữ liệu chuẩn 3NF (Task 1).
"""
import os
import sys
import shutil
from datetime import datetime

# Đảm bảo đường dẫn gốc có trong sys.path
sys.path.insert(0, os.path.abspath("."))
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if sys.stderr and hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from src.db.models import Base, Vehicle, AISignalState, FuelLog, FuelEvent
from src.db.database import DatabaseManager

TEST_DB_FILE = "scratch/test_fuel_3nf.db"
TEST_DB_URL = f"sqlite:///{TEST_DB_FILE}"


def run_tests():
    print("=" * 70)
    print("🚀 BẮT ĐẦU KIỂM THỬ TASK 1: MODULE CƠ SỞ DỮ LIỆU CHUẨN 3NF")
    print("=" * 70)

    # Dọn dẹp file test cũ nếu có
    if os.path.exists(TEST_DB_FILE):
        os.remove(TEST_DB_FILE)

    # 1. Khởi tạo DatabaseManager với SQLite test
    print("\n[1/6] Khởi tạo DatabaseManager...")
    db = DatabaseManager(database_url=TEST_DB_URL)
    assert db.enabled is True, "Lỗi: DatabaseManager không ở trạng thái enabled"
    print("  -> Khởi tạo thành công kết nối tới test DB!")

    # 2. Kiểm tra cấu trúc 4 bảng đã được tạo đúng chuẩn 3NF
    print("\n[2/6] Kiểm tra cấu trúc bảng trong CSDL...")
    from sqlalchemy import inspect
    inspector = inspect(db.engine)
    table_names = inspector.get_table_names()
    print(f"  -> Các bảng hiện có: {table_names}")
    for expected_table in ["vehicles", "ai_signal_states", "fuel_logs", "fuel_events"]:
        assert expected_table in table_names, f"Lỗi: Thiếu bảng {expected_table} trong CSDL"
    print("  -> Xác nhận đủ 4 bảng chuẩn 3NF!")

    # 3. Kiểm tra Seed dữ liệu từ điển trạng thái AI
    print("\n[3/6] Kiểm tra bảng từ điển ai_signal_states...")
    session = db.session_factory()
    states = session.query(AISignalState).all()
    print(f"  -> Số lượng trạng thái AI đã nạp: {len(states)}")
    state_codes = {s.state_code: s.state_name_vi for s in states}
    for code in ["UPWARD_SHIFT", "DOWNWARD_SHIFT", "GRADUAL_CHANGE", "STABLE_JITTER", "OSCILLATION_NOISE"]:
        assert code in state_codes, f"Lỗi: Thiếu mã trạng thái {code}"
        print(f"     * [{code}] -> {state_codes[code]}")
    session.close()

    # 4. Kiểm tra lưu điểm đo đạc sạch (fuel_logs & liên kết vehicles)
    print("\n[4/6] Kiểm tra hàm save_measurement (tự liên kết khóa ngoại vehicles)...")
    log_id1 = db.save_measurement(
        vehicle_id="29H-75028",
        timestamp="2026-08-15 09:16:00",
        raw_fuel=93.3,
        clean_fuel=93.3,
        speed=61.0,
        lat=21.0285,
        lng=105.8542,
        state_code="UPWARD_SHIFT",
        capacity_est=95.0,
    )
    assert log_id1 is not None, "Lỗi: Không lưu được điểm đo đạc"
    print(f"  -> Lưu thành công bản ghi log ID: {log_id1}")

    # Xác nhận bảng vehicles đã được tự động thêm xe 29H-75028 với dung tích 95L
    session = db.session_factory()
    v = session.get(Vehicle, "29H-75028")
    assert v is not None, "Lỗi: Xe không được tự động thêm vào bảng vehicles"
    assert v.capacity_liters == 95.0, f"Lỗi: Dung tích xe không đúng (kỳ vọng 95.0, có {v.capacity_liters})"
    print(f"  -> Bảng vehicles đã tự lưu: {v.vehicle_id} | Dung tích chuẩn: {v.capacity_liters}L")

    # Kiểm tra log đọc ra kèm tên tiếng Việt từ quan hệ 3NF
    logs = db.get_logs(vehicle_id="29H-75028")
    assert len(logs) == 1, "Lỗi: Số lượng log đọc ra không khớp"
    print(f"  -> Dữ liệu log truy vấn: Xe={logs[0]['vehicle_id']} | Xăng={logs[0]['clean_fuel']}L | Trạng thái={logs[0]['state_name_vi']}")
    session.close()

    # 5. Kiểm tra lưu sự kiện đổ xăng (fuel_events)
    print("\n[5/6] Kiểm tra hàm save_event (lưu sự kiện Đổ xăng / Hụt dầu)...")
    event_id = db.save_event(
        vehicle_id="29H-75028",
        event_type="REFUEL",
        start_time="2026-08-15 09:10:00",
        end_time="2026-08-15 09:16:00",
        start_fuel=48.3,
        end_fuel=93.3,
        change_liters=45.0,
        lat=21.0285,
        lng=105.8542,
        address="Cây xăng QL1A, Hà Nội",
        capacity_est=95.0,
    )
    assert event_id is not None, "Lỗi: Không lưu được sự kiện fuel_events"
    events = db.get_events(vehicle_id="29H-75028")
    assert len(events) == 1, "Lỗi: Số lượng sự kiện không khớp"
    ev = events[0]
    print(f"  -> Sự kiện lưu thành công ID={ev['id']}: Loại={ev['event_type']} | Tăng={ev['change_liters']}L | Địa điểm={ev['address']}")

    # 6. Kiểm tra chế độ Zero-DB (không dùng CSDL)
    print("\n[6/6] Kiểm tra chế độ Zero-DB (DATABASE_URL=NONE)...")
    zero_db = DatabaseManager(database_url="NONE")
    assert zero_db.enabled is False, "Lỗi: Zero-DB phải có enabled = False"
    # Gọi các hàm không được văng lỗi
    res_log = zero_db.save_measurement("XE_TEST", datetime.now(), 50.0, 50.0)
    assert res_log is None
    res_ev = zero_db.save_event("XE_TEST", "REFUEL", datetime.now(), datetime.now(), 50.0, 80.0, 30.0)
    assert res_ev is None
    assert zero_db.get_logs("XE_TEST") == []
    assert zero_db.get_events("XE_TEST") == []
    print("  -> Chế độ Zero-DB hoạt động an toàn, không gây crash!")

    # Đóng kết nối để giải phóng file SQLite
    db.close()

    # Dọn dẹp
    if os.path.exists(TEST_DB_FILE):
        os.remove(TEST_DB_FILE)

    print("\n" + "=" * 70)
    print("🎉 TẤT CẢ 6/6 BÀI KIỂM THỬ TASK 1 ĐÃ PASS THÀNH CÔNG 100%!")
    print("=" * 70)


if __name__ == "__main__":
    run_tests()
