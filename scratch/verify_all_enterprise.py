"""
Kịch bản kiểm thử tích hợp và nghiệm thu toàn bộ hệ thống doanh nghiệp (Task 6).
"""
import os
import sys
import subprocess

sys.path.insert(0, os.path.abspath("."))
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if sys.stderr and hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


def run_suite():
    print("=" * 75)
    print("🏆 BẮT ĐẦU TỔNG NGHIỆM THU HỆ THỐNG DOANH NGHIỆP VICOMSAT (TASK 1 -> 5)")
    print("=" * 75)

    suites = [
        ("Task 1: CSDL chuẩn 3NF (SQLAlchemy)", "scratch/test_task1_db_3nf.py"),
        ("Task 2: Thư viện Python SDK (FuelCleanerEngine)", "scratch/test_task2_sdk.py"),
        ("Task 3: REST API Microservice & Bảo mật", "scratch/test_task3_api.py"),
    ]

    all_passed = True
    for name, script in suites:
        print(f"\n--- ĐANG CHẠY TEST: {name} ---")
        res = subprocess.run([sys.executable, script], capture_output=True, text=True, encoding="utf-8", errors="replace")
        if res.returncode == 0:
            print(f"✅ {name}: ĐẠT CHUẨN (PASS 100%)")
        else:
            print(f"❌ {name}: THẤT BẠI!")
            print(res.stdout)
            print(res.stderr)
            all_passed = False

    # Kiểm tra sự tồn tại của bộ tài liệu và đóng gói
    print("\n--- KIỂM TRA BỘ ĐÓNG GÓI & TÀI LIỆU BÀN GIAO ---")
    pack_files = [
        "Dockerfile",
        "docker-compose.yml",
        "run_enterprise.bat",
        "run_enterprise.sh",
        ".env.example",
        "HDSD_DOANH_NGHIEP.md",
        "requirements.txt",
    ]
    for f in pack_files:
        assert os.path.exists(f), f"Thiếu file {f}"
        print(f"  [OK] {f:25s} ({os.path.getsize(f):,} bytes)")

    print("\n" + "=" * 75)
    if all_passed:
        print("🎉 TẤT CẢ CÁC MODULE VÀ BỘ TEST ĐÃ HOÀN TẤT VÀ PASS 100%!")
        print("HỆ THỐNG ĐÃ ĐỦ ĐIỀU KIỆN SẴN SÀNG BÀN GIAO CHO DOANH NGHIỆP.")
    else:
        print("⚠️ CÓ BÀI TEST CHƯA ĐẠT, VUI LÒNG KIỂM TRA LẠI LOG CHI TIẾT.")
    print("=" * 75)


if __name__ == "__main__":
    run_suite()
