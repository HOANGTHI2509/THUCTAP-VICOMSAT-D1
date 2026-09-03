@echo off
chcp 65001 >nul
title VICOMSAT Fuel Denoising & Filtering Service - Enterprise Launcher

echo =====================================================================
echo    VICOMSAT REALTIME FUEL CLEANING SERVICE (ENTERPRISE EDITION)
echo =====================================================================
echo.

:: 1. Kiểm tra Python
where python >nul 2>nul
if %errorlevel% neq 0 (
    echo [LỖI] Máy tính chưa cài đặt Python hoặc chưa thêm Python vào PATH!
    echo Vui lòng cài đặt Python 3.10 trở lên từ https://www.python.org/
    pause
    exit /b 1
)

echo [1/3] Kiểm tra môi trường Python: OK!
python --version

:: 2. Kiểm tra file cấu hình .env
if not exist ".env" (
    echo [2/3] Chưa tìm thấy file .env, đang tự động tạo từ .env.example...
    copy ".env.example" ".env" >nul
    echo       -> Đã tạo file cấu hình .env mặc định!
) else (
    echo [2/3] Tìm thấy file cấu hình .env: OK!
)

:: 3. Kiểm tra và cài đặt thư viện phụ thuộc
echo [3/3] Đang kiểm tra thư viện hệ thống...
python -c "import fastapi, uvicorn, sklearn, sqlalchemy, pandas" >nul 2>nul
if %errorlevel% neq 0 (
    echo       -> Đang tự động cài đặt các thư viện cần thiết (vui lòng đợi 1 chút)...
    pip install -r requirements.txt
) else (
    echo       -> Toàn bộ thư viện đã sẵn sàng!
)

echo.
echo =====================================================================
echo [THÀNH CÔNG] KHỞI ĐỘNG HỆ THỐNG LỌC NHIÊN LIỆU AI VICOMSAT...
echo =====================================================================
echo - Tài liệu API tương tác (Swagger): http://localhost:8000/docs
echo - Endpoint kiểm tra sức khỏe:      http://localhost:8000/api/v1/health
echo - Endpoint lọc dữ liệu Doanh nghiệp: POST http://localhost:8000/api/v1/clean
echo =====================================================================
echo (Nhấn Ctrl + C để dừng dịch vụ)
echo.

python -m uvicorn src.service.api:app --host 0.0.0.0 --port 8000
pause
