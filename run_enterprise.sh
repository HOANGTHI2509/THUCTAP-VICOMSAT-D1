#!/usr/bin/env bash
# =====================================================================
# VICOMSAT REALTIME FUEL CLEANING SERVICE (LINUX LAUNCHER)
# =====================================================================

set -e

echo "====================================================================="
echo "   VICOMSAT REALTIME FUEL CLEANING SERVICE (ENTERPRISE EDITION)"
echo "====================================================================="

# 1. Kiểm tra Python
if ! command -v python3 &> /dev/null; then
    echo "[LỖI] Chưa cài đặt Python3!"
    exit 1
fi

echo "[1/3] Kiểm tra Python: $(python3 --version)"

# 2. Kiểm tra file .env
if [ ! -f ".env" ]; then
    echo "[2/3] Đang tạo file .env từ .env.example..."
    cp .env.example .env
fi

# 3. Khởi động dịch vụ
echo "[3/3] Khởi động dịch vụ Uvicorn..."
echo "- Tài liệu Swagger: http://localhost:8000/docs"
echo "- Health Check:     http://localhost:8000/api/v1/health"
echo "====================================================================="

python3 -m uvicorn src.service.api:app --host 0.0.0.0 --port 8000
