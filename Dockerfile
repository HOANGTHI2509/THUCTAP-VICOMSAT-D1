# =====================================================================
# DOCKERFILE CHO VICOMSAT FUEL DENOISING & FILTERING SERVICE
# =====================================================================
FROM python:3.11-slim

# Thiết lập biến môi trường không ghi file .pyc và in log trực tiếp
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

WORKDIR /app

# Cài đặt curl để phục vụ kiểm tra sức khỏe container (Healthcheck)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Cài đặt các thư viện phụ thuộc
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Sao chép mô hình AI đã huấn luyện và mã nguồn hệ thống
COPY models/ ./models/
COPY src/ ./src/

# Mở cổng API
EXPOSE 8000

# Kiểm tra sức khỏe định kỳ container
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/api/v1/health || exit 1

# Khởi động Uvicorn Web Server
CMD ["python", "-m", "uvicorn", "src.service.api:app", "--host", "0.0.0.0", "--port", "8000"]
