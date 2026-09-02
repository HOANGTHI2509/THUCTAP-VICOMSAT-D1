# 1. Base image Python slim siêu nhẹ
FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# 2. Cài đặt các thư viện Python (dùng pre-built wheels, không cần build compiler gcc)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 3. Chỉ copy code xử lý, model đã train, script test và tài liệu vào container
COPY src /app/src
COPY models/rf_signal_state_causal_v3 /app/models/rf_signal_state_causal_v3
COPY models/fuel_state_classifier /app/models/fuel_state_classifier
COPY scripts /app/scripts
COPY docs /app/docs

EXPOSE 8000

# 4. Chạy API microservice
CMD ["uvicorn", "src.service.api:app", "--host", "0.0.0.0", "--port", "8000"]
