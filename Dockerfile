# Syntax: docker/dockerfile:1
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# curl serves Docker health checks; no build toolchain is needed for wheel-based
# dependencies in requirements.txt.
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/* \
    && addgroup --system appgroup \
    && adduser --system --ingroup appgroup appuser

COPY requirements.txt ./
# Tận dụng BuildKit cache mount để lưu lại các wheel đã tải, không bị tải lại từ đầu nếu gián đoạn
# Chia thành các bước để Docker cache từng layer (nhẹ trước, nặng sau)
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --timeout=60 --retries=5 --prefer-binary \
    fastapi>=0.111.0 uvicorn>=0.30.0 pydantic>=2.7.0 sqlalchemy>=2.0.0 \
    requests>=2.31.0 python-dotenv>=1.0.0 httpx>=0.27.0 redis>=5.0.0 openpyxl>=3.1.0

RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --timeout=60 --retries=5 --prefer-binary \
    numpy>=1.26.0

RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --timeout=60 --retries=5 --prefer-binary \
    pandas>=2.0.0

RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --timeout=60 --retries=5 --prefer-binary \
    scikit-learn>=1.4.0

COPY --chown=appuser:appgroup models/ ./models/
COPY --chown=appuser:appgroup src/ ./src/
RUN mkdir -p /app/fuel_data && chown -R appuser:appgroup /app/fuel_data

USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD curl --fail --silent http://localhost:8000/api/v1/health || exit 1

CMD ["python", "-m", "uvicorn", "src.service.api:app", "--host", "0.0.0.0", "--port", "8000"]
