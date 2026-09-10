# Hướng dẫn tích hợp — VICOMSAT Fuel Denoising (Đề tài 1)

## Mục tiêu bàn giao

Module nhận từng bản tin cảm biến nhiên liệu đã đổi sang lít và trả về `CleanFuel` theo thời gian thực. Mỗi xe có state causal độc lập; dữ liệu xe A không ảnh hưởng xe B.

Đầu ra `SignalState`, `QualityFlag` và `MotionState` phục vụ kiểm tra chất lượng lọc. Chúng không phải kết luận nạp, rút trộm, đỗ xe hay tắt máy.

## Tích hợp REST API

Endpoint chính:

```text
POST /api/v1/fuel/clean-point
```

Ví dụ request:

```json
{
  "VehicleID": "21H-02058",
  "FuelTime": "2026-08-13T10:54:00",
  "FuelLevel": 175.2,
  "Speed": 32.0,
  "Lat": 21.0285,
  "Lng": 105.8542,
  "CapacityEst": 200.0
}
```

Doanh nghiệp nên lưu cả `RawFuel` và `CleanFuel`. Giao diện người dùng sử dụng `CleanFuel`; `RawFuel` được giữ để truy vết và đánh giá thuật toán.

Chi tiết schema và toàn bộ từ điển trạng thái nằm tại [docs/API_DOCUMENTATION.md](docs/API_DOCUMENTATION.md).

## Tích hợp Python SDK

```python
from src.sdk.fuel_cleaner import FuelCleanerEngine

cleaner = FuelCleanerEngine()
result = cleaner.clean_point(
    vehicle_id="21H-02058",
    timestamp="2026-08-13 10:54:00",
    raw_fuel=175.2,
    speed=32.0,
    lat=21.0285,
    lng=105.8542,
    capacity_est=200.0,
)
print(result["clean_fuel"])
```

`ai_state` vẫn tồn tại trong SDK như alias cũ; tích hợp mới sử dụng `signal_state`.

## Quản lý `CapacityEst`

- Có dữ liệu calib: truyền dung tích chính thức.
- Chưa có: bỏ trường này hoặc truyền `null`; bộ lọc suy ra tạm từ điểm nhiên liệu hợp lệ đầu tiên.
- Khi thấy raw cao hơn dung tích tạm, lõi tự nâng giới hạn.
- Giá trị suy ra chỉ dùng để scale ngưỡng lọc, không được công bố như dung tích thật của xe.

## State theo xe

Local/demo mặc định dùng RAM:

```env
STATE_BACKEND=memory
STATE_TTL_SECONDS=259200
```

Production có thể bật Redis:

```env
STATE_BACKEND=redis
REDIS_URL=redis://redis-host:6379/0
STATE_TTL_SECONDS=259200
```

State JSON gồm Kalman state, mức sạch cuối, lịch sử raw/speed/GPS và các bộ đếm xác nhận causal. `schema_version` được lưu kèm để hỗ trợ nâng cấp.

Lưu ý vận hành: cùng một `VehicleID` phải được xử lý theo thứ tự thời gian. Nếu triển khai nhiều consumer/instance, tầng nhận dữ liệu phải partition theo `VehicleID` để tránh hai instance đồng thời xử lý cùng một xe.

## Kiểm thử trước khi bàn giao

```powershell
.\.venv\Scripts\python.exe -m pytest tests -q -p no:cacheprovider
.\.venv\Scripts\python.exe scripts\evaluate_smooth_tracking.py
```

Kết quả KPI:

- `artifacts/evaluation/metrics.json`
- `artifacts/evaluation/segment_metrics.csv`
- `artifacts/evaluation/report.md`

Không đổi expected curve của golden test chỉ để làm test pass. Mọi case `pending_domain_review` cần được xem trên biểu đồ và xác nhận trước khi chuyển thành `approved`.

## Chạy bằng Docker

Docker Compose chỉ khởi động API ở `http://localhost:8000`.

```powershell
docker compose up --build -d
docker compose ps
docker compose logs -f fuel-api
```

Dashboard chạy local, không được đóng vào Docker image. Nó chỉ hiển thị
`RawFuel`, `CleanFuel` màu tím, `SignalState`, `QualityFlag` và `MotionState`.

```powershell
.\.venv\Scripts\pip.exe install -r requirements-dashboard.txt
.\.venv\Scripts\streamlit.exe run src\dashboard\app_dashboard_tienxuly.py
```

Muốn dùng Redis do Compose quản lý:

```powershell
$env:STATE_BACKEND = "redis"
$env:REDIS_URL = "redis://redis:6379/0"
docker compose --profile redis up --build -d
```

Tắt toàn bộ service nhưng giữ volume Redis:

```powershell
docker compose down
```
