# VICOMSAT Topic 1 — Realtime Fuel Denoising API

## 1. Phạm vi

Dịch vụ nhận dữ liệu cảm biến nhiên liệu đã quy đổi sang lít, vận tốc và GPS tùy chọn; đầu ra là mức nhiên liệu đã làm sạch theo thời gian thực. Bộ lọc chỉ dùng điểm hiện tại và lịch sử của đúng xe đó.

Dịch vụ **không kết luận** nạp nhiên liệu, rút trộm, bật/tắt máy hay trạng thái đỗ. Các nghiệp vụ này thuộc tầng phân tích phía sau (đề tài 2).

## 2. Endpoint chính

### `POST /api/v1/fuel/clean-point`

Request chuẩn:

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

Tên trường dạng snake_case (`vehicle_id`, `fuel_time`, `fuel_level`, ...) vẫn được chấp nhận. `Lat`, `Lng` và `CapacityEst` không bắt buộc.

Response chuẩn:

```json
{
  "VehicleID": "21H-02058",
  "FuelTime": "2026-08-13T10:54:00",
  "RawFuel": 175.2,
  "CleanFuel": 174.8,
  "SignalState": "STABLE_JITTER",
  "Confidence": 1.0,
  "QualityFlag": "SMOOTH_KALMAN",
  "LatencyMs": 2.4,
  "MotionState": "MOVING",
  "MotionConfidence": 0.95,
  "GpsDisplacementMeters": 88.9
}
```

`AI_State`/`ai_signal_state` là alias tương thích trong code Python cũ. Client tích hợp mới phải dùng `SignalState`.

### `POST /api/v1/fuel/clean-batch`

Nhận các điểm của một xe và xử lý theo thứ tự `FuelTime`. Mỗi phần tử trong `results` có cùng schema với endpoint một điểm.

### `POST /api/v1/clean`

Endpoint tương thích cho request dùng tên trường `vehicle_id`, `timestamp`, `raw_fuel`, `speed`, `lat`, `lng`, `capacity_est`. Response vẫn dùng schema chuẩn ở trên.

### Quản trị

| Method | Endpoint | Ý nghĩa |
|---|---|---|
| GET | `/api/v1/health` | Model, state store và trạng thái service |
| POST | `/api/v1/vehicles/{vehicle_id}/reset-state` | Xóa state và lịch sử causal của một xe |

## 3. Data dictionary

### Input

| Trường | Bắt buộc | Ý nghĩa |
|---|---:|---|
| `VehicleID` | Có | Định danh/biển số xe |
| `FuelTime` | Có | Thời điểm đo; các điểm phải đến theo thứ tự thời gian |
| `FuelLevel` | Có | Nhiên liệu thô đã quy đổi sang lít |
| `Speed` | Không | Vận tốc GPS, mặc định 0 km/h |
| `Lat`, `Lng` | Không | Tọa độ GPS; `(0,0)` được coi là thiếu |
| `CapacityEst` | Không | Dung tích ước tính để scale ngưỡng; nếu thiếu sẽ suy ra tạm từ mức hợp lệ đầu tiên |

### `SignalState`

| Giá trị | Ý nghĩa tín hiệu |
|---|---|
| `INIT` | Điểm khởi tạo |
| `STABLE_JITTER` | Dao động nhỏ quanh một mặt bằng |
| `OSCILLATION_NOISE` | Nhiễu dao động/răng cưa |
| `SLOSHING` | Sóng sánh do chuyển động |
| `GRADUAL_CHANGE` | Thay đổi có xu hướng theo thời gian |
| `UPWARD_SHIFT` | Dịch chuyển mặt bằng theo chiều tăng, chưa gắn nghĩa nghiệp vụ |
| `DOWNWARD_SHIFT` | Dịch chuyển mặt bằng theo chiều giảm, chưa gắn nghĩa nghiệp vụ |
| `SPIKE` | Xung nhiễu tức thời |
| `UNCERTAIN` | Chưa đủ bằng chứng phân loại |

### `QualityFlag`

| Giá trị | Hành động của bộ lọc |
|---|---|
| `VALID` | Điểm hợp lệ/khởi tạo |
| `SMOOTH_KALMAN` | Đã cập nhật qua Adaptive Kalman |
| `ZERO_DROPOUT_HELD` | Raw bằng 0/không hợp lệ, giữ mức sạch trước |
| `SPIKE_HELD` | Xung bị giữ lại |
| `PENDING_UPWARD_SHIFT_HELD` | Chờ thêm bằng chứng cho mặt bằng tăng |
| `UPWARD_SHIFT_TRACKED` | Đã bám mặt bằng tăng bền vững |
| `UPWARD_REVERSAL_REJECTED` | Ứng viên tăng quay về nền nên bị loại |
| `UPWARD_REVERSAL_RESET` | Mặt bằng tăng vừa bám bị đảo chiều |
| `PENDING_DOWNWARD_SHIFT_HELD` | Chờ thêm bằng chứng cho mặt bằng giảm |
| `DOWNWARD_SHIFT_TRACKED` | Đã bám mặt bằng giảm bền vững |
| `STABLE_LEVEL_TRACKING` | Bám mặt bằng ổn định mới |

### `MotionState`

| Giá trị | Điều kiện |
|---|---|
| `MOVING` | Vận tốc và dịch chuyển GPS cùng xác nhận xe đang di chuyển |
| `LOW_MOTION` | Vận tốc thấp và các điểm GPS nằm trong cụm nhỏ |
| `UNCERTAIN` | Thiếu GPS, chưa đủ điểm hoặc GPS/vận tốc mâu thuẫn |

`LOW_MOTION` không đồng nghĩa với đỗ hoặc tắt máy.

## 4. State store

Mặc định service dùng RAM:

```env
STATE_BACKEND=memory
STATE_TTL_SECONDS=259200
```

Production có thể dùng Redis:

```env
STATE_BACKEND=redis
REDIS_URL=redis://localhost:6379/0
STATE_TTL_SECONDS=259200
```

State được lưu dưới dạng JSON có `schema_version`. Nếu Redis tạm lỗi, tiến trình hiện tại tiếp tục bằng memory fallback và health check trả trạng thái `degraded`.

Ngoài TTL lưu trữ, lõi lọc tự đặt lại Kalman khi khoảng cách giữa hai điểm vượt `reset_gap_minutes` (mặc định 120 phút).

## 5. Kiểm tra chất lượng

```powershell
.\.venv\Scripts\python.exe -m pytest tests -q -p no:cacheprovider
.\.venv\Scripts\python.exe scripts\evaluate_smooth_tracking.py
```

Báo cáo KPI được ghi vào `artifacts/evaluation/`. Các golden segment có trạng thái `pending_domain_review` chỉ là baseline kỹ thuật; cần người phụ trách dữ liệu duyệt trước khi coi là chuẩn nghiệp vụ.

## 6. Docker và dashboard

```powershell
docker compose up --build -d
docker compose ps
```

- API: `http://localhost:8000`, health: `/api/v1/health`.
- Dashboard chạy local, không được đóng vào Docker image; chỉ vẽ `RawFuel` và `CleanFuel` màu tím.
- Dashboard hiển thị `SignalState`, `QualityFlag`, `MotionState` trong tooltip và bảng chi tiết.

Để chạy dashboard local: `pip install -r requirements-dashboard.txt`, sau đó
`streamlit run src/dashboard/app_dashboard_tienxuly.py`.

Nếu chạy Redis trong Compose, đặt `STATE_BACKEND=redis`, `REDIS_URL=redis://redis:6379/0` và dùng `docker compose --profile redis up --build -d`.
