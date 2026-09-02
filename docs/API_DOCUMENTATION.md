# Tài liệu Tích hợp API: Real-time Fuel Denoising Microservice

## 1. Tổng quan Kiến trúc

Microservice xử lý nhiễu nhiên liệu được thiết kế chuyên biệt cho hệ thống giám sát hành trình của VICOMSAT. 
Dịch vụ nhận luồng dữ liệu thô từ cảm biến nhiên liệu, áp dụng mô hình **Causal AI (Random Forest)** để nhận diện trạng thái, và sử dụng **Adaptive Kalman Filter** để làm sạch nhiễu thời gian thực.

Đặc điểm cốt lõi:
- **In-Memory State Management:** Tự động duy trì trạng thái của hàng nghìn xe trong RAM mà không cần dùng đến database hay Redis.
- **Auto-Reset:** Tự động xoá trạng thái của xe nếu không có tín hiệu mới quá 120 phút.
- **Latency tối ưu:** Thời gian xử lý trung bình `< 80ms / điểm`.

---

## 2. Thông tin Kết nối
- **Giao thức:** HTTP/RESTful
- **Cổng mặc định (Docker):** `8000`
- **Tài liệu Swagger UI:** `http://<domain_hoac_ip>:8000/docs`

---

## 3. Các API Endpoints Chi tiết

### 3.1. Nhận dữ liệu Streaming từng điểm (Single Point)
Sử dụng endpoint này khi thiết bị GPS bắn dữ liệu từng nhịp (ví dụ: mỗi 5-10 giây/lần).

- **Endpoint:** `POST /api/v1/fuel/clean-point`
- **Content-Type:** `application/json`

**Request Payload:**
```json
{
  "vehicle_id": "29E-45520",
  "fuel_time": "2026-08-27T10:00:00",
  "fuel_level": 105.0,
  "speed": 45.0,
  "lat": 21.0285,
  "lng": 105.8542,
  "distance_meters": 350.0,
  "capacity_est": 200.0,
  "noise_sigma_liters": 0.8
}
```
*(Ghi chú: `capacity_est` và `noise_sigma_liters` là không bắt buộc. Nếu không có, API sẽ tự động cấu hình mặc định (200L, 0.8L) hoặc cấu hình động theo lịch sử xe).*

**Response (200 OK):**
```json
{
  "vehicle_id": "29E-45520",
  "fuel_time": "2026-08-27T10:00:00",
  "raw_fuel_liters": 105.0,
  "clean_fuel_liters": 104.8,
  "ai_signal_state": "GRADUAL_CHANGE",
  "confidence": 0.98,
  "quality_flag": "VALID",
  "latency_ms": 68.5
}
```

**Các trạng thái `ai_signal_state` trả về:**
- `STABLE_JITTER`: Nhiễu tĩnh, xe đang đỗ.
- `GRADUAL_CHANGE`: Xe đang chạy, tiêu hao từ từ.
- `OSCILLATION_NOISE`: Nhiễu sóng sánh hoặc hố sụt mất tín hiệu.
- `UPWARD_SHIFT`: Nạp nhiên liệu thực sự (bơm xăng).
- `DOWNWARD_SHIFT`: Sự cố sụt giảm nhiên liệu bất thường khi xe đỗ.

---

### 3.2. Nhận dữ liệu theo gói (Batch Processing)
Sử dụng endpoint này để đồng bộ lại dữ liệu khi thiết bị vào vùng mất sóng và bắn lại 1 mảng các điểm dữ liệu.

- **Endpoint:** `POST /api/v1/fuel/clean-batch`
- **Content-Type:** `application/json`

**Request Payload:**
```json
{
  "vehicle_id": "29E-45520",
  "points": [
    {
      "vehicle_id": "29E-45520",
      "fuel_time": "2026-08-27T10:00:00",
      "fuel_level": 105.0,
      "speed": 0.0
    },
    {
      "vehicle_id": "29E-45520",
      "fuel_time": "2026-08-27T10:05:00",
      "fuel_level": 103.5,
      "speed": 50.0
    }
  ]
}
```

**Response (200 OK):** Trả về mảng `results` chứa dữ liệu sạch cho từng điểm.

---

### 3.3. Các Endpoint Quản trị Hệ thống

| Method | Endpoint | Mục đích |
| :--- | :--- | :--- |
| `GET` | `/api/v1/health` | Kiểm tra trạng thái hoạt động của Service (Alive, Model loaded). |
| `GET` | `/api/v1/vehicles` | Xem danh sách các xe đang được lưu trữ State trong RAM. |
| `POST`| `/api/v1/vehicles/{vehicle_id}/reset-state` | Ép hệ thống xoá cache lịch sử của 1 xe (dùng khi thay cảm biến, đổ xăng lớn, reset khẩn cấp). |

---

## 4. Hướng dẫn Triển khai bằng Docker

Toàn bộ dịch vụ đã được đóng gói sẵn. Doanh nghiệp chỉ cần cài đặt `Docker` & `Docker Compose` trên Server Backend.

**Bước 1:** Đặt toàn bộ source code vào một thư mục trên Server (Ví dụ: `/opt/vicomsat/fuel-api/`).

**Bước 2:** Chạy lệnh build và khởi động ở chế độ ngầm (detached):
```bash
cd /opt/vicomsat/fuel-api/
docker compose up -d
```

**Bước 3:** Kiểm tra log hoạt động:
```bash
docker compose logs -f
```

**Bước 4:** Gửi thử một Request:
```bash
curl -X GET http://localhost:8000/api/v1/health
```

**Kịch bản Tích hợp vào VICOMSAT:**
Tại Gateway nhận dữ liệu thiết bị GPS hiện tại (có thể viết bằng .NET, Java, Go...), thêm một logic:
- Mỗi khi nhận bản tin Telemetry chứa Fuel Level, gọi sang REST API `POST /api/v1/fuel/clean-point`.
- Nhận kết quả `clean_fuel_liters`, sau đó mới lưu kết quả đã làm sạch xuống Database để cho Dashboard và Mobile App hiển thị cho Khách hàng.
