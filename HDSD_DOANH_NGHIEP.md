# HƯỚNG DẪN TÍCH HỢP & BÀN GIAO HỆ THỐNG LỌC NHIÊN LIỆU AI VICOMSAT
*(VICOMSAT Realtime AI-Kalman Fuel Denoising & Filtering Service)*

Tài liệu này cung cấp hướng dẫn đầy đủ để đội ngũ kỹ thuật của doanh nghiệp có thể tích hợp và triển khai hệ thống lọc dữ liệu nhiên liệu theo **3 kịch bản tùy chọn** (REST API, Python SDK hoặc Trọn gói Web App/Docker).

---

## MỤC LỤC
1. [Tổng Quan Kiến Trúc](#1-tổng-quan-kiến-trúc)
2. [Kịch Bản 1: Tích Hợp Qua REST API Microservice (Java, C#, Node.js, PHP, Go)](#2-kịch-bản-1-tích-hợp-qua-rest-api-microservice)
3. [Kịch Bản 2: Nhúng Trực Tiếp Bằng Python SDK](#3-kịch-bản-2-nhúng-trực-tiếp-bằng-python-sdk)
4. [Kịch Bản 3: Triển Khai Trọn Gói (Docker & Script 1-Click)](#4-kịch-bản-3-triển-khai-trọn-gói-docker--script-1-click)
5. [Cấu Hình Bảo Mật & Cơ Sở Dữ Liệu Chuẩn 3NF](#5-cấu-hình-bảo-mật--cơ-sở-dữ-liệu-chuẩn-3nf)
6. [Từ Điển Dữ Liệu & Mã Trạng Thái (Data Dictionary)](#6-từ-điển-dữ-liệu--mã-trạng-thái)

---

## 1. TỔNG QUAN KIẾN TRÚC
Hệ thống kết hợp mô hình học máy **Random Forest Causal** (28 đặc trưng vật lý) và bộ lọc thích nghi **Adaptive Kalman Filter** hoạt động theo thời gian thực (chỉ nhìn điểm hiện tại và quá khứ, không phụ thuộc vào tương lai).

- **Độ trễ xử lý**: < 40 ms / điểm đo.
- **Tương thích**: Hỗ trợ mọi loại xe tải, xe bồn (dung tích 50L – 1.000L).
- **Khử nhiễu**: Nén triệt để sóng sánh do quán tính xe chạy, xung giật điện áp cảm biến.
- **Bắt biến cố**: Nhận diện tức thời sự kiện Đổ xăng (kể cả khi vừa bơm xong xe chạy ngay) và cảnh báo rút trộm dầu.

---

## 2. KỊCH BẢN 1: TÍCH HỢP QUA REST API MICROSERVICE
*(Dành cho doanh nghiệp đã có sẵn hệ thống Backend viết bằng Java, C#, Node.js, PHP, Go,...)*

Máy chủ API lắng nghe tại cổng mặc định `8000`.

### 2.1. Lọc 1 điểm dữ liệu thời gian thực (`POST /api/v1/clean`)
- **Endpoint**: `POST http://<server-ip>:8000/api/v1/clean`
- **Headers**:
  ```http
  Content-Type: application/json
  X-API-Key: vicomsat_secret_key_2026
  ```

#### Dữ liệu gửi lên (Request Body):
Hệ thống hỗ trợ cả tên trường chuẩn tiếng Anh hoặc bí danh tiếng Việt:
```json
{
  "vehicle_id": "29H-75028",
  "timestamp": "2026-08-15 09:16:00",
  "raw_fuel": 93.3,
  "speed": 61.0,
  "lat": 21.0285,
  "lng": 105.8542,
  "capacity_est": 95.0
}
```
*(Hoặc dùng bí danh tiếng Việt: `bien_so`, `thoi_gian`, `xang_tho`, `van_toc`, `dung_tich`)*.

#### Kết quả trả về (Response Body - HTTP 200):
```json
{
  "vehicle_id": "29H-75028",
  "timestamp": "2026-08-15 09:16:00",
  "raw_fuel": 93.3,
  "clean_fuel": 93.3,
  "speed": 61.0,
  "ai_state": "UPWARD_SHIFT",
  "ai_state_desc": "Bơm/Đổ nhiên liệu",
  "event_label": "Đổ xăng",
  "confidence": 0.9521,
  "is_refuel": true,
  "is_drain": false,
  "is_spike": false,
  "processing_time_ms": 38.45,
  "saved_to_db": true
}
```

#### Mẫu cURL kiểm tra nhanh:
```bash
curl -X POST "http://localhost:8000/api/v1/clean" \
     -H "Content-Type: application/json" \
     -H "X-API-Key: vicomsat_secret_key_2026" \
     -d '{
       "vehicle_id": "29H-75028",
       "timestamp": "2026-08-15 09:16:00",
       "raw_fuel": 93.3,
       "speed": 61.0
     }'
```

---

### 2.2. Lọc hàng loạt theo cụm (`POST /api/v1/clean-batch`)
Dành cho thiết bị truyền tin theo cụm (batch) 10–500 điểm đo đạc:
- **Endpoint**: `POST http://<server-ip>:8000/api/v1/clean-batch`
- **Request Body**:
```json
{
  "points": [
    {"vehicle_id": "29H-75028", "timestamp": "2026-08-15 09:10:00", "raw_fuel": 48.3, "speed": 0.0},
    {"vehicle_id": "29H-75028", "timestamp": "2026-08-15 09:12:00", "raw_fuel": 60.4, "speed": 0.0},
    {"vehicle_id": "29H-75028", "timestamp": "2026-08-15 09:16:00", "raw_fuel": 93.3, "speed": 61.0}
  ]
}
```

---

### 2.3. Tra cứu sự kiện Đổ xăng / Rút dầu (`GET /api/v1/events`)
- **Endpoint**: `GET http://<server-ip>:8000/api/v1/events?vehicle_id=29H-75028&limit=50`
- **Kết quả trả về**:
```json
[
  {
    "id": 1,
    "vehicle_id": "29H-75028",
    "event_type": "REFUEL",
    "start_time": "2026-08-15T09:16:00",
    "end_time": "2026-08-15T09:16:00",
    "start_fuel": 48.3,
    "end_fuel": 93.3,
    "change_liters": 45.0,
    "lat": 21.0285,
    "lng": 105.8542,
    "address": "Xã Hợp Thịnh, Tỉnh Bắc Ninh"
  }
]
```

---

## 3. KỊCH BẢN 2: NHÚNG TRỰC TIẾP BẰNG PYTHON SDK
*(Dành cho đội ngũ kỹ sư dữ liệu hoặc backend Python muốn nhúng trực tiếp vào code có sẵn)*

Không cần chạy thêm máy chủ web riêng, chỉ cần `import` module SDK:

```python
from src.sdk import FuelCleanerEngine

# 1. Khởi tạo Engine (Tự động nạp model AI Causal)
cleaner = FuelCleanerEngine()

# 2. Lọc 1 điểm dữ liệu thời gian thực (Streaming point-by-point)
result = cleaner.clean_point(
    vehicle_id="29H-75028",
    timestamp="2026-08-15 09:16:00",
    raw_fuel=93.3,
    speed=61.0,
    capacity_est=95.0,
)

print(f"Mức nhiên liệu sạch: {result['clean_fuel']} Lít")
print(f"Trạng thái AI:       {result['ai_state_desc']}")
print(f"Biến cố:             {result['event_label']}")
print(f"Cờ đổ xăng:          {result['is_refuel']}")
```

### Xử lý hàng loạt từ file CSV / Excel (Batch Processing):
```python
import pandas as pd
from src.sdk import FuelCleanerEngine

cleaner = FuelCleanerEngine()

# Đọc file dữ liệu lịch sử đo đạc
df = pd.read_csv("dulieu_xe.csv")

# Xử lý toàn bộ bảng và tự động sinh các cột sạch
df_cleaned = cleaner.clean_batch(df, capacity_est=850.0)

# Xuất ra file kết quả
df_cleaned.to_excel("ket_qua_loc_sach.xlsx", index=False)
```

---

## 4. KỊCH BẢN 3: TRIỂN KHAI TRỌN GÓI (DOCKER & SCRIPT 1-CLICK)
*(Dành cho quản trị viên hệ thống hoặc khách hàng muốn có ngay cả Backend lẫn Giao diện Web)*

### Cách 1: Triển khai bằng Docker (Khuyên dùng cho Server Linux/Ubuntu)
Đảm bảo máy chủ đã cài đặt Docker và Docker Compose:
```bash
# 1. Khởi động dịch vụ chạy ngầm 24/7
docker compose up -d

# 2. Xem nhật ký hoạt động (Logs)
docker compose logs -f

# 3. Dừng dịch vụ khi cần
docker compose down
```

### Cách 2: Triển khai 1-Click trên Windows (Không cần cài Docker)
- Mở thư mục dự án và nhấp đúp chuột vào file:
  **`run_enterprise.bat`**
- Script sẽ tự động kiểm tra Python, cài đặt thư viện thiếu, nạp file cấu hình và bật server.

---

## 5. CẤU HÌNH BẢO MẬT & CƠ SỞ DỮ LIỆU CHUẨN 3NF

### 5.1. File cấu hình môi trường (`.env`)
Tạo hoặc chỉnh sửa file `.env` tại thư mục gốc:
```ini
# Cổng mạng
HOST=0.0.0.0
PORT=8000

# Bảo mật API Key
API_KEY=vicomsat_secret_key_2026
REQUIRE_API_KEY=false    # Đặt true trên Production để bắt buộc xác thực

# CSDL chuẩn 3NF (Pluggable Adapter)
DATABASE_URL=sqlite:///./fuel_records.db

# Nếu kết nối vào PostgreSQL / TimescaleDB của Doanh nghiệp:
# DATABASE_URL=postgresql://user:password@localhost:5432/fuel_records

# Nếu Doanh nghiệp đã có DB riêng và không muốn hệ thống tự lưu:
# DATABASE_URL=NONE
```

### 5.2. Cấu trúc CSDL chuẩn 3NF (4 Bảng)
1. **`vehicles` (Thông tin xe)**:
   - `vehicle_id` (PK): Biển số xe.
   - `capacity_liters`: Dung tích bình chứa chuẩn (Lít). Lưu 1 lần duy nhất, tránh trùng lặp.
2. **`ai_signal_states` (Từ điển mã trạng thái AI)**:
   - `state_code` (PK): Mã trạng thái tiếng Anh (`UPWARD_SHIFT`, `STABLE_JITTER`,...).
   - `state_name_vi`: Diễn giải tiếng Việt chuẩn.
3. **`fuel_logs` (Nhật ký đo đạc thời gian thực)**:
   - `id` (PK): Khóa chính tự tăng.
   - `vehicle_id` (FK -> `vehicles`): Mã xe.
   - `timestamp`: Thời điểm đo.
   - `raw_fuel`, `clean_fuel`, `speed`, `lat`, `lng`.
   - `state_code` (FK -> `ai_signal_states`): Khóa ngoại liên kết trạng thái.
4. **`fuel_events` (Tổng hợp biến cố Đổ xăng / Rút dầu)**:
   - `id` (PK), `vehicle_id` (FK), `event_type` (`REFUEL` / `DRAIN`).
   - `start_time`, `end_time`, `start_fuel`, `end_fuel`, `change_liters`, `address`.

---

## 6. TỪ ĐIỂN DỮ LIỆU & MÃ TRẠNG THÁI

| Mã Trạng Thái (`ai_state`) | Tên Tiếng Việt (`ai_state_desc`) | Ý Nghĩa Kỹ Thuật | Hành Vi Bộ Lọc Kalman |
| :--- | :--- | :--- | :--- |
| **`UPWARD_SHIFT`** | Bơm/Đổ nhiên liệu | Mức xăng tăng thực tế do tiếp nhiên liệu | Nhảy mức tức thì (`jump_to_z`), cập nhật lượng xăng mới |
| **`DOWNWARD_SHIFT`** | Hụt dầu đột ngột | Mức xăng tụt giảm nhanh bất thường | Cảnh báo nghi rút trộm dầu (`is_drain = true`) |
| **`GRADUAL_CHANGE`** | Tiêu thụ khi chạy | Nhiên liệu giảm dần do động cơ hoạt động | Bám dốc tiêu thụ mượt mà, không tạo bậc thang |
| **`STABLE_JITTER`** | Xe đỗ ổn định | Xe đứng yên, rung động nhỏ quanh giá trị thực | Giữ mức phẳng tuyệt đối, triệt tiêu dao động |
| **`OSCILLATION_NOISE`** | Nhiễu sóng sánh/xung | Sóng sánh do quán tính hoặc xung điện cảm biến | Nén phẳng triệt để, không bị leo theo đỉnh xung nhọn |

---
*Tài liệu bàn giao bản quyền thuộc dự án Hệ thống Giám sát Nhiên liệu Vicomsat.*
