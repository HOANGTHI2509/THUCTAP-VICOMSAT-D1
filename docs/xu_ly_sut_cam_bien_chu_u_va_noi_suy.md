# BÁO CÁO KỸ THUẬT: XỬ LÝ SỤT CẢM BIẾN CHỮ U, GẮN NHÃN VÀ NỐI THẲNG NỘI SUY

Document Path: `d:\THUCTAP_VICOMSAT\docs\xu_ly_sut_cam_bien_chu_u_va_noi_suy.md`

---

## 1. ĐẶT VẤN ĐỀ VÀ BẢN CHẤT KỸ THUẬT

Trong hệ thống giám sát nhiên liệu xe vận tải, dữ liệu nhiên liệu thường xuất hiện các đoạn sụt bất thường có hình **chữ U** (kéo dài từ 10 phút đến hơn 2 tiếng). 

### Bản chất kỹ thuật của đợt sụt chữ U:
* **Nguyên nhân**: Do lái xe tắt máy nghỉ trưa, sụt áp nguồn cảm biến khi tắt khóa điện, hoặc xe đỗ ở độ nghiêng tạm thời.
* **Đặc điểm**: Mức nhiên liệu sụt đột ngột từ 6 L đến 30 L, duy trì ở mức thấp trong khoảng thời gian đỗ, sau đó **bật tăng trở lại sát mức cũ ngay khi nổ máy di chuyển**.
* **Yêu cầu xử lý**: Không được gọt bỏ dữ liệu thô, mà cần **phát hiện tự động, gắn cờ `SENSOR_DROPOUT` và tính toán đường thẳng nội suy tuyến tính (`CleanedFuel`) nối mượt từ mốc trước sụt sang mốc phục hồi**.

---

## 2. PHÂN BIỆT SỤT CẢM BIẾN (SENSOR_DROPOUT) VS TRỘM DẦU (DRAIN)

Để không bao giờ bị nhầm lẫn giữa sụt cảm biến và hút trộm dầu thật, hệ thống áp dụng cơ chế phân loại dựa trên **Cửa sổ kiểm chứng tương lai (Future Window from 30 mins to 6 hours)**:

| Tiêu chí phân biệt | SỤT CẢM BIẾN CHỮ U (SENSOR_DROPOUT) | TRỘM DẦU THỰC TẾ (DRAIN) |
| :--- | :--- | :--- |
| **Mức sụt ban đầu** | Sụt mạnh `z_before - z_drop >= 6 L`. | Sụt mạnh `z_before - z_drop >= 8 L`. |
| **Mức nhiên liệu tương lai** | **Bật tăng trở lại sát mức cũ** khi nổ máy chạy tiếp (`abs(z_recovery - z_before) <= 5 L`). | **Giữ nguyên ở mức thấp mới** hoặc tiếp tục giảm theo dốc tiêu hao đi xuống. |
| **Sự kiện Phục hồi (Recovery)** | CÓ điểm nẩy vọt tự nhiên về lại mức cũ. | KHÔNG BAO GIỜ có điểm nẩy vọt về lại mức cũ (trừ khi ghé cây xăng `REFUEL`). |
| **Hành động của thuật toán** | Gắn cờ `SENSOR_DROPOUT`, **nội suy đường thẳng mượt (`CleanedFuel`)**. | Gắn nhãn `DRAIN`, **bám sát mức thấp mới**, KHÔNG nối thẳng. |

---

## 3. THUẬT TOÁN GẮN NHÃN VÀ NỐI THẲNG NỘI SUY

### 3.1. Phân loại theo độ dài khoảng sụt:
* **Độ dài 1 mốc dữ liệu (5 phút)**: Được gán nhãn `SPIKE` (Nhiễu gai 1 điểm) -> Triệt tiêu 100% tại điểm đó.
* **Độ dài từ 2 mốc trở lên (>= 10 phút)**: Được gán nhãn `SENSOR_DROPOUT` -> Tiến hành **nội suy đường thẳng mượt**.

### 3.2. Công thức nội suy tuyến tính đường thẳng (`CleanedFuel`):
Khi xác định được điểm bắt đầu sụt `start_idx` (thời điểm `t_start`, giá trị `baseline_before`) và điểm phục hồi `recovery_idx` (thời điểm `t_recovery`, giá trị `baseline_after`), mọi mốc thời gian `t_step` ở giữa sẽ được nối thẳng theo công thức:

```
ratio = (t_step - t_start) / (t_recovery - t_start)
CleanedFuel = baseline_before + ratio * (baseline_after - baseline_before)
```

---

## 4. VÍ DỤ MINH HỌA SỐ LIỆU CHI TIẾT

### Ví dụ 1: Sụt cảm biến chữ U kéo dài 2 tiếng 20 phút (GẮN CỜ & NỐI THẲNG NỘI SUY)

* **Số liệu thực tế**:
  * `11:45`: `91.3 L` (Mốc chuẩn trước sụt `baseline_before = 91.3 L`)
  * `11:50`: `82.5 L` (Sụt `-8.8 L` do tắt máy nghỉ trưa)
  * `12:00 - 14:05`: `82.5 L -> 81.8 L -> 79.6 L` (Nằm ở đáy vùng 80 L)
  * `14:10`: `90.1 L` (Nổ máy di chuyển, bật tăng `+10.5 L`, `abs(90.1 - 91.3) = 1.2 L <= 5 L` -> Điểm Recovery!)

* **Kết quả xử lý của thuật toán**:
  * Gắn cờ cho đoạn 11:50 đến 14:05: `FuelAnomalyType = SENSOR_DROPOUT`.
  * **Đường nội suy mượt (`CleanedFuel`)**: Vẽ 1 đường thẳng mịn đi từ `91.3 L` (11:45) nối thẳng sang `90.1 L` (14:10), loại bỏ hoàn toàn khoảng lún 82 L.

---

### Ví dụ 2: Sụt cảm biến chữ U ngắn 15 phút (GẮN CỜ & NỐI THẲNG NỘI SUY)

* **Số liệu thực tế**:
  * `09:00`: `150.0 L` (Mốc trước sụt `baseline_before = 150.0 L`)
  * `09:05`: `138.0 L` (Sụt `-12.0 L` do xe đỗ nghiêng tạm thời 15 phút)
  * `09:10`: `138.2 L` (Nằm ở đáy)
  * `09:15`: `149.5 L` (Xe di chuyển tiếp, bật tăng trở lại `+11.3 L` -> Điểm Recovery!)

* **Kết quả xử lý của thuật toán**:
  * Độ dài khoảng sụt = 15 phút (`>= 10 phút`) -> Thỏa mãn `min_low_minutes = 10.0`.
  * Gắn cờ: `FuelAnomalyType = SENSOR_DROPOUT`.
  * **Đường nội suy mượt (`CleanedFuel`)**: Nối 1 đường thẳng từ `150.0 L` (09:00) sang `149.5 L` (09:15).

---

### Ví dụ 3: Trộm dầu thực tế 20 L (BÁM SÁT MỨC THẤP, KHÔNG NỐI THẲNG)

* **Số liệu thực tế**:
  * `22:00`: `150.0 L` (Mốc trước sụt `baseline_before = 150.0 L`)
  * `22:05`: `130.0 L` (Sụt `-20.0 L` do bị rút dầu)
  * `22:10 - 06:00`: `130.0 L -> 129.5 L -> 128.0 L` (Không có mốc nào nẩy tăng lại 150 L)

* **Kết quả xử lý của thuật toán**:
  * Không tìm thấy điểm Recovery -> Bác bỏ SENSOR_DROPOUT.
  * Gắn nhãn: `AI_State = DRAIN`.
  * **Bộ lọc bám sát mức thấp 130 L**, KHÔNG nối thẳng nội suy để ghi nhận mất 20 L dầu thật.

---

## 5. VỊ TRÍ MÃ NGUỒN VÀ HƯỚNG DẪN KIỂM TRA TRÊN DASHBOARD

* **File xử lý cốt lõi**: [`src/core/filters/anomaly_detector.py`](file:///d:/THUCTAP_VICOMSAT/src/core/filters/anomaly_detector.py)
  * Hàm `detect_and_clean(df)` tự động quét và tính toán cột `CleanedFuel` cùng nhãn `FuelAnomalyType`.
  * Tham số `min_low_minutes = 10.0` phút.
* **Hiển thị Dashboard Streamlit**: [`src/dashboard/app_dashboard_tienxuly.py`](file:///d:/THUCTAP_VICOMSAT/src/dashboard/app_dashboard_tienxuly.py)
  * Chọn xem đường **`Cleaned & Interpolated (Nối phẳng mượt Chữ U)`** trên biểu đồ Plotly để quan sát đường nối thẳng mượt qua các đoạn sụt cảm biến.
