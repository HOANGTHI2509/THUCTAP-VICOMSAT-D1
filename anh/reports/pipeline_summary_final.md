# Tổng Kết Quy Trình Tiền Xử Lý Dữ Liệu (Phục vụ Đề tài 1: Lọc Nhiễu)

Tài liệu này ghi chú lại toàn bộ các bước đã thực hiện, các vấn đề logic đã khắc phục và bộ đặc trưng cuối cùng được chốt để phục vụ cho **Đề tài 1: Xử lý nhiễu dữ liệu cảm biến và lọc nhiễu tín hiệu nhiên liệu theo thời gian thực**.

---

## 1. Luồng Xử Lý (Pipeline Logic)

Quy trình được triển khai bằng Python (pandas) chạy độc lập trên từng xe (từng sheet), trải qua các bước chặt chẽ sau:

1. **Chuẩn hóa ban đầu:** Ép kiểu thời gian (`datetime`) và kiểu số (`numeric`) cho các cột tín hiệu. Sắp xếp dữ liệu theo trình tự thời gian.
2. **Cách ly dữ liệu ảo (Zero-value isolation):** 
   - Đánh cờ `FlagFuelZero = 1` cho các bản ghi có mức xăng = 0.
   - Tạo một bản clone dữ liệu (`FuelLevel_calc`) trong đó các mức xăng = 0 bị biến thành `NaN`. Việc này ngăn chặn thuật toán lấy số 0 để tính toán và sinh ra các biến thiên giả mạo hàng trăm lít ở dòng kế tiếp.
3. **Phân mảnh chuỗi thời gian (Segmentation):**
   - Tính `TimeGapMinutes` giữa các bản ghi.
   - Nếu khoảng cách > 30 phút, đánh cờ `IsSegmentStart = 1` và tạo một `SegmentID` mới để chia cắt chuỗi dữ liệu, tránh việc thuật toán tính toán vắt ngang qua khoảng thời gian xe tắt máy hoặc mất sóng.
4. **Tính toán đặc trưng vi phân & thống kê (Chỉ tính trong nội bộ từng SegmentID):**
   - `DeltaFuel`, `FuelRate`.
   - Tính toán cửa sổ trượt (Sliding Window): `RollingMedian` (xu hướng mượt) và `RollingStd` (mức độ nhiễu). Cửa sổ được **reset hoàn toàn về rỗng** mỗi khi sang một `SegmentID` mới.
5. **Gắn cờ trạng thái & Chất lượng:**
   - Dựa vào Speed để tạo cờ `MovementState` (Moving / Stopped / Uncertain).
   - Tính toán ngưỡng dao động tự động (Adaptive Threshold) dựa trên phân vị 99% (`P99`) của từng xe để gán cờ `FlagLargeDelta` (cảnh báo biến động bất thường).
   - Tổng hợp tất cả các lỗi thành cờ `QualityFlag` và chuỗi lý do `QualityReason`.

---

## 2. Tám (8) Đặc Trưng Cốt Lõi Được Chọn Đưa Vào Báo Cáo

Bộ 8 đặc trưng này được tinh gọn để bám sát mục tiêu của Đề tài 1 (Đo mức nhiễu, nhận biết trạng thái vận hành, hỗ trợ bộ lọc Kalman):

| Tên Cột | Ý nghĩa / Tác dụng |
| :--- | :--- |
| `TimeGapMinutes` | Khoảng thời gian (bước nhảy) giữa 2 mẫu. Quan trọng cho phương trình dự đoán trạng thái của Kalman. |
| `SegmentID` | Đảm bảo tính liên tục của chuỗi. Cảnh báo cho bộ lọc cần khởi tạo lại trạng thái (Re-initialize) khi chuyển đoạn. |
| `DeltaFuel` | Đo mức biến thiên / dao động nhiên liệu. |
| `FuelRate` | Chuẩn hóa mức biến thiên nhiên liệu theo thời gian thực (Lít/Phút). |
| `RollingMedian` | Đại diện cho mức nhiên liệu cục bộ (đã loại trừ gai nhiễu), làm cơ sở so sánh (baseline). |
| `RollingStd` | **Quan trọng nhất**: Đo mức dao động trong cửa sổ 5 mẫu. Trực tiếp làm đầu vào cho "Ma trận phương sai nhiễu đo lường" trong Kalman Filter. |
| `MovementState` | Phân biệt trạng thái (Stopped/Moving) để bộ lọc áp dụng linh hoạt các hệ số nhiễu khác nhau (Adaptive filtering). |
| `QualityFlag` | (0 hoặc 1). Gắn cờ 1 nếu bản ghi có bất cứ biểu hiện dị thường nào (Xăng=0, Mất kết nối lâu, Xăng giật quá ngưỡng P99). Hỗ trợ loại bỏ rác. |

---

## 3. Các Lỗi Logic (Edge Cases) Kinh Điển Đã Khắc Phục

Để đạt được độ chuẩn xác 100%, quy trình đã khắc phục các "cạm bẫy" sau (Rất tốt để trình bày bảo vệ trước hội đồng):

*   **Lỗi DeltaFuel khổng lồ giả mạo:** Đã khắc phục bằng cách đặt điểm `FuelLevel = 0` thành `NaN`. Dòng liền kề sau đó sẽ được gán `FeatureStatus = PREVIOUS_INVALID` và không tính Delta, cắt đứt hoàn toàn cú giật ảo.
*   **Lỗi chưa đủ mẫu đã kết luận nhiễu = 0:** Ở dòng đầu tiên của cửa sổ trượt, `RollingStd` sẽ trả về `NaN` thay vì `0.0`. Hệ thống có thêm cờ `RollingWindowReady` để báo hiệu khi nào gom đủ 3 mẫu trở lên thì mô hình AI mới được phép bắt đầu tính toán.
*   **Lỗi rò rỉ dữ liệu (Data Leakage) giữa các lần ngắt kết nối:** Hàm `Rolling` đã được khóa chặt bằng `groupby('SegmentID')`. Dữ liệu của 30 phút trước khi xe chui vào hầm sẽ không bao giờ được phép cộng dồn với dữ liệu khi xe đi ra khỏi hầm.
*   **Ngưỡng nhiễu động (Adaptive P99):** Không dùng 1 con số (Ví dụ 15 lít) để ép cho tất cả các xe. Hệ thống tự quét và học ngưỡng dao động cực đại (P99) của từng xe riêng biệt (VD: Xe tải to dao động 12 lít là bình thường, Xe con dao động 5 lít là bất thường).

---
*(Tài liệu này được tạo tự động để lưu trữ. Anh/chị có thể xem thêm source code tại `preprocess_pipeline.py` và bảng kết quả đầu ra `CarFuelHistory_Processed_*.csv`)*
