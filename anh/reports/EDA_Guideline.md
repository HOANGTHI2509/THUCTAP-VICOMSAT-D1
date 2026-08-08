# Hướng Dẫn Phân Tích Dữ Liệu Khám Phá (EDA) - Lọc Nhiễu Nhiên Liệu

Với đặc thù của dự án **Lọc nhiễu cảm biến nhiên liệu**, bước EDA mang đậm tính chất xử lý chuỗi thời gian (Time-series Analysis) và Xử lý tín hiệu (Signal Processing). 

Dưới đây là 5 đầu việc cụ thể bạn cần code và vẽ ra (nên sử dụng Jupyter Notebook để dễ trực quan):

## 1. Khám sức khỏe tổng quát dữ liệu (Data Integrity)
- **Việc cần làm:** Đếm số dòng, số cột. Kiểm tra xem có bao nhiêu điểm dữ liệu bị khuyết thiếu (Null/NaN) ở cột `FuelLevel` và `Speed`. Tìm các dòng bị trùng lặp thời gian (`FuelTime` giống hệt nhau).
- **Mục đích:** Để biết dữ liệu thô "sạch" hay "bẩn" ở mức độ nào. Nếu có dòng trùng lặp, bạn phải viết code xóa dòng trùng lặp (drop duplicates) trước khi làm bước tiếp theo.

## 2. Phân tích Tần suất lấy mẫu & Phân phối TimeGap
- **Việc cần làm:** Tính toán cột `TimeGap` (Khoảng cách thời gian giữa 2 lần gửi). Sau đó vẽ biểu đồ Histogram của `TimeGap`.
- **Mục đích:** 
  - Xem bình thường xe gửi dữ liệu bao nhiêu giây một lần? (10 giây, 30 giây hay 1 phút/lần). Điều này rất quan trọng để cấu hình tần số lấy mẫu cho bộ lọc Kalman.
  - Nhìn trên biểu đồ xem các điểm đứt gãy (xe tắt máy) thường kéo dài bao lâu. Từ đó bạn chốt được **ngưỡng cắt SegmentID** (Ví dụ: Nếu đa số ngắt quãng dài hơn 15 phút, chốt `TimeGap > 15 phút` thì cắt sang Segment mới).

## 3. Trực quan hóa Mối quan hệ: Vận tốc vs Nhiễu xăng (Quan trọng nhất)
- **Việc cần làm:** Chọn ra 1 xe bất kỳ trong 1 ngày. Vẽ một biểu đồ kép (2 trục y hoặc 2 biểu đồ trên dưới): 
  - Đường phía trên: Mức nhiên liệu (`FuelLevel`).
  - Đường phía dưới: Vận tốc (`Speed`).
- **Mục đích:** Đây là bằng chứng vật lý cốt lõi của bài toán. Bạn phải dùng mắt để nhìn thấy sự tương quan: **Cứ đoạn nào vận tốc lớn hơn 0, đường mức xăng ở trên lập tức giật hình răng cưa (Sóng sánh). Đoạn nào vận tốc bám đáy 0, đường xăng đi ngang êm ái.** 

## 4. Truy tìm Outliers (Gai nhiễu đột biến) để đặt Cờ
- **Việc cần làm:** Vẽ Boxplot (Biểu đồ hộp) hoặc Histogram cho cột `DeltaFuel` (Mức chênh lệch xăng). Lọc ra các điểm dữ liệu có `Speed > 150 km/h` hoặc `FuelLevel = 0`.
- **Mục đích:** Để tìm ra các giới hạn vật lý phi lý. 
  - Ví dụ bạn phát hiện có lúc xăng nhảy vọt lên 80 lít chỉ trong 10 giây. Bạn sẽ dùng con số này để thiết lập luật cho cờ: `Nếu |DeltaFuel| > 20 lít / 10s -> Kích hoạt FlagLargeDelta`.

## 5. Khảo sát biên độ nhiễu (Noise Variance) để nạp cho Kalman Filter
- **Việc cần làm:** Lọc lấy toàn bộ các đoạn xe Đang Chạy (`Speed > 0`). Tính độ lệch chuẩn (`Std`) của `DeltaFuel` trong các đoạn này. Sau đó làm tương tự với đoạn xe Đang Dừng (`Speed = 0`).
- **Mục đích:** 
  - Tìm ra ma trận nhiễu đo lường ($R$) cho thuật toán Kalman. 
  - Ví dụ EDA cho thấy: Khi dừng, xăng dao động $\pm 1$ lít (Nhiễu điện từ). Khi chạy, xăng dao động $\pm 8$ lít (Sóng sánh). Bạn sẽ lấy ngay số $1$ và số $8$ này ném vào thuật toán Kalman lúc code mô hình để nó tự động thay đổi hệ số nhiễu (Adaptive Filtering) tùy thuộc vào xe đang đỗ hay chạy.

---
**💡 KẾT LUẬN:** 
EDA bài này không phải là vẽ biểu đồ tròn/cột phân tích doanh thu, mà là vẽ **đồ thị hình sin, hình răng cưa theo thời gian** để "bắt mạch" xem cảm biến nhiễu nặng nhẹ ra sao, từ đó mới cấu hình được bộ lọc chuẩn xác.
