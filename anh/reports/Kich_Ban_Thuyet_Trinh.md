# Kịch Bản Thuyết Trình: Xử Lý Nhiễu Dữ Liệu Cảm Biến Nhiên Liệu Theo Thời Gian Thực

**Lưu ý quan trọng cho các biểu đồ chèn vào Slide:**
> [!IMPORTANT]
> Tất cả các biểu đồ sử dụng trong Slide PHẢI LÀ DỮ LIỆU THẬT trích xuất từ hệ thống. 
> Trục hoành (Trục X) của biểu đồ phải hiển thị rõ Ngày/Tháng/Năm và Giờ/Phút (Ví dụ: `2024-11-01 08:00`) để chứng minh tính thực tế của dữ liệu. Hãy cấu hình lại các file `generate_slide...py` của bạn để format lại trục X bằng thư viện `matplotlib.dates`.

---

## PHẦN 1: TỪ DỮ LIỆU THÔ ĐẾN DỮ LIỆU SẠCH (Tiền xử lý & Trích xuất)

### Slide 1: Tiền xử lý - "Dọn rác" trước khi lọc nhiễu
- **Nội dung trên Slide:** 
  - Sơ đồ khối: Kiểm tra Null/Trùng lặp $\rightarrow$ Tính TimeGap $\rightarrow$ Gắn cờ lỗi (Quality Flags) $\rightarrow$ Chia SegmentID.
  - ![Biểu đồ minh họa ngắt đứt đồ thị (TimeGap & SegmentID)](file:///d:/THUCTAP_VICOMSAT/slide9_2_timegap_segment.png)
- **Kịch bản nói (Speaker Notes):** 
  "Thưa thầy cô/anh chị, trước khi đưa dữ liệu vào bất kỳ thuật toán AI nào, nhóm đã tiến hành bước tiền xử lý nghiêm ngặt. Thay vì xóa dữ liệu lỗi bừa bãi, nhóm dùng các Cờ chất lượng để đánh dấu điểm nhiễu đột biến. Đặc biệt, nhóm dùng ngưỡng `TimeGap` để cắt chuỗi dữ liệu thành các đoạn `SegmentID` liên tục. Việc này giúp thuật toán phía sau không bị 'nối nhầm' dữ liệu qua đêm làm sai lệch kết quả đồ thị."

### Slide 2: Bắt mạch dữ liệu - Trích xuất đặc trưng
- **Nội dung trên Slide:**
  - Danh sách đặc trưng: Vận tốc (Chạy/Dừng), Biến động (DeltaFuel, FuelRate), Thống kê cửa sổ trượt.
  - Mục đích: Nhận diện biến động, phân biệt trạng thái xe, hỗ trợ điều chỉnh bộ lọc.
  - ![Biểu đồ kép Nhiên liệu và Vận tốc](file:///d:/THUCTAP_VICOMSAT/slide6_noise_analysis.png)
- **Kịch bản nói (Speaker Notes):** 
  "Sau khi có data sạch, nhóm tính toán các đặc trưng vi phân. Quan trọng nhất là tương quan giữa Vận tốc và Độ dao động nhiên liệu. Các đặc trưng này đóng vai trò là 'bộ cảm giác' giúp thuật toán nhận biết: lúc nào xe đang đỗ (mặt xăng phẳng lặng), lúc nào xe đang chạy (xăng sóng sánh dữ dội), từ đó làm cơ sở để tự động điều chỉnh bộ lọc ở bước sau."

---

## PHẦN 2: CÁC VŨ KHÍ LỌC NHIỄU (Thuật toán)

### Slide 3: Giới thiệu thuật toán cơ bản (Moving Average & Median)
- **Nội dung trên Slide:** 
  - Moving Average (MA): San phẳng tốt nhưng BỊ TRỄ nhịp.
  - Median Filter: Khử gai nhiễu điện từ cực tốt nhưng không mượt toàn cục.
- **Kịch bản nói (Speaker Notes):** 
  "Nhóm bắt đầu thử nghiệm với 2 bộ lọc truyền thống. Moving Average làm mượt rất tốt nhưng có nhược điểm chí mạng là sinh ra độ trễ cao, phản ứng chậm khi nhiên liệu bị rút trộm đột ngột. Còn Median Filter thì rất giỏi diệt nhiễu gai điện từ, nhưng lại làm đường đồ thị tổng thể bị giật cục, thiếu độ mượt."

### Slide 4: Trái tim của hệ thống - Standard Kalman Filter
- **Nội dung trên Slide:**
  - Sơ đồ nguyên lý: Dự đoán (Predict) $\leftrightarrow$ Cập nhật (Update).
  - Vai trò của 2 tham số: **Q** (Nhiễu hệ thống - Tin vào toán học) và **R** (Nhiễu đo lường - Tin vào cảm biến).
- **Kịch bản nói (Speaker Notes):** 
  "Giải pháp tối ưu nhóm tập trung nghiên cứu là Kalman Filter. Sức mạnh của nó nằm ở việc liên tục 'Dự đoán tương lai' và dùng 'Thực tế' để sửa sai qua biến Kalman Gain. Bằng cách tinh chỉnh 2 ma trận Q và R, chúng ta có thể cân bằng được giữa Độ mượt và Độ trễ của tín hiệu."

---

## PHẦN 3: ĐẤU TRƯỜNG THUẬT TOÁN (Thử nghiệm & Đánh giá)

### Slide 5: Kịch bản thử nghiệm và So sánh kết quả
- **Nội dung trên Slide:** 
  - ![Biểu đồ đè 3 đường thuật toán](file:///d:/THUCTAP_VICOMSAT/slide9_features_illustration.png)
- **Kịch bản nói (Speaker Notes):** 
  "Đây là kết quả thực nghiệm khi nhóm áp cả 3 thuật toán lên cùng một đoạn dữ liệu thực tế. Như mọi người thấy ở điểm sụt giảm khoanh đỏ, trong khi Moving Average vẫn đang chạy lạch bạch theo sau (bị trễ nhịp), thì Kalman Filter đã lập tức phản ứng bám sát sự sụt giảm thực tế nhờ cơ chế Kalman Gain, đảm bảo tính realtime rất cao."

### Slide 6: Bảng đánh giá tổng quan
- **Nội dung trên Slide:** Bảng Matrix (Tốt, Khá, Kém) so sánh 3 thuật toán dựa trên:
  1. Khả năng giảm nhiễu.
  2. Bảo toàn xu hướng thực tế.
  3. Độ trễ (Latency).
  4. Thời gian tính toán xử lý.
- **Kịch bản nói (Speaker Notes):** 
  "Tổng kết lại, Standard Kalman Filter mang lại hiệu năng cân bằng tuyệt vời nhất: Nó vượt trội về khả năng bảo toàn xu hướng thực, bám sát các sự kiện thay đổi đột ngột với độ trễ thấp nhất, cực kỳ phù hợp cho ứng dụng theo dõi thời gian thực."

---

## PHẦN 4: HƯỚNG ĐI TƯƠNG LAI (Cải tiến)

### Slide 7: Nâng cấp lên Adaptive Kalman Filter
- **Nội dung trên Slide:** 
  - Hạn chế của Standard Kalman: Cài đặt tham số tham số R (nhiễu đo lường) cứng (tĩnh).
  - Hướng cải tiến: Adaptive Kalman Filter - Biến thiên tham số R tự động theo Vận Tốc (Speed) và Mức nhiễu cục bộ (Rolling Std).
- **Kịch bản nói (Speaker Notes):** 
  "Tuy nhiên, việc cài đặt cứng ma trận R ở Standard Kalman là chưa phản ánh đúng vật lý ngoài đời. Thực tế, nhiễu khi đỗ xe và khi chạy là hoàn toàn khác nhau. Hướng cải tiến tiếp theo của nhóm là sử dụng chính các đặc trưng Vận tốc và Cửa sổ trượt đã làm ở Phần 1 để xây dựng 'Adaptive Kalman Filter'. Hệ thống sẽ tự động hạ R (lọc nhẹ) khi xe đỗ để phản ứng cực nhanh nếu có trộm, và tự động tăng R (lọc mạnh) khi xe đang chạy trên đường xóc để triệt tiêu sóng sánh."
