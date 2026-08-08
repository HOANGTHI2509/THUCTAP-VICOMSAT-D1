# Kịch Bản Thuyết Trình Chi Tiết: Chuyên sâu phần Tiền Xử Lý Dữ Liệu

*(File này chứa các lời thoại mở rộng, đi sâu vào bản chất kỹ thuật của 5 bước Tiền xử lý. Bạn có thể dùng để nói diễn giải dài hơn trên slide, hoặc dùng làm tài liệu để trả lời câu hỏi phản biện của hội đồng).*

---

### Mở đầu phần Tiền xử lý
"Thưa hội đồng, dữ liệu cảm biến thực tế gửi về từ các xe tải/xe khách là một mớ dữ liệu vô cùng hỗn độn và nhiễu loạn. Nếu ném trực tiếp dữ liệu thô này vào thuật toán AI, hệ thống sẽ sụp đổ hoặc đưa ra kết quả sai lệch hoàn toàn. Vì vậy, nhóm em đã xây dựng một đường ống (Pipeline) Tiền xử lý gồm 5 bước cốt lõi sau đây:

### 1. Chuẩn hóa phân luồng (Tách VehicleID)
**Kịch bản nói chi tiết:**
"Bước đầu tiên là phân luồng dữ liệu. Trên thực tế, Server nhận một luồng data tổng hợp đan xen từ hàng trăm xe khác nhau. Nếu không tiến hành gom nhóm (Group by) theo từng `VehicleID`, thuật toán sẽ đọc tuần tự và lấy mức xăng của Xe tải A đem đi trừ cho mức xăng của Xe khách B. Hậu quả là sinh ra những bước nhảy vọt hàng trăm lít một cách vô lý. Việc bóc tách luồng riêng biệt đảm bảo tính toàn vẹn dữ liệu, xe nào xử lý theo xe đó, không bao giờ bị tính toán chéo."

### 2. Làm sạch cơ bản (Loại bỏ Null và Trùng lặp)
**Kịch bản nói chi tiết:**
"Tiếp theo là bước làm sạch các lỗi phần cứng cơ bản. Rất nhiều trường hợp thiết bị 3G bị nghẽn mạng, sau đó gửi dồn dập 2-3 bản ghi có cùng chung một mốc thời gian (Trùng lặp), hoặc bị mất gói tin khiến tọa độ/xăng bị bỏ trống (Null).
Nhóm em bắt buộc phải dùng code quét qua để xóa các điểm trùng lặp này và xử lý các giá trị Null. Lý do cực kỳ quan trọng: Lõi của bộ lọc Kalman là các phép nhân ma trận toán học. Chỉ cần lọt vào một giá trị Null (NaN) hoặc chia cho thời gian bằng 0 (do 2 điểm trùng thời gian), toàn bộ hệ thống ma trận sẽ báo lỗi (Crash) và ngừng hoạt động ngay lập tức."

### 3. Khắc phục nhiễu trôi tọa độ (GPS Drift bằng Deadband)
**Kịch bản nói chi tiết:**
"Đây là một lỗi vật lý cố hữu của mọi định vị GPS được gọi là Sai số đa đường (Multipath Error). Khi xe đỗ ở bãi, sóng vệ tinh đập vào các tòa nhà cao tầng hoặc mây mù khiến tọa độ bị nhiễu, nhảy cóc loạn xạ trên bản đồ. Hậu quả là thiết bị báo về vận tốc ảo (3-5 km/h) dù xe đang tắt máy đứng im.
Để trị lỗi này, nhóm em thiết lập một thuật toán **Ngưỡng chết (Deadband)**. Nếu tốc độ nhỏ hơn 5 km/h, hệ thống lập tức ép nó về bằng 0. Bước nắn chỉnh này mang tính chất sống còn! Vì nhờ có nó, bộ lọc Kalman ở phía sau mới không bị lừa, nhận diện chuẩn xác lúc nào xe đang thực sự đứng yên (để lọc nhẹ) và lúc nào xe đang chạy (để lọc mạnh)."

### 4. Thiết lập Cờ chất lượng (Quality Flags)
**Kịch bản nói chi tiết:**
"Quan điểm của nhóm em ở bước này là: **Tuyệt đối không xóa bừa bãi dữ liệu**. Nếu xóa mất, Nhóm 2 sẽ không còn bằng chứng để bắt trộm. Thay vì xóa, nhóm em dùng Cờ (Flag) để đánh dấu.
Ví dụ như cờ `FlagLargeDelta` (Sụt giảm bất thường). Nhóm không hề gán cứng một con số như 10 lít hay 20 lít, bởi vì độ dao động của một chiếc xe con khác hoàn toàn với một chiếc xe bồn 1000 lít. Nhóm đã dùng hàm thống kê `Quantile 99%` để thuật toán **tự động học và tìm ra ngưỡng nhảy vọt riêng biệt cho từng xe**. Những điểm bị cắm cờ này, bộ lọc mượt sẽ chủ động 'từ chối' xử lý để không làm bóp méo đường trung bình, nhưng vẫn giữ nguyên giá trị gốc trong file để Nhóm 2 điều tra."

### 5. Cắt đoạn liên tục (SegmentID bằng TimeGap)
**Kịch bản nói chi tiết:**
"Bước cuối cùng, và cũng là bước đắt giá nhất trong tiền xử lý: Cắt đoạn dữ liệu.
Trong thực tế, khi xe chui vào hầm Hải Vân hoặc tài xế tắt máy qua đêm, tín hiệu sẽ bị đứt quãng nhiều tiếng đồng hồ. Nếu chúng ta nhắm mắt để cho thuật toán tự động kẻ một đường thẳng nối điểm tối hôm qua với điểm sáng hôm nay, chúng ta đang **bịa ra dữ liệu**, sinh ra ảo giác là xăng bị rò rỉ từ từ suốt đêm.
Nhóm em khắc phục bằng cách tính khoảng thời gian trống `TimeGap`. Cứ mất tín hiệu quá 30 phút, hệ thống chủ động cầm kéo cắt đứt đôi đồ thị, sinh ra 2 đoạn `SegmentID` rời rạc. Việc ngắt đoạn này vừa giúp thuật toán làm mượt được reset lại trí nhớ, vừa bảo toàn nguyên vẹn khoảng trống chân thực nhất. Và chính cái khoảng trống chân thực này là manh mối duy nhất để Nhóm 2 soi vào và phát hiện ra hành vi hút trộm xăng trong hầm tối!"
