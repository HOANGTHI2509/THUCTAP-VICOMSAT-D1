# Đánh giá Baseline Random Forest

Confusion matrix hiện tại đủ tốt để đưa vào báo cáo baseline, nhưng cần nêu rõ hạn chế.

## Điểm tốt

- `STABLE_JITTER` nhận diện rất tốt: tập test đúng `8563/9082`, sai chủ yếu sang `SLOSHING_NOISE`. Đây là lỗi tương đối chấp nhận được vì hai lớp này gần nhau.
- `REFUEL` nhận diện rất tốt: tập test đúng `61/62`. Model bắt nạp nhiên liệu ổn.
- `DRAIN` nhận diện tốt: tập test đúng `110/118`. Model bắt rút/tụt nhiên liệu lớn khá ổn.
- `SLOSHING_NOISE` nhận diện khá tốt: tập test đúng `3404/3583`.

## Điểm yếu

- `SPIKE` là lớp yếu nhất: tập test đúng `71/123`.
- Lỗi chính:
  - `SPIKE -> SLOSHING_NOISE`: `46`
  - `SLOSHING_NOISE -> SPIKE`: `108`
- Điều này cho thấy model vẫn khó tách spike đơn lẻ với nhiễu sóng bình theo cụm. Đây cũng là khó khăn thực tế vì ranh giới giữa hai loại nhiễu này không rõ tuyệt đối.

## Nhận xét cho báo cáo

- Accuracy test đạt khoảng `0.93`, nhưng dataset lệch lớp mạnh vì `STABLE_JITTER` chiếm đa số.
- Vì vậy không nên chỉ báo cáo accuracy, cần báo cáo thêm precision/recall/F1-score theo từng lớp.
- Kết quả tốt nhất nằm ở các sự kiện quan trọng `REFUEL` và `DRAIN`.
- Hạn chế chính hiện tại là phân biệt `SPIKE` và `SLOSHING_NOISE`.

## Đoạn có thể đưa vào slide

Mô hình Random Forest đạt độ chính xác khoảng 93% trên tập kiểm thử. Các lớp `REFUEL` và `DRAIN` được nhận diện tốt với F1-score xấp xỉ 0.96, cho thấy mô hình có khả năng phát hiện các biến động nhiên liệu lớn như nạp và rút nhiên liệu. Tuy nhiên, lớp `SPIKE` có F1-score thấp hơn do dễ nhầm với `SLOSHING_NOISE`, phản ánh sự chồng lấn giữa nhiễu điểm đơn lẻ và nhiễu dao động liên tục trong dữ liệu cảm biến thực tế.
