## 1. Bảng tổng quan dữ liệu
| Xe | Số bản ghi | Thời gian dữ liệu | Số ngày | Fuel Min | Fuel Median | Fuel Max | Tỷ lệ Speed = 0 |
|---|---|---|---|---|---|---|---|
| Car 1 | 69,474 | 2025-11 đến 2026-07 | 244 | 0.0 | 167.2 | 371.8 | 90.7% |
| Car 2 | 63,331 | 2025-11 đến 2026-07 | 244 | 1.8 | 105.3 | 198.9 | 88.3% |
| Car 3 | 69,541 | 2025-11 đến 2026-07 | 244 | 0.0 | 211.5 | 359.8 | 91.7% |
| Car 4 | 69,476 | 2025-11 đến 2026-07 | 244 | 0.0 | 167.2 | 371.8 | 90.7% |
| Car 5 | 13,861 | 2025-11 đến 2026-07 | 245 | 72.8 | 287.6 | 600.0 | 77.4% |


## 2. Bảng chất lượng dữ liệu
| Xe | Thiếu dữ liệu | Timestamp trùng | Fuel = 0 | GPS không hợp lệ | Speed bất thường | Khoảng mất dữ liệu |
|---|---|---|---|---|---|---|
| Car 1 | 0 | 0 | 2 (0.00%) | 0 | 0 | 13 (0.02%) |
| Car 2 | 0 | 0 | 0 | 0 | 0 | 122 (0.19%) |
| Car 3 | 0 | 0 | 39 (0.06%) | 0 | 0 | 10 (0.01%) |
| Car 4 | 0 | 0 | 2 (0.00%) | 0 | 0 | 13 (0.02%) |
| Car 5 | 0 | 0 | 0 | 27 (0.19%) | 0 | 305 (2.20%) |


## 3. Phân tích tần suất truyền
| Xe | TimeGap phổ biến | TimeGap trung vị | Tỷ lệ đúng nhịp | Khoảng lớn nhất | Số gap > 30 phút |
|---|---|---|---|---|---|
| Car 1 | 5.0 ph | 5.0 ph | 99.4% | 2554 ph | 13 |
| Car 2 | 5.0 ph | 5.0 ph | 97.9% | 9368 ph | 122 |
| Car 3 | 5.0 ph | 5.0 ph | 99.6% | 2649 ph | 10 |
| Car 4 | 5.0 ph | 5.0 ph | 99.4% | 2554 ph | 13 |
| Car 5 | 5.0 ph | 5.0 ph | 94.3% | 18219 ph | 305 |


## 4. Đặc điểm tín hiệu nhiên liệu thô
| Chỉ số | Car 1 | Car 2 | Car 3 | Car 4 | Car 5 |
|---|---|---|---|---|---|
| Median |DeltaFuel| | 0.400 | 0.000 | 0.400 | 0.400 | 0.000 |
| P95 |DeltaFuel| | 1.600 | 1.000 | 2.300 | 1.600 | 8.400 |
| Max |DeltaFuel| | 201.5 | 178.9 | 278.8 | 201.5 | 443.0 |


## 5. Mối liên hệ Fuel - Speed - GPS
| Xe | Median |ΔFuel| khi dừng | Median |ΔFuel| khi chạy | GPS dịch chuyển khi Speed = 0 (lần) | Số mâu thuẫn Speed-GPS |
|---|---|---|---|---|
| Car 1 | 0.300 | 1.000 | 2,299 | 2,299 |
| Car 2 | 0.000 | 0.500 | 3,271 | 3,271 |
| Car 3 | 0.300 | 1.400 | 1,776 | 1,776 |
| Car 4 | 0.300 | 1.000 | 2,300 | 2,300 |
| Car 5 | 0.000 | 2.600 | 1,083 | 1,083 |


