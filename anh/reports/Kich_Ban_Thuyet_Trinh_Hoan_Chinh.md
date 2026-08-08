# Kịch bản Thuyết trình (Bản Học thuật - An toàn)

---

## Slide 1 — Quy trình tiền xử lý dữ liệu

### Nội dung trên slide

```text
1. Dữ liệu cảm biến thô
    ↓
2. Gắn VehicleID và chuẩn hóa kiểu dữ liệu
    ↓
3. Sắp xếp theo FuelTime
    ↓
4. Kiểm tra dữ liệu thiếu và trùng
    ↓
5. Kiểm tra FuelLevel, Speed, GPS
    ↓
6. Gắn QualityFlag
    ↓
7. Tính TimeGap và chia SegmentID
    ↓
8. Dữ liệu hợp lệ cho bước trích xuất đặc trưng
```
*Ghi chú: Speed ≤ 5 km/h được quy về trạng thái dừng. Việc cắt Segment dựa trên **Ngưỡng động**: TimeGap > k × (chu kỳ gửi trung bình cục bộ của từng xe), với k=3 (dùng Rolling Median để tránh bị ảnh hưởng bởi outlier).*

### Kịch bản nói

> Kính thưa hội đồng, để dữ liệu thô từ cảm biến có thể sử dụng được cho các thuật toán học máy và bộ lọc, nhóm em đã xây dựng một Pipeline tiền xử lý vô cùng chặt chẽ gồm 8 bước.
>
> Đầu tiên (Bước 1 đến 3), dữ liệu thô được phân tách theo từng mã xe (VehicleID), sau đó chuẩn hóa kiểu dữ liệu và sắp xếp tăng dần theo thời gian. Đây là nguyên tắc bắt buộc để đảm bảo tính nhân quả (causality) của dữ liệu.
>
> Tại bước 4 và 5, nhóm không xóa bỏ các điểm dữ liệu dị thường như thiếu sóng, trùng lặp thời gian hay nhiên liệu tụt về 0. Thay vào đó (tại Bước 6), nhóm giữ nguyên chúng và "gắn cờ chất lượng" (QualityFlag). Điều này giúp thuật toán ở phía sau biết được đâu là vùng dữ liệu kém tin cậy để tự động giảm trọng số, thay vì làm mất đi tính liên tục của thời gian. Đối với Vận tốc (Speed), nhóm áp dụng deadband, coi mọi vận tốc ≤ 5 km/h là trạng thái dừng để lọc bỏ nhiễu rung sai số của GPS.
>
> Đặc biệt, tại bước 7 (chia Segment), để giải quyết bài toán mất sóng khi xe đi qua hầm hoặc vùng núi, nhóm đã áp dụng phương pháp **Ngưỡng Động (Adaptive Threshold)** kết hợp thuật toán **Haversine Distance**. Thay vì dùng một ngưỡng cố định 15 phút cho mọi xe, hệ thống tính toán Chu kỳ gửi chuẩn cục bộ của từng xe bằng hàm Rolling Median. Nếu xe mất kết nối quá 3 lần chu kỳ này, VÀ khoảng cách tọa độ GPS không thay đổi (tức là xe đỗ thật sự), hệ thống mới tiến hành cắt đứt Segment. Nếu xe vẫn đang di chuyển, hệ thống chỉ gắn cờ "Mất sóng tạm thời" chứ không cắt vụn dữ liệu. 
>
> Nhờ 8 bước tiền xử lý chuẩn Công nghiệp này (Bước 8), dữ liệu đưa vào trích xuất đặc trưng luôn sạch sẽ và đạt độ tin cậy cao nhất.

---

## Slide 2 — Minh họa dữ liệu sau tiền xử lý (Ví dụ tiêu biểu)

### Nội dung trên slide

*Bảng dữ liệu thực tế trích xuất từ **Car 1** (Tháng 11 & 12 năm 2025)*

| VehicleID | FuelTime | FuelLevel (L) | Speed (km/h) | TimeGap (phút) | SegmentID | QualityFlag | IsSegmentStart |
|---|---|---|---|---|---|---|---|
| Car 1 | 19/11/2025 - 23:00:29 | **0.0** | 0.0 | - | 1 | **FUEL_ZERO** | TRUE |
| Car 1 | 19/11/2025 - 23:00:59 | 191.4 | 0.0 | 0.50 | 1 | VALID | FALSE |
| ... | ... | ... | ... | ... | ... | ... | ... |
| Car 1 | 09/12/2025 - 14:18:42 | 158.4 | 0.0 | 8.50 | 1 | VALID | FALSE |
| Car 1 | 09/12/2025 - 14:20:42 | 157.9 | 0.0 | 2.00 | 1 | VALID | FALSE |
| Car 1 | 09/12/2025 - 14:44:09 | 155.2 | 0.0 | **23.45** | **2** | **LONG_GAP** | **TRUE** |
| Car 1 | 09/12/2025 - 14:45:12 | 155.4 | 0.0 | 1.05 | 2 | VALID | FALSE |

**Kết luận:** Dữ liệu gốc được giữ nguyên; các điểm bất thường được nhận diện thông qua `QualityFlag`. Dữ liệu đứt gãy được chủ động phân tách thành các Segment độc lập thông qua `IsSegmentStart`.

### Kịch bản nói

> Để hội đồng dễ hình dung 8 bước tiền xử lý vừa rồi hoạt động ra sao, nhóm em xin trích xuất một đoạn dữ liệu thực tế của **Car 1** trong tháng 11 và tháng 12 năm 2025.
> 
> Nhìn vào bảng, thầy cô có thể thấy 3 trường hợp điển hình đã được xử lý tự động:
> * **Thứ nhất:** Vào lúc 23:00 ngày 19/11, cảm biến bị lỗi trả về mức nhiên liệu 0. Thay vì xóa dòng này gây đứt gãy thời gian, hệ thống gắn cờ **FUEL_ZERO** và vô hiệu hóa nó khỏi bước tính toán.
> * **Thứ hai:** Lúc 14:20 ngày 09/12, xe đi rất chậm (2.5 km/h). Hệ thống kích hoạt Deadband, tự động quy nó về trạng thái **STOPPED** để tránh nhiễu rung lắc của GPS.
> * **Thứ ba:** Ngay sau đó, xe bị mất tín hiệu tới 23.5 phút. Thuật toán phát hiện đây là khoảng trễ vượt quá Chu kỳ chuẩn, nên lập tức gắn cờ **LONG_GAP** và chủ động cắt vỡ dòng thời gian sang **SegmentID số 2**. Tất cả các trạng thái đặc trưng sẽ được Reset lại từ đầu.
>
> Nhờ cách tiếp cận này, dữ liệu gốc luôn được bảo toàn trọn vẹn, không bị mất mát thông tin lịch sử, rất thuận lợi cho việc truy vết lỗi cảm biến sau này!

---

## Slide 4 — Đặc trưng biến thiên nhiên liệu

### Nội dung trên slide

Hiển thị:
* FuelLevel
* DeltaFuel
* AbsDeltaFuel
* FuelRate

Công thức:
$\Delta Fuel_t = Fuel_t - Fuel_{t-1}$
$FuelRate_t = \frac{\Delta Fuel_t}{\Delta Time_t}$

*Ghi chú: Các đặc trưng chỉ mô tả mức biến động, chưa kết luận nguyên nhân của biến động.*

### Kịch bản nói

> Sau bước tiền xử lý, nhóm trích xuất các đặc trưng mô tả sự thay đổi của tín hiệu nhiên liệu. DeltaFuel cho biết mức thay đổi giữa hai mẫu liên tiếp, còn FuelRate chuẩn hóa sự thay đổi theo khoảng thời gian giữa các mẫu.
>
> Những đoạn tín hiệu ổn định thường có DeltaFuel và FuelRate nhỏ. Khi xuất hiện spike hoặc một thay đổi mức kéo dài, độ lớn của các đặc trưng này tăng lên rõ rệt.
>
> Tuy nhiên, đây mới chỉ là dấu hiệu biến động của tín hiệu. Nhóm không sử dụng DeltaFuel để khẳng định nguyên nhân là nạp hay rút nhiên liệu.

---

## Slide 4 — Đặc trưng chuyển động và thống kê cục bộ

### Nội dung trên slide

**Chuyển động**
```text
MovementState = STOPPED nếu Speed ≤ 5 km/h
MovementState = MOVING nếu Speed > 5 km/h
```

**Thống kê cục bộ**
```text
RollingStd — độ lệch chuẩn trên 5 mẫu gần nhất
```
*(Biểu đồ 3 phần: FuelLevel, Speed/MovementState, RollingStd)*

### Kịch bản nói

> Nhóm đặc trưng tiếp theo mô tả bối cảnh chuyển động và mức dao động cục bộ. MovementState được suy ra từ Speed sau khi áp dụng deadband, giúp thuật toán biết xe đang ở trạng thái dừng hay di chuyển theo dữ liệu hiện có.
>
> RollingStd được tính trên một cửa sổ gồm 5 mẫu quá khứ, thể hiện mức biến động của FuelLevel trong vùng lân cận. RollingStd lớn cho biết tín hiệu đang dao động mạnh, nhưng chưa đủ để kết luận nguyên nhân là đường xóc, cảm biến hay thay đổi mức nhiên liệu.
>
> Các đặc trưng này được sử dụng để hỗ trợ bộ lọc điều chỉnh mức độ tin tưởng vào phép đo theo từng thời điểm.

---

## Slide 5 — Nguyên lý và thử nghiệm Moving Average

### Nội dung trên slide

**Bố cục (2 cột):**

**Cột trái (Nội dung chữ):**
**1. Nguyên lý**
Moving Average lấy trung bình của N mẫu gần nhất để làm mượt tín hiệu.
$y_t = \frac{1}{N}\sum_{i=0}^{N-1}x_{t-i}$

**2. Cách dùng trong bài toán**
* Áp dụng theo từng SegmentID.
* Chỉ dùng mẫu hiện tại và quá khứ → phù hợp realtime.
* Cửa sổ thử nghiệm: N = 10.

**3. Ưu điểm**
* Đơn giản, dễ triển khai.
* Làm mượt dao động ngắn hạn.
* Làm baseline để so sánh.

**4. Hạn chế**
* Có độ trễ.
* Làm mờ các thay đổi nhanh.
* Cửa sổ càng lớn tín hiệu càng mượt nhưng phản ứng càng chậm.

**Cột phải (Biểu đồ):**
*Ảnh minh họa (Chèn biểu đồ slide5_ma_real.png vào đây)*
*(Ghi chú dưới ảnh: Trích xuất từ **Car 1**, giai đoạn **25/11/2025 - 26/11/2025**)*

### Kịch bản nói

> *(Chỉ vào Cột trái)*
> Để có cơ sở đánh giá độ hiệu quả, nhóm đã thiết lập Moving Average làm phương pháp baseline đầu tiên. Bộ lọc này đơn giản là lấy trung bình của N mẫu gần nhất trong cùng một SegmentID, tuyệt đối không dùng dữ liệu tương lai để đảm bảo tính realtime.
>
> *(Chỉ vào Cột phải - Biểu đồ)*
> Nhìn vào biểu đồ áp dụng trên dữ liệu thực tế của Car 1, hội đồng có thể thấy rất rõ hai đặc tính của Moving Average:
> * Ở **vùng màu xanh**, biên độ dao động lởm chởm đã được làm mượt khá tốt, thể hiện đúng ưu điểm của phương pháp.
> * Tuy nhiên ở **vùng màu đỏ**, khi lượng nhiên liệu thực sự bị giảm nhanh, đường Moving Average (màu xanh dương) phản ứng không kịp và tạo ra một khoảng trễ rất lớn so với đường gốc.
> 
> Đây là một hạn chế chí mạng. Nếu ta tăng cửa sổ N lên để làm mượt vùng xanh, thì vùng đỏ lại càng bị trễ nặng hơn, không đáp ứng được yêu cầu cảnh báo theo thời gian thực!

---

## Slide 6 — Median Filter

### Nội dung trên slide

**Median Filter**
$y_t = \operatorname{median}(x_t, x_{t-1}, \ldots, x_{t-N+1})$
* Chống spike đơn lẻ tốt.
* Có thể tạo đầu ra dạng bậc thang.
* Không mô hình hóa trạng thái hệ thống.

### Kịch bản nói

> Nhóm cũng thử nghiệm Median Filter với kỳ vọng loại bỏ các spike nhiễu tốt hơn. Tuy nhiên, Median Filter thường biến tín hiệu đầu ra thành các đoạn bậc thang cứng nhắc và gặp khó khăn tương tự khi mức nhiên liệu thay đổi liên tục. Do đó, cần một bộ lọc thông minh hơn như Kalman Filter.

---

## Slide 7 — Vai trò của Q, R và cách sử dụng Kalman Filter

### Nội dung trên slide

*Ảnh minh họa (Chèn 2 biểu đồ slide5_qr_impact_real.png vào đây)*
*(Ghi chú dưới ảnh: Trích xuất từ **Car 1**, giai đoạn **25/11/2025 - 26/11/2025**)*
**Bố cục (2 cột):**
- **Cột trái:** Tác động của Q (Giữ R = 9 cố định). Biểu diễn đường Raw, Kalman (Q nhỏ = 0.01) và Kalman (Q lớn = 5).
- **Cột phải:** Tác động của R (Giữ Q = 1 cố định). Biểu diễn đường Raw, Kalman (R nhỏ = 1) và Kalman (R lớn = 100).

**Cách sử dụng thuật toán:**
*(4 ô vuông hiển thị ở dưới cùng)*
[ Mỗi VehicleID một trạng thái ] - [ Reset tại Segment mới ] - [ Bỏ qua Update tại điểm lỗi ] - [ Baseline Q = 1, R = 9 ]
*(Nhãn nổi bật bên cạnh)*: **Causal – phù hợp xử lý realtime**

### Kịch bản nói

> *(Chỉ vào biểu đồ bên trái)*
> Khi giữ R cố định, Q nhỏ làm bộ lọc tin trạng thái trước nhiều hơn nên đường đầu ra mượt nhưng bám thay đổi chậm. Q lớn làm bộ lọc chấp nhận trạng thái thay đổi nhanh hơn nên phản ứng nhanh hơn, nhưng đầu ra có thể dao động nhiều hơn.
>
> *(Chỉ vào biểu đồ bên phải)*
> Khi giữ Q cố định, R nhỏ làm Kalman tin phép đo nhiều hơn nên bám đường raw nhanh. R lớn làm bộ lọc ít tin cảm biến hơn, vì vậy đầu ra mượt hơn nhưng phản ứng chậm hơn.
>
> Về cách triển khai thực tế, hệ thống duy trì mỗi VehicleID một trạng thái độc lập và sẽ được **Reset hoàn toàn** khi bước sang một Segment mới. Thuật toán hoạt động theo nguyên tắc nhân quả (Causal), tức là chỉ sử dụng dữ liệu hiện tại và quá khứ, phù hợp tuyệt đối cho xử lý Realtime.

---

## Slide 8 — So sánh trực quan Moving Average và Standard Kalman

### Nội dung trên slide

*Ảnh minh họa (Chèn biểu đồ anh/slide7_comparison.png vào đây)*
*(Ghi chú dưới ảnh: Trích xuất từ **Car 1**, giai đoạn **25/11/2025 - 26/11/2025**)*

**(Văn bản ngắn gọn trên Slide):**
* **Moving Average:** Mượt nhưng có độ trễ.
* **Standard Kalman:** Phản ứng nhanh hơn nhưng phụ thuộc Q, R.
* **Kết luận:** Standard Kalman bám tín hiệu tốt hơn, Moving Average làm mượt đơn giản hơn.

### Kịch bản nói

> Chắc hẳn hội đồng đang thắc mắc: "Vậy rốt cuộc Moving Average và Standard Kalman khác nhau như thế nào khi chạy thực tế?". Để trả lời câu hỏi đó, nhóm xin trình bày biểu đồ so sánh trực tiếp cả hai thuật toán trên cùng một đoạn dữ liệu của Car 1.
>
> *(Chỉ vào vùng Xanh lá)* 
> Ở **đoạn dao động ngắn hạn**, cả hai thuật toán đều hoàn thành xuất sắc nhiệm vụ làm mượt tín hiệu.
> 
> *(Chỉ vào vùng Cam)*
> Tuy nhiên, ở **đoạn thay đổi mức nhanh** (tín hiệu vọt lên đột ngột), Moving Average bộc lộ rõ yếu điểm khi bị trễ một đoạn rất dài. Trong khi đó, Standard Kalman phản ứng và bắt kịp đường gốc nhanh hơn hẳn.
>
> *(Chỉ vào vùng Tím)*
> Đặc biệt, khi gặp **điểm bất thường hoặc spike ngắn**, Standard Kalman cũng ít bị kéo lệch xuống dưới hơn so với Moving Average.
>
> **Kết luận:** Standard Kalman tỏ ra ưu việt hơn trong việc bám sát tín hiệu biến đổi nhanh, nhưng bù lại nó yêu cầu phải tinh chỉnh tham số Q và R rất kỹ lưỡng. Đây chính là tiền đề để nhóm phát triển bộ lọc Adaptive Kalman ở phần sau!

---

## Slide 9 — Đánh giá tổng quan (Tất cả kịch bản gộp chung)

### Nội dung trên slide

**Kết quả đánh giá trên toàn bộ tập dữ liệu Car 1**

| Thuật toán | Mean ∣ΔOutput∣ (Mức dao động) ↓ | Tracking RMSE ↓ | Max Deviation ↓ |
|---|---|---|---|
| **Raw Data** | 0.711 | 0.000 | 0.000 |
| **Moving Average (N=10)** | 0.373 | 8.964 | 223.960 |
| **Median Filter (N=10)** | 0.373 | 10.846 | 262.800 |
| **Standard Kalman (Q=1, R=9)** | 0.400 | 5.412 | 163.930 |
| **Adaptive Kalman (Nhóm đề xuất)**| **0.390** | **1.333** | **15.264** |

*Chú thích:*
> Tracking RMSE và Max deviation được tính so với tín hiệu tham chiếu hiện có. Bộ dữ liệu chưa có ground truth về mức nhiên liệu thật.
> Mean (|\Delta Output|) = Mean Absolute First Difference.

### Kịch bản nói

> Kính thưa hội đồng, để đánh giá chính xác, nhóm đề xuất 3 chỉ số:
> 1. **Mean |ΔOutput|**: Đo mức độ dao động liên tiếp (chỉ số càng nhỏ càng mượt).
> 2. **Tracking RMSE**: Đo độ bám sát tín hiệu gốc.
> 3. **Max Deviation**: Đo sai số trễ lớn nhất ở các sự kiện đột biến.
> 
> Bảng kết quả trên toàn bộ xe Car 1 cho thấy một **nghịch lý (Trade-off)** rất rõ ràng:
> - **Moving Average** làm mượt rất tốt (đạt 0.373), nhưng đổi lại độ trễ cực cao, sai số cực đại lên tới hơn 220 lít.
> - **Standard Kalman (R=9)** cố gắng bám sát tín hiệu để giảm độ trễ (RMSE tụt xuống 5.41), nhưng lập tức phải trả giá bằng việc mất đi độ mượt (chỉ số dao động tăng lên 0.400). Và dù vậy, nó vẫn bị trễ nặng ở các sự kiện nạp/rút xăng bất ngờ (lệch tới 164 lít).
>
> Câu hỏi đặt ra là: Liệu chúng ta có thể vừa làm mượt cực tốt, lại vừa bám sát và không bị trễ? Đó chính là tiền đề để nhóm phát triển thuật toán tiếp theo!

---

## Slide 10 — Kết quả cải thiện của Adaptive Kalman

### Nội dung trên slide

**Bố cục 2 cột:**

**Cột Trái: Biểu đồ so sánh**
*(Chèn ảnh anh/Slide10_Adaptive_Comparison.png)*
* **Đường tín hiệu:** Raw FuelLevel, Standard Kalman, Adaptive Kalman.
* **Vùng được đánh dấu:**
  * **Spike đơn lẻ:** Adaptive Kalman ít bị kéo lệch hơn nhờ Gating.
  * **Thay đổi kéo dài:** Adaptive Kalman bám mức mới nhanh hơn nhờ Memory.

**Cột Phải: Bảng kết quả & Nhận xét**

| Chỉ số | Standard Kalman | Adaptive Kalman | Cải thiện |
|---|---:|---:|---:|
| **Mean (\|ΔOutput\|)** ↓ | 0.400 | 0.390 | **2.5%** |
| **Tracking RMSE** ↓ | 5.412 L | 1.333 L | **75.4%** |
| **Max Deviation** ↓ | 163.9 L | 15.2 L | **90.7%** |

*Ghi chú: Các chỉ số được đo trên cùng tập bản ghi, cùng SegmentID và điều kiện xử lý của Car 1.*

**Kết quả cải tiến:**
* Giảm sai lệch trung bình khi bám tín hiệu.
* Hạn chế rõ rệt sai lệch cực đại.
* Vẫn duy trì mức dao động đầu ra thấp.

**Kết luận:**
> Adaptive Kalman cân bằng tốt hơn giữa độ mượt, khả năng bám và khả năng hạn chế sai lệch bất thường.

### Kịch bản nói

> Sang Slide 10, chúng ta sẽ xem xét sự cải thiện thực tế của Adaptive Kalman bằng cách kết hợp cả biểu đồ và bảng chỉ số.
>
> Ở biểu đồ bên trái, mọi người có thể thấy rõ hai vùng khoanh tròn. Ở vùng **Spike đơn lẻ (nhiễu)**, Adaptive Kalman ít bị kéo lệch hơn hẳn nhờ cơ chế Gating tự động chặn nhiễu. Ở vùng **Thay đổi kéo dài (xăng thực sự tụt)**, Adaptive Kalman "bứt tốc" và bám lấy mức mới cực kỳ nhanh nhờ cơ chế Memory, trái ngược với độ trễ kéo dài của Standard Kalman.
>
> Bảng chỉ số bên phải chính là bằng chứng đanh thép nhất. Adaptive Kalman giúp giảm tới **75.4%** sai lệch bám theo tín hiệu (RMSE) và hạn chế tới **90.7%** sai lệch cực đại (Max Deviation). 
> 
> **Kết luận:** Adaptive Kalman chính là lời giải hoàn hảo, cân bằng tốt hơn giữa độ mượt, khả năng bám sát và khả năng hạn chế sai lệch bất thường.

---

## Slide 11 — Demo hệ thống và hướng phát triển

### Nội dung trên slide

**Kết quả hiện tại:**
* Pipeline tiền xử lý theo từng VehicleID (Tự động cắt Segment, Gắn cờ QualityFlag).
* Trích xuất đặc trưng theo thời gian.
* Bốn thuật toán lọc causal.
* Prototype mô phỏng luồng realtime trên dashboard.

**Hạn chế và Hướng cải tiến (Quan trọng):**
* Chưa có ground truth mức nhiên liệu thật.
* **Ngưỡng cắt Segment đang quá ưu tiên an toàn:** Việc bắt buộc phải đủ điều kiện "Đỗ xe thật sự" (Vận tốc = 0, Tọa độ lệch < 50m) khiến thuật toán bỏ lọt các đoạn mất sóng rất dài (VD: Có trường hợp mất sóng 73 tiếng, xe di chuyển 45km khi mất kết nối nhưng không bị cắt Segment).
* **Đề xuất:** Cần bổ sung thêm "Ngưỡng trần tuyệt đối" (Hard Ceiling) bên cạnh Ngưỡng động. VD: Nếu mất sóng > 12 tiếng thì tự động cắt Segment bất chấp điều kiện vận tốc/tọa độ, để cân bằng giữa độ an toàn và khả năng kiểm soát độ dài Segment tối đa.

**Hướng phát triển dài hạn:**
* Tạo dữ liệu mô phỏng có ground truth.
* Huấn luyện ML để dự đoán tham số $R_t$ thay vì dùng luật cứng.
* Triển khai API streaming thực tế.

### Kịch bản nói

> Nhóm đã xây dựng một prototype gồm pipeline tiền xử lý, các bộ lọc causal và dashboard so sánh kết quả. 
>
> Tuy nhiên, trong quá trình phân tích kỹ các ngoại lệ (như Car 5), nhóm phát hiện thuật toán cắt đoạn (Segmentation) hiện đang ưu tiên an toàn quá mức. Cụ thể, thuật toán chỉ cắt Segment khi có đủ bằng chứng xe đứng yên (Speed=0 và tọa độ lệch < 50m). Điều này giúp tránh cắt nhầm khi xe đang đi qua vùng mất sóng, nhưng lại dẫn đến việc bỏ lọt các khoảng gián đoạn rất dài. Ví dụ thực tế: có lần xe mất kết nối tới 73 tiếng, trong khoảng thời gian đó xe bị di dời 45km. Vì vi phạm điều kiện tọa độ, hệ thống không cắt Segment mà chỉ gắn cờ khả nghi.
> 
> Hướng cải tiến thiết thực nhất là nhóm đề xuất bổ sung thêm "Ngưỡng trần tuyệt đối". Ví dụ, theo luật giao thông, tài xế không được lái xe liên tục quá 10 tiếng/ngày. Do đó, nếu mất kết nối vượt quá 12 tiếng, hệ thống sẽ tự động cắt Segment bất kể điều kiện Speed/tọa độ. Điều này giúp giảm thiểu rủi ro segment kéo dài bất hợp lý, dù phải đánh đổi một phần độ chính xác ở các trường hợp xe chạy đường dài hiếm gặp.
>
> **Kết luận:** Kết quả của nhóm không nhằm khẳng định đã dò được chính xác 100% nhiên liệu thật (vì thiếu Ground Truth), mà chứng minh rằng việc kết hợp tiền xử lý chặt chẽ và Adaptive Kalman đã giải quyết được nghịch lý giữa Độ mượt và Độ trễ trên bộ dữ liệu hiện tại. Em xin cảm ơn hội đồng đã lắng nghe!

---

## Slide 12 (Backup) — Lý do sử dụng Ngưỡng Động (Adaptive TimeGap)

### Nội dung trên slide

**Minh họa sự khác biệt về chu kỳ gửi tín hiệu giữa các xe:**

| Đặc điểm | Chu kỳ chuẩn (Trung vị) | Hệ số k=3 | Ngưỡng cắt động (Adaptive Threshold) |
| :--- | :---: | :---: | :---: |
| **Xe số 1** | 5 phút / lần | $\times 3$ | **> 15 phút** (Trễ 15p mới cắt) |
| **Xe số 2 (Đoạn A)** | 3 phút / lần | $\times 3$ | **> 9 phút** (Trễ 9p đã cắt do chu kỳ vốn nhanh) |
| **Xe số 2 (Đoạn B)** | 8 phút / lần | $\times 3$ | **> 24 phút** (Trễ 24p mới cắt do chu kỳ vốn chậm) |

### Kịch bản nói

> *(Slide này dùng để trả lời câu hỏi: "Tại sao không dùng 1 ngưỡng cố định 15 phút cho tất cả các xe?")*
> 
> Dạ thưa thầy/cô, trong quá trình phân tích dữ liệu thực tế, nhóm phát hiện ra rằng chu kỳ gửi dữ liệu của các thiết bị IoT không hề giống nhau. Có xe cài đặt 3 phút gửi 1 lần, có xe chạy đường dài lại cài 8 phút gửi 1 lần. Thậm chí cùng 1 xe, chu kỳ cũng thay đổi khi vào vùng sóng yếu.
>
> Nếu nhóm áp 1 ngưỡng cố định (ví dụ 15 phút):
> - Xe gửi 3 phút/lần mà mất sóng 15 phút (gấp 5 lần chu kỳ) thì ngưỡng này quá dễ dãi.
> - Xe gửi 8 phút/lần mà mất sóng 12 phút (gấp 1.5 lần chu kỳ, có thể chỉ là trễ nhẹ) thì ngưỡng này lại quá khắt khe, dẫn đến cắt vụn dữ liệu oan uổng.
>
> Nhờ việc tính toán chu kỳ gửi chuẩn (Baseline Interval) bằng Rolling Median cho từng xe, từng thời điểm, thuật toán của nhóm có thể linh hoạt siết chặt hoặc nới lỏng ngưỡng cắt, giải quyết triệt để vấn đề "nhiễu cấu trúc" (Structural Noise) đặc thù của hệ thống IoT.
> Nhóm đã xây dựng một prototype gồm pipeline tiền xử lý, trích xuất đặc trưng, các bộ lọc causal và dashboard so sánh kết quả theo từng xe. Trạng thái của Kalman được quản lý riêng cho từng VehicleID và được reset khi xuất hiện Segment mới.
>
> Hạn chế lớn nhất hiện nay là bộ dữ liệu chưa có ground truth về mức nhiên liệu thực tế. Vì vậy, kết quả hiện tại chủ yếu đánh giá độ mượt, độ bám và khả năng xử lý các trường hợp bất thường đã khảo sát.
>
> Trong giai đoạn tiếp theo, nhóm định hướng xây dựng dữ liệu mô phỏng có ground truth, đánh giá trên các Segment tách biệt và sử dụng mô hình học máy để dự đoán mức nhiễu hoặc tham số R cho Adaptive Kalman.
>
> **Kết luận:** Kết quả của nhóm không nhằm khẳng định đã xác định được mức nhiên liệu thật, mà chứng minh rằng việc kết hợp tiền xử lý, đặc trưng ngữ cảnh và Adaptive Kalman có thể cải thiện độ mượt và khả năng bám realtime so với các baseline trên bộ dữ liệu hiện có. Bước tiếp theo là kiểm chứng bằng ground truth hoặc dữ liệu mô phỏng có tín hiệu sạch. Em xin cảm ơn hội đồng đã lắng nghe!
