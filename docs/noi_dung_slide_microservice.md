# Nội dung slide — Real-time Fuel Data Denoising & Filtering

## Slide 1 — Đề tài và mục tiêu

**Xử lý nhiễu dữ liệu cảm biến và lọc tín hiệu nhiên liệu theo thời gian thực**

- Đầu vào: mức nhiên liệu đã calib sang lít, thời gian, GPS và tốc độ.
- Đầu ra: mức nhiên liệu sạch, nhãn trạng thái tín hiệu, confidence và quality flag.
- Mục tiêu triển khai: chạy theo luồng từng điểm dữ liệu, theo từng xe.

## Slide 2 — Vấn đề cần xử lý

- Xe rung lắc, tăng/giảm tốc, lên/xuống dốc làm tín hiệu nhiên liệu dao động mạnh.
- Cảm biến có thể tạo spike, dao động ngắn hạn hoặc mất tín hiệu xuống gần 0L.
- Dùng trực tiếp dữ liệu raw làm sai trải nghiệm hiển thị và các phân tích tầng trên.

**Hình minh họa:** một biểu đồ Raw FuelLevel bị răng cưa từ dashboard.

## Slide 3 — Pipeline causal realtime

```text
Raw Fuel + Time + GPS + Speed
        ↓
Feature causal (hiện tại và quá khứ)
        ↓
RF Causal v3: nhận diện trạng thái tín hiệu
        ↓
AI-Enhanced Adaptive Kalman Realtime
        ↓
CleanFuelRealtime + AI_SignalState + Confidence + QualityFlag
```

**Thông điệp:** pipeline không dùng dữ liệu tương lai nên có thể chạy realtime.

## Slide 4 — Taxonomy nhãn AI

| Nhãn | Ý nghĩa kỹ thuật |
|---|---|
| `STABLE_JITTER` | Dao động nhỏ quanh mức ổn định |
| `OSCILLATION_NOISE` | Dao động/sóng sánh/nhiễu cảm biến |
| `GRADUAL_CHANGE` | Thay đổi giảm dần có xu hướng |
| `UPWARD_SHIFT` | Dịch chuyển mức tăng cần xác nhận |
| `DOWNWARD_SHIFT` | Dịch chuyển mức giảm cần xác nhận |

**Ghi chú:** dùng tên trung tính của Đề tài 1, không kết luận nghiệp vụ.

## Slide 5 — Random Forest Causal v3

- Model Random Forest sử dụng 20 feature causal.
- Ví dụ: FuelLevel, Speed, TimeGapMinutes, DeltaFuel, RollingStd12, DistanceMeters, GPS speed, ngưỡng noise và PrevMedian3.
- Đã loại các feature nhìn tương lai: FutureMedian3/5, ReturnToPrevLevel, LocalRange, reversal flags, TransientScore.

| Metric test | Giá trị |
|---|---:|
| Accuracy | 0.96 |
| Macro-F1 | 0.88 |
| STABLE_JITTER F1 | 0.98 |
| GRADUAL_CHANGE F1 | 0.93 |
| OSCILLATION_NOISE F1 | 0.89 |

**Ghi chú:** DOWNWARD_SHIFT có ít mẫu train, cần bổ sung nhãn thực tế để đánh giá chắc chắn hơn.

## Slide 6 — Function Matrix / Confusion Matrix có ground truth

- Dataset test đã gắn nhãn có confusion matrix theo xe.
- Hai xe benchmark chính: `90H-03494`, `92H-02687`.
- `Car 5` dùng làm regression test mất tín hiệu, không dùng để kết luận metric chính.

**Ảnh dùng trên slide:**

- `models/rf_signal_state_causal_v3/test_90h-03494_confusion_matrix.png`
- `models/rf_signal_state_causal_v3/test_92h-02687_confusion_matrix.png`

## Slide 7 — Replay realtime trên hai trace raw mới

- Chạy riêng RF Causal v3 + `ai_enhanced_adaptive_realtime.py` trên hai file raw mới.
- `TEST DO DOC.csv`: mean absolute step giảm từ **0.7605** xuống **0.3784**.
- `2026-08-27T00-48_export.csv`: mean absolute step giảm từ **1.1291** xuống **0.6366**.

**Ảnh dùng trên slide:**

- `TestDoDoc/results_new_raw_v3/TEST DO DOC_realtime_chart_v3.svg`
- `TestDoDoc/results_new_raw_v3/2026-08-27T00-48_export_realtime_chart_v3.svg`

## Slide 8 — State Transition Matrix trên raw trace

- Hai file raw không có nhãn ground truth nên không dùng để báo cáo accuracy.
- Thay vào đó dùng State Transition Matrix: nhãn RF ở điểm trước → nhãn RF ở điểm hiện tại.
- Mục đích: quan sát độ ổn định chuyển trạng thái khi replay dữ liệu thực.

**Ảnh dùng trên slide:**

- `TestDoDoc/results_new_raw_v3/2026-08-27T00-48_export_state_transition_matrix.png`
- `TestDoDoc/results_new_raw_v3/TEST DO DOC_state_transition_matrix.png`

## Slide 9 — Microservice hiện có

**FastAPI prototype đã có:**

```text
POST /api/v1/fuel/clean-point
POST /api/v1/fuel/clean-batch
GET  /api/v1/health
GET  /api/v1/vehicles
POST /api/v1/vehicles/{vehicle_id}/reset-state
```

**Input:** `vehicle_id`, `fuel_time`, `fuel_level`, `speed`, `lat`, `lng`.

**Output:** `clean_fuel_liters`, `ai_signal_state`, `confidence`, `quality_flag`, `latency_ms`.

**Thông điệp:** API đã xử lý raw point/batch qua RF Causal v3 và realtime Kalman.

## Slide 10 — Quản lý state theo vehicle_id

- Service đã giữ history feature và Kalman state riêng theo từng `vehicle_id`.
- Tự reset state khi gap dữ liệu vượt 120 phút.
- Hiện state lưu trong RAM; phù hợp demo/one instance.
- Khi triển khai nhiều instance hoặc cần chịu restart: chuyển state sang Redis.

```text
vehicle_id → history + RF feature context + Kalman state
```

## Slide 11 — Mức độ hoàn thành và bước tiếp theo

| Hạng mục | Trạng thái |
|---|---|
| Taxonomy nhãn và RF Causal v3 | Đã có |
| AI-Enhanced Adaptive Kalman realtime | Đã có |
| Dashboard/replay biểu đồ | Đã có |
| API FastAPI raw → clean fuel | Đã có prototype |
| State theo xe trong RAM | Đã có |
| Ground truth Vcomsat xác nhận | Cần bổ sung |
| Redis persistence / multi-instance | Cần triển khai |
| Load test, auth, monitoring, retry/idempotency | Cần hoàn thiện trước production |

## Slide 12 — Kết luận

> Đã xây dựng được lõi xử lý causal realtime: RF Causal v3 nhận diện trạng thái tín hiệu và Adaptive Kalman tạo mức nhiên liệu sạch theo từng xe.

> Bước chuyển giao doanh nghiệp: xác nhận ground truth, chuẩn hóa vận hành API và chuyển state từ RAM sang Redis.
