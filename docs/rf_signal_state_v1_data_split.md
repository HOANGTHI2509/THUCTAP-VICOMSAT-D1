# RF Signal-State v1 — Chia dữ liệu train/validation/test

## Taxonomy nhãn của Đề 1

Model nhận diện trạng thái tín hiệu, không kết luận nghiệp vụ đổ/rút nhiên liệu:

- `UPWARD_SHIFT`
- `DOWNWARD_SHIFT`
- `GRADUAL_CHANGE`
- `STABLE_JITTER`
- `OSCILLATION_NOISE`

`SPIKE`, `DROPOUT` và các nhãn `TRANSIENT_*` lịch sử được gộp vào
`OSCILLATION_NOISE`.

## Nguồn dữ liệu

Các tập được tạo từ `data/fuel_label_dataset/`:

| Tập | File | Số dòng | Tỷ lệ trên tổng 312,570 dòng |
|---|---|---:|---:|
| Train | `train.csv` | 268,916 | 86.03% |
| Validation | `val.csv` | 20,171 | 6.45% |
| Test | `test.csv` | 23,483 | 7.51% |

Khi train, lớp `STABLE_JITTER` được giới hạn lấy mẫu để giảm mất cân bằng;
dataset train sau cân bằng có 58,755 dòng. Validation và test giữ phân bố gốc.

## Chia theo xe

Tập test được tách theo xe, không trộn điểm dữ liệu của các xe test vào train:

| Xe test | Số dòng | Vai trò |
|---|---:|---|
| `90H-03494` | 4,979 | Benchmark phân loại chính |
| `92H-02687` | 4,806 | Benchmark phân loại chính |
| `Car 5` | 13,698 | Chỉ dùng regression test `SENSOR_DROPOUT`; loại khỏi benchmark AI chính vì mất tín hiệu nhiều |

Train và validation gồm các xe còn lại sau khi tách ba xe test trên. Validation
được dùng để theo dõi metric trong quá trình train; test chỉ dùng báo cáo cuối.

## Kết quả đánh giá chính

Khi báo cáo theo xe, chỉ sử dụng `90H-03494` và `92H-02687`. `Car 5` không
được gộp vào accuracy/F1 benchmark vì tỷ lệ dropout cao làm sai lệch đánh giá
khả năng nhận diện tín hiệu hợp lệ.

> Lưu ý: model v1 vẫn có các feature nhìn tương lai (`FutureMedian3`,
> `FutureMedian5`, `ReturnToPrevLevel`); do đó metric v1 là benchmark offline.
> Bản realtime-causal tiếp theo phải bỏ các feature này và đánh giá lại.

## Danh sách xe thực tế theo tập

### Train (23 xe)

- `12H-04470`
- `19H-06956`
- `20B-27762`
- `21C-04064`
- `21H-02058`
- `21H-03052`
- `21H-03221`
- `24H-03439`
- `24H-04058`
- `24H-05088`
- `29E-44284`
- `29E-45520`
- `29E-45560`
- `29E-51878`
- `29H-41394`
- `35H-09245`
- `35H-14767`
- `36C-31893`
- `92H-02653`
- `92H-07095`
- `Car 1`
- `Car 2`
- `Car 3`

### Val (1 xe)

- `Car 3`

### Test (3 xe)

- `90H-03494`
- `92H-02687`
- `Car 5`
