# AI Smooth-Tracking

Bo loc causal realtime cua duong mau tim. Moi `vehicle_id` co mot state rieng va
khong su dung diem tuong lai.

## Cau truc

- `config.py`: tham so Q/R, nguong van toc, jitter va chuyen muc.
- `state.py`: state tung xe va cac kieu bang chung cua so.
- `features.py`: rolling statistics, directionality, trend va feature cho model.
- `kalman.py`: chon Q/R thich nghi va cap nhat Kalman.
- `engine.py`: dieu phoi mot diem realtime, candidate va quality flag.
- `dataframe.py`: adapter cho dashboard va xu ly DataFrame.

## Public API

```python
from src.core.filters.smooth_tracking import (
    AISmoothTrackingFilter,
    SmoothTrackingConfig,
)

config = SmoothTrackingConfig(moving_r=45.0, moving_q=0.08)
engine = AISmoothTrackingFilter(config=config)
result = engine.process_point(
    vehicle_id="29E-44284",
    timestamp="2026-09-09 13:30:57",
    raw_fuel=500.0,
    speed=45.0,
    capacity_est=800.0,
)
```

File `ai_smooth_tracking_filter.py` chi la facade tuong thich nguoc. Code moi nen
import tu package `smooth_tracking`.

## Nguyen tac tinh chinh

- Tang `R`: it tin raw hon, muot hon va tre hon.
- Giam `R`: bam raw nhanh hon.
- Tang `Q`: thich nghi nhanh hon voi thay doi muc that.
- Giam `Q`: on dinh hon nhung co the bi i.

Moi thay doi config can chay `tests/test_smooth_tracking_noise_symmetry.py` de
kiem tra cac doan U, doi nho, nhieu hai chieu va xu huong tieu hao.
