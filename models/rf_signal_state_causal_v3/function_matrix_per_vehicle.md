# RF Causal v3 — Function Matrix theo xe

Tập test chính: 90H-03494 và 92H-02687. Car 5 được loại khỏi bảng chính do mất tín hiệu cảm biến thường xuyên. Dấu — nghĩa là nhãn không có mẫu thực tế trên xe đó, nên không dùng để kết luận metric riêng theo xe.

| Vehicle | Label | Support | Precision | Recall | F1 | Mean confidence |
|---|---|---:|---:|---:|---:|---:|
| 90H-03494 | UPWARD_SHIFT | 2 | 100.00% | 50.00% | 66.67% | 69.79% |
| 90H-03494 | DOWNWARD_SHIFT | 0 | — | — | — | — |
| 90H-03494 | GRADUAL_CHANGE | 241 | 100.00% | 99.59% | 99.79% | 98.59% |
| 90H-03494 | STABLE_JITTER | 4699 | 100.00% | 99.74% | 99.87% | 98.92% |
| 90H-03494 | OSCILLATION_NOISE | 37 | 72.55% | 100.00% | 84.09% | 93.04% |
| 92H-02687 | UPWARD_SHIFT | 0 | — | — | — | — |
| 92H-02687 | DOWNWARD_SHIFT | 0 | — | — | — | — |
| 92H-02687 | GRADUAL_CHANGE | 576 | 99.83% | 99.83% | 99.83% | 99.30% |
| 92H-02687 | STABLE_JITTER | 4158 | 100.00% | 99.35% | 99.67% | 98.19% |
| 92H-02687 | OSCILLATION_NOISE | 72 | 71.43% | 97.22% | 82.35% | 95.16% |
