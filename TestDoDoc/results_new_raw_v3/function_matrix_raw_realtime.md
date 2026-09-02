# RF Causal v3 — Function Matrix trên dữ liệu raw mới

Đây là inference không ground truth: chỉ báo cáo nhãn AI dự đoán và confidence. Không tính precision/recall/F1.

| File | Rows | Predicted label | Count | Rate | Mean confidence |
|---|---:|---|---:|---:|---:|
| 2026-08-27T00-48_export.csv | 2154 | UPWARD_SHIFT | 10 | 0.46% | 82.75% |
| 2026-08-27T00-48_export.csv | 2154 | DOWNWARD_SHIFT | 0 | 0.00% | — |
| 2026-08-27T00-48_export.csv | 2154 | GRADUAL_CHANGE | 313 | 14.53% | 93.01% |
| 2026-08-27T00-48_export.csv | 2154 | STABLE_JITTER | 1731 | 80.36% | 96.88% |
| 2026-08-27T00-48_export.csv | 2154 | OSCILLATION_NOISE | 100 | 4.64% | 79.07% |

| TEST DO DOC.csv | 2154 | UPWARD_SHIFT | 4 | 0.19% | 77.23% |
| TEST DO DOC.csv | 2154 | DOWNWARD_SHIFT | 0 | 0.00% | — |
| TEST DO DOC.csv | 2154 | GRADUAL_CHANGE | 164 | 7.61% | 94.72% |
| TEST DO DOC.csv | 2154 | STABLE_JITTER | 1912 | 88.77% | 97.07% |
| TEST DO DOC.csv | 2154 | OSCILLATION_NOISE | 74 | 3.44% | 84.63% |
