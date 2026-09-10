# Smooth-Tracking KPI report

- Generated: 2026-09-10T15:09:10.834931+00:00
- Model: random_forest (814a70e28918)
- Config: sha256:1d41721c8472
- Golden fixture: sha256:9ee8b7da4daf
- Golden segments: 8
- Pending domain review: 8
- Behavior checks: 18/19
- Latency P50/P95/P99: 33.5937/37.83158/45.762072 ms
- Throughput: 31.785 points/s

`Noise reduction` is calculated from detrended standard deviation so a
real linear consumption trend is not counted as noise. `Regression MAE`
compares this run with the locked expected production curve.

| Segment | Review | Noise reduction | Regression MAE | Checks |
|---|---|---:|---:|---:|
| 21H-02058_stationary_noise_pulse | initial_baseline_pending_domain_review | 95.989% | 0.0 L | 1/2 |
| 21H-03221_short_valley_recovery | initial_baseline_pending_domain_review | 83.869% | 0.0 L | 3/3 |
| 92H-02687_steady_moving_consumption | initial_baseline_pending_domain_review | -7.37% | 0.0 L | 3/3 |
| 29E-44284_noisy_moving_consumption | initial_baseline_pending_domain_review | 39.991% | 0.0 L | 3/3 |
| 15H-08128_zero_dropout_hold | pending_domain_review | 0.0% | 0.0 L | 2/2 |
| 21H-03221_stationary_gps_cluster | pending_domain_review | 0.0% | 0.0 L | 2/2 |
| 21H-03221_u_shape_with_gps | pending_domain_review | 82.521% | 0.0 L | 2/2 |
| 21H-03221_stationary_gps_jump | pending_domain_review | 98.989% | 0.0 L | 2/2 |

## Failed behavior checks

- `21H-02058_stationary_noise_pulse`: `max_clean_span`
