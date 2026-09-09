# Golden fuel segments

`golden_fuel_segments.json` contains short windows copied from processed
telemetry selected as representative. The initial four cases have status
`initial_baseline_pending_domain_review`: their source and regression curve are
locked now, but they should be promoted only after your visual/business review.
Every case records:

- source CSV and inclusive source-row range for traceability;
- raw fuel already expressed in litres, speed, and timestamps;
- the `capacity_est_liters` supplied to the service path;
- expected clean curves for both the no-model fallback and the deployed model;
- plain-language behaviour limits such as maximum noise span or minimum tracked
  consumption.

When adding or approving a case, first inspect the source range in the dashboard
and agree the expected behaviour. Do not replace an expected clean curve merely
to make a test pass: changing it is an algorithm/product decision and should be
reviewed.

To scan potential source windows without changing data:

```powershell
.\.venv\Scripts\python.exe scripts\find_golden_candidates.py TienXuLy\21H-02058_processed.csv --size 12
```
