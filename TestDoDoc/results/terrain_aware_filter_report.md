# Terrain-aware realtime filter test

Terrain rule: grade >= 3% while moving -> multiply Kalman R by 4. No direct litre offset.

## TEST DO DOC

- Rows marked terrain suspect: 19
- Mean absolute output change on those rows: 0.191 L
- Output: `TEST DO DOC_terrain_aware_clean.csv`

## 2026-08-27T00-48_export

- Rows marked terrain suspect: 12
- Mean absolute output change on those rows: 0.184 L
- Output: `2026-08-27T00-48_export_terrain_aware_clean.csv`
