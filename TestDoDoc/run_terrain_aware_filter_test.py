"""Run the realtime filter with DEM road-grade used as a measurement-confidence flag."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.core.filters.ai_enhanced_adaptive_realtime import filter_ai_enhanced_adaptive_realtime


RESULTS = Path(__file__).resolve().parent / "results"
CASES = ["TEST DO DOC", "2026-08-27T00-48_export"]


def main() -> None:
    report = ["# Terrain-aware realtime filter test", "", "Terrain rule: grade >= 3% while moving -> multiply Kalman R by 4. No direct litre offset.", ""]
    for case in CASES:
        baseline = pd.read_csv(RESULTS / f"{case}_realtime_clean.csv")
        terrain = pd.read_csv(RESULTS / f"{case}_terrain.csv")
        if len(baseline) != len(terrain):
            raise ValueError(f"{case}: realtime/terrain row count mismatch")
        baseline["RoadGradePct"] = terrain["RoadGradePct"]
        baseline["SlopeSuspect"] = terrain["SlopeSuspect"]
        baseline["ElevationMSmoothed"] = terrain["ElevationMSmoothed"]
        baseline["CleanFuelTerrainAware"] = filter_ai_enhanced_adaptive_realtime(baseline)
        baseline["TerrainAdjustmentL"] = baseline["CleanFuelTerrainAware"] - baseline["CleanFuelRealtime"]
        output = RESULTS / f"{case}_terrain_aware_clean.csv"
        baseline.to_csv(output, index=False, encoding="utf-8-sig")
        suspect = baseline["SlopeSuspect"].fillna(0).astype(bool)
        mean_adjustment = baseline.loc[suspect, "TerrainAdjustmentL"].abs().mean()
        report.extend([
            f"## {case}", "",
            f"- Rows marked terrain suspect: {int(suspect.sum())}",
            f"- Mean absolute output change on those rows: {mean_adjustment:.3f} L",
            f"- Output: `{output.name}`", "",
        ])
        print(f"Saved {output}")
    (RESULTS / "terrain_aware_filter_report.md").write_text("\n".join(report), encoding="utf-8")


if __name__ == "__main__":
    main()
