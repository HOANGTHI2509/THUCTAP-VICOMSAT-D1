"""Create isolated RF Causal v3 + realtime Kalman evidence for Car 5."""

from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd

_tcn_stub = types.ModuleType("src.core.filters.tcn_state_classifier")
_tcn_stub.predict_tcn_fuel_state = lambda *args, **kwargs: None
sys.modules.setdefault("src.core.filters.tcn_state_classifier", _tcn_stub)

from src.core.filters.ai_enhanced_adaptive_realtime import filter_ai_enhanced_adaptive_realtime
from src.core.filters.ai_state_filter import load_fuel_state_classifier, predict_ai_fuel_state
from TestDoDoc.run_realtime_filter_test import summarize, write_svg_plot


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "data" / "processed" / "CarFuelHistory_Processed_Car5.csv"
MODEL_DIR = ROOT / "models" / "rf_signal_state_causal_v3"
OUTPUT = Path(__file__).resolve().parent / "results_car5_realtime_v3"


def load_car5() -> pd.DataFrame:
    frame = pd.read_csv(INPUT)
    frame["FuelTime"] = pd.to_datetime(frame["FuelTime"], errors="coerce")
    for column in ["FuelLevel", "Speed", "Lat", "Lng"]:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame.sort_values("FuelTime", kind="stable").reset_index(drop=True)


def find_largest_transition_window(frame: pd.DataFrame, radius: int = 180) -> pd.DataFrame:
    fuel = pd.to_numeric(frame["FuelLevel"], errors="coerce")
    prior = fuel.shift(1).rolling(12, min_periods=3).median()
    candidates = np.flatnonzero(((fuel <= 5.0) & (prior >= 50.0)).to_numpy())
    if len(candidates):
        center = int(candidates[0])
    else:
        center = int(fuel.diff().abs().fillna(0.0).idxmax())
    return frame.iloc[max(0, center - radius) : min(len(frame), center + radius + 1)].copy()


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    model, metadata = load_fuel_state_classifier(str(MODEL_DIR))
    if model is None or metadata is None:
        raise RuntimeError(f"Model not found: {MODEL_DIR}")

    frame = load_car5()
    enriched = predict_ai_fuel_state(frame, model, metadata, mode="realtime")
    enriched["CleanFuelRealtime"] = filter_ai_enhanced_adaptive_realtime(
        enriched, config={"source_col": "FuelLevel"}
    )
    full_csv = OUTPUT / "Car5_realtime_clean_v3.csv"
    enriched.to_csv(full_csv, index=False, encoding="utf-8-sig")

    focus = find_largest_transition_window(enriched)
    focus_csv = OUTPUT / "Car5_transition_focus_realtime_clean_v3.csv"
    focus_svg = OUTPUT / "Car5_transition_focus_realtime_chart_v3.svg"
    focus.to_csv(focus_csv, index=False, encoding="utf-8-sig")
    write_svg_plot(focus, focus_svg, "Car 5 — Realtime largest-transition stress test")

    report = OUTPUT / "Car5_realtime_report.md"
    report.write_text(
        "# Car 5 realtime stress test\n\n"
        "Model: RF Causal v3 + ai_enhanced_adaptive_realtime.py\n\n"
        f"Full trace metrics: {summarize(enriched)}\n\n"
        f"Focus window: {focus['FuelTime'].min()} → {focus['FuelTime'].max()}, rows={len(focus)}\n",
        encoding="utf-8",
    )
    print(f"Saved {full_csv}")
    print(f"Saved {focus_csv}")
    print(f"Saved {focus_svg}")
    print(f"Saved {report}")
    print(summarize(enriched))


if __name__ == "__main__":
    main()
