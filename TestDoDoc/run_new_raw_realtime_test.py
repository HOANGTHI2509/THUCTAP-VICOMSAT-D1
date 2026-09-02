"""Isolated realtime test for the two root-level raw traces."""

from __future__ import annotations

import sys
import types
from pathlib import Path

# The isolated RF test does not need optional PyTorch/TCN code.
_tcn_stub = types.ModuleType("src.core.filters.tcn_state_classifier")
_tcn_stub.predict_tcn_fuel_state = lambda *args, **kwargs: None
sys.modules.setdefault("src.core.filters.tcn_state_classifier", _tcn_stub)

from src.core.filters.ai_enhanced_adaptive_realtime import filter_ai_enhanced_adaptive_realtime
from src.core.filters.ai_state_filter import load_fuel_state_classifier, predict_ai_fuel_state
from TestDoDoc.run_realtime_filter_test import load_trace, summarize, write_svg_plot


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = Path(__file__).resolve().parent / "results_new_raw_v3"
MODEL_DIR = ROOT / "models" / "rf_signal_state_causal_v3"
INPUTS = [ROOT / "2026-08-27T00-48_export.csv", ROOT / "TEST DO DOC.csv"]


def main() -> None:
    model, metadata = load_fuel_state_classifier(str(MODEL_DIR))
    if model is None or metadata is None:
        raise RuntimeError(f"Model not found: {MODEL_DIR}")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for input_path in INPUTS:
        frame = load_trace(input_path)
        enriched = predict_ai_fuel_state(frame, model, metadata, mode="realtime")
        enriched["CleanFuelRealtime"] = filter_ai_enhanced_adaptive_realtime(
            enriched, config={"source_col": "FuelLevel"}
        )
        csv_path = OUTPUT / f"{input_path.stem}_realtime_clean_v3.csv"
        chart_path = OUTPUT / f"{input_path.stem}_realtime_chart_v3.svg"
        enriched.to_csv(csv_path, index=False, encoding="utf-8-sig")
        write_svg_plot(enriched, chart_path, f"Realtime fuel cleaning V3 - {input_path.name}")
        print(f"Saved {csv_path}")
        print(f"Saved {chart_path}")
        print(summarize(enriched))


if __name__ == "__main__":
    main()
