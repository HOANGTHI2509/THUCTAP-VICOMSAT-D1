"""Run the causal fuel-cleaning pipeline independently for TestDoDoc CSVs.

The script intentionally does not start Streamlit or FastAPI.  It creates a
reproducible CSV and compact metric report for each input file so the two
provided traces can be inspected in isolation.
"""

from __future__ import annotations

from pathlib import Path
import sys
import types
from html import escape

import numpy as np
import pandas as pd

from src.core.filters.ai_enhanced_adaptive_realtime import (
    filter_ai_enhanced_adaptive_realtime,
)

# The production dashboard disables the TCN model below its acceptance target.
# Stub its optional dependency so this isolated RF test does not require torch.
_tcn_stub = types.ModuleType("src.core.filters.tcn_state_classifier")
_tcn_stub.predict_tcn_fuel_state = lambda *args, **kwargs: None
sys.modules.setdefault("src.core.filters.tcn_state_classifier", _tcn_stub)
from src.core.filters.ai_state_filter import load_fuel_state_classifier, predict_ai_fuel_state


BASE_DIR = Path(__file__).resolve().parent
INPUT_FILES = [
    BASE_DIR / "TEST DO DOC.csv",
    BASE_DIR / "2026-08-27T00-48_export.csv",
]
OUTPUT_DIR = BASE_DIR / "results"
MODEL_DIR = Path("models") / "rf_signal_state_causal_v3"


def load_trace(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.loc[:, ~df.columns.str.startswith("Unnamed")].copy()
    required = {"FuelTime", "FuelLevel", "Speed"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path.name} missing columns: {sorted(missing)}")

    for column in ["FuelLevel", "Speed", "Lat", "Lng"]:
        if column in df.columns:
            df[column] = pd.to_numeric(
                df[column].astype(str).str.replace(",", ".", regex=False),
                errors="coerce",
            )
    df["FuelTime"] = pd.to_datetime(df["FuelTime"], errors="coerce")
    df = df.dropna(subset=["FuelTime", "FuelLevel"]).sort_values("FuelTime", kind="stable").reset_index(drop=True)
    df["VehicleID"] = path.stem
    df["TimeGapMinutes"] = df["FuelTime"].diff().dt.total_seconds().div(60.0).fillna(2.0)
    df["SegmentID"] = (df["TimeGapMinutes"] >= 120.0).cumsum()
    df["DeltaFuel"] = df.groupby("SegmentID")["FuelLevel"].diff().fillna(0.0)
    df["RollingStd"] = df.groupby("SegmentID")["FuelLevel"].transform(
        lambda value: value.rolling(12, min_periods=1).std().fillna(0.0)
    )
    df["QualityFlag"] = (df["FuelLevel"] <= 0).astype(int)
    df["QualityReason"] = np.where(df["FuelLevel"] <= 0, "FUEL_ZERO", "VALID")
    return df


def summarize(df: pd.DataFrame) -> dict[str, float | int]:
    raw = df["FuelLevel"]
    clean = df["CleanFuelRealtime"]
    low_raw = raw <= 5.0
    clean_at_low_raw = clean[low_raw]
    return {
        "rows": len(df),
        "raw_zero_or_low_rows": int(low_raw.sum()),
        "clean_zero_or_low_at_raw_low": int((clean_at_low_raw <= 5.0).sum()),
        "clean_missing_at_raw_low": int(clean_at_low_raw.isna().sum()),
        "raw_mean_abs_step": round(float(raw.diff().abs().mean()), 4),
        "clean_mean_abs_step": round(float(clean.diff().abs().mean()), 4),
        "raw_min": round(float(raw.min()), 3),
        "clean_min": round(float(clean.min()), 3),
    }


def write_svg_plot(df: pd.DataFrame, output_path: Path, title: str) -> None:
    """Create a dependency-free chart suitable for quick review/slides."""
    width, height = 1500, 640
    left, right, top, bottom = 90, 35, 65, 75
    plot_width, plot_height = width - left - right, height - top - bottom
    raw = pd.to_numeric(df["FuelLevel"], errors="coerce").to_numpy(dtype=float)
    clean = pd.to_numeric(df["CleanFuelRealtime"], errors="coerce").to_numpy(dtype=float)
    values = np.concatenate([raw[np.isfinite(raw)], clean[np.isfinite(clean)]])
    y_min, y_max = float(values.min()), float(values.max())
    padding = max((y_max - y_min) * 0.08, 1.0)
    y_min, y_max = y_min - padding, y_max + padding

    def points(values: np.ndarray) -> str:
        result = []
        denominator = max(len(values) - 1, 1)
        for index, value in enumerate(values):
            if not np.isfinite(value):
                continue
            x = left + index / denominator * plot_width
            y = top + (y_max - value) / (y_max - y_min) * plot_height
            result.append(f"{x:.2f},{y:.2f}")
        return " ".join(result)

    grid = []
    for step in range(6):
        value = y_min + (y_max - y_min) * step / 5
        y = top + plot_height - step / 5 * plot_height
        grid.append(f'<line x1="{left}" y1="{y:.2f}" x2="{width-right}" y2="{y:.2f}" class="grid"/>')
        grid.append(f'<text x="{left-12}" y="{y+5:.2f}" class="axis" text-anchor="end">{value:.1f}</text>')

    start = df["FuelTime"].iloc[0]
    end = df["FuelTime"].iloc[-1]
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<style>
  .grid {{ stroke:#dbe4f0; stroke-width:1; }} .axis {{ fill:#53627a; font:16px Arial; }}
  .title {{ fill:#17243b; font:700 25px Arial; }} .legend {{ fill:#24334d; font:16px Arial; }}
</style>
<rect width="100%" height="100%" fill="white"/>
<text x="{left}" y="34" class="title">{escape(title)}</text>
<text x="{left}" y="{height-25}" class="axis">{escape(str(start))}  →  {escape(str(end))}</text>
{''.join(grid)}
<text x="28" y="{top + plot_height / 2:.2f}" class="axis" transform="rotate(-90 28,{top + plot_height / 2:.2f})">Fuel (L)</text>
<polyline points="{points(raw)}" fill="none" stroke="#e53935" stroke-width="1.5" opacity="0.8"/>
<polyline points="{points(clean)}" fill="none" stroke="#ff7f0e" stroke-width="3"/>
<line x1="{width-360}" y1="34" x2="{width-315}" y2="34" stroke="#e53935" stroke-width="2"/>
<text x="{width-305}" y="40" class="legend">Raw FuelLevel</text>
<line x1="{width-170}" y1="34" x2="{width-125}" y2="34" stroke="#ff7f0e" stroke-width="3"/>
<text x="{width-115}" y="40" class="legend">Clean realtime</text>
</svg>'''
    output_path.write_text(svg, encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    model, metadata = load_fuel_state_classifier(str(MODEL_DIR))
    if model is None or metadata is None:
        raise RuntimeError(f"Random Forest model/metadata not found in {MODEL_DIR}")

    report_lines = ["# Realtime Filter Test Report", "", "Model: rf_signal_state_causal_v3", "", "Mode: causal RF features + realtime adaptive Kalman", ""]
    for input_path in INPUT_FILES:
        df = load_trace(input_path)
        enriched = predict_ai_fuel_state(df, model, metadata, mode="realtime")
        enriched["CleanFuelRealtime"] = filter_ai_enhanced_adaptive_realtime(enriched)
        # Keep V1/V2 comparison outputs intact and avoid replacing a CSV that
        # may currently be open in Excel.
        output_path = OUTPUT_DIR / f"{input_path.stem}_realtime_clean_v3.csv"
        enriched.to_csv(output_path, index=False, encoding="utf-8-sig")
        plot_path = OUTPUT_DIR / f"{input_path.stem}_realtime_chart_v3.svg"
        write_svg_plot(enriched, plot_path, f"Realtime fuel cleaning — {input_path.name}")
        metrics = summarize(enriched)
        report_lines.extend([f"## {input_path.name}", "", "| Metric | Value |", "|---|---:|"])
        report_lines.extend(f"| {key} | {value} |" for key, value in metrics.items())
        report_lines.append("")
        print(f"Saved {output_path}")
        print(f"Saved {plot_path}")
        print(metrics)

    report_path = OUTPUT_DIR / "realtime_filter_report_v3.md"
    report_path.write_text("\n".join(report_lines), encoding="utf-8")
    print(f"Saved {report_path}")


if __name__ == "__main__":
    main()
