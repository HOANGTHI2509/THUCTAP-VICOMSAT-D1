"""Regression tests curated from real VICOMSAT telemetry windows.

The fixture is deliberately compact: tests do not process full historical CSV
files, but each selected window remains traceable to a source file and row
range.  Update an expected clean curve only after visual/business review.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from src.core.filters.smooth_tracking import AISmoothTrackingFilter


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIXTURE_PATH = PROJECT_ROOT / "tests" / "fixtures" / "golden_fuel_segments.json"
GOLDEN_SEGMENTS = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))


def _run_segment(segment: dict, model_dir: str = "") -> list[dict]:
    """Run a complete real-data window through one fresh causal context."""
    engine = AISmoothTrackingFilter(model_dir=model_dir)
    results = []
    latitudes = segment.get("latitudes", [None] * len(segment["timestamps"]))
    longitudes = segment.get("longitudes", [None] * len(segment["timestamps"]))
    for timestamp, raw_fuel, speed, latitude, longitude in zip(
        segment["timestamps"],
        segment["raw_liters"],
        segment["speed_kmh"],
        latitudes,
        longitudes,
    ):
        results.append(
            engine.process_point(
                vehicle_id=segment["id"],
                timestamp=timestamp,
                raw_fuel=raw_fuel,
                speed=speed,
                capacity_est=segment["capacity_est_liters"],
                lat=latitude,
                lng=longitude,
            )
        )
    return results


@pytest.mark.parametrize("segment", GOLDEN_SEGMENTS, ids=lambda item: item["id"])
def test_golden_source_window_still_matches_real_telemetry(segment: dict):
    """Keep the compact fixture auditable against the original processed CSV."""
    assert segment["review_status"] in {
        "initial_baseline_pending_domain_review",
        "pending_domain_review",
        "approved",
    }
    source = PROJECT_ROOT / segment["source"]
    if not source.is_file():
        pytest.skip("Raw telemetry source is intentionally excluded from this checkout")
    start_row, end_row = segment["source_rows"]
    source_window = pd.read_csv(source).iloc[start_row : end_row + 1]

    assert source_window["FuelTime"].astype(str).tolist() == segment["timestamps"]
    assert source_window["FuelLevel"].round(2).tolist() == segment["raw_liters"]
    assert source_window["Speed"].fillna(0.0).round(2).tolist() == segment["speed_kmh"]
    if "latitudes" in segment:
        assert source_window["Lat"].round(6).tolist() == segment["latitudes"]
        assert source_window["Lng"].round(6).tolist() == segment["longitudes"]


@pytest.mark.parametrize("segment", GOLDEN_SEGMENTS, ids=lambda item: item["id"])
def test_golden_clean_curve_regression(segment: dict):
    """Lock the currently reviewed clean curve for each real-data segment."""
    actual_clean = [item["clean_fuel"] for item in _run_segment(segment)]
    assert actual_clean == pytest.approx(segment["expected_clean_liters"], abs=0.01)


@pytest.mark.parametrize("segment", GOLDEN_SEGMENTS, ids=lambda item: item["id"])
def test_golden_production_model_regression(segment: dict):
    """Lock the deployed classifier plus purple filter as one production path."""
    output = _run_segment(segment, model_dir="models/fuel_state_classifier")
    assert [item["clean_fuel"] for item in output] == pytest.approx(
        segment["expected_production_clean_liters"],
        abs=0.01,
    )
    assert [item["ai_state"] for item in output] == segment["expected_production_states"]


@pytest.mark.parametrize("segment", GOLDEN_SEGMENTS, ids=lambda item: item["id"])
def test_golden_signal_behavior_contract(segment: dict):
    """Give failures a business-readable reason in addition to exact curves."""
    clean = [item["clean_fuel"] for item in _run_segment(segment)]
    output = _run_segment(segment)
    raw = segment["raw_liters"]
    checks = segment["checks"]

    if "max_clean_span" in checks:
        assert max(clean) - min(clean) <= checks["max_clean_span"]
    if "min_clean_liters" in checks:
        assert min(clean) >= checks["min_clean_liters"]
    if "min_valley_attenuation_liters" in checks:
        attenuation = min(clean) - min(raw)
        assert attenuation >= checks["min_valley_attenuation_liters"]
    if "min_total_clean_drop" in checks:
        assert clean[0] - clean[-1] >= checks["min_total_clean_drop"]
    if "max_upward_step" in checks:
        assert max(next_value - value for value, next_value in zip(clean, clean[1:])) <= checks["max_upward_step"]
    if "max_final_lag_liters" in checks:
        assert clean[-1] - raw[-1] <= checks["max_final_lag_liters"]
    if "max_step_size" in checks:
        assert max(abs(next_value - value) for value, next_value in zip(clean, clean[1:])) <= checks["max_step_size"]
    if "zero_dropout_indexes" in checks:
        for index in checks["zero_dropout_indexes"]:
            assert output[index]["quality_flag"] == "ZERO_DROPOUT_HELD"
            assert output[index]["clean_fuel"] == output[index - 1]["clean_fuel"]
    if "required_motion_states" in checks:
        assert set(checks["required_motion_states"]).issubset(
            {item["motion_state"] for item in output}
        )
