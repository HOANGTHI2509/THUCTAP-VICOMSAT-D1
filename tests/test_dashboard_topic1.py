from datetime import datetime, timedelta

import pandas as pd
import numpy as np

from src.core.filters.smooth_tracking import AISmoothTrackingFilter
from src.dashboard.dashboard_data import (
    _predict_causal_states_batch,
    estimate_capacity_liters,
    run_topic1_filter,
)


def test_dashboard_uses_purple_topic1_output_and_canonical_diagnostics():
    start = datetime(2026, 1, 1, 8, 0)
    source = pd.DataFrame(
        {
            "VehicleID": ["DASHBOARD-CAR"] * 4,
            "FuelTime": [start + timedelta(minutes=2 * index) for index in range(4)],
            "FuelLevel": [300.0, 299.0, 0.0, 298.0],
            "Speed": [20.0, 20.0, 0.0, 20.0],
            "SegmentID": [1, 1, 1, 1],
            # Historical labels must not dictate the current dashboard result.
            "AI_State": ["REFUEL", "REFUEL", "REFUEL", "REFUEL"],
        }
    )

    result = run_topic1_filter(
        source,
        vehicle_id="DASHBOARD-CAR",
        capacity_est_liters=500.0,
        model_dir="",
    )

    assert set(
        {
            "CleanFuel",
            "SignalState",
            "QualityFlag",
            "MotionState",
            "RollingStd",
        }
    ).issubset(result.columns)
    assert result.loc[result.index[2], "QualityFlag"] == "ZERO_DROPOUT_HELD"
    assert result.loc[result.index[2], "CleanFuel"] == result.loc[result.index[1], "CleanFuel"]
    assert "REFUEL" not in result["SignalState"].tolist()


def test_dashboard_capacity_estimate_stays_unknown_without_calibration():
    frame = pd.DataFrame({"FuelLevel": [0.0, 20.0, 100.0, 110.0]})
    assert estimate_capacity_liters(frame) is None
    assert estimate_capacity_liters(frame, vehicle_id="92H-03625") is None
    assert estimate_capacity_liters(frame, vehicle_id="Car 3") is None


def test_dashboard_vehicle_is_not_clamped_by_removed_capacity_mapping():
    start = datetime(2026, 1, 1, 8, 0)
    frame = pd.DataFrame(
        {
            "FuelTime": [start, start + timedelta(minutes=2)],
            "FuelLevel": [520.0, 519.5],
            "Speed": [0.0, 0.0],
            "SegmentID": [1, 1],
        }
    )

    result = run_topic1_filter(
        frame,
        vehicle_id="92H-03625",
        capacity_est_liters=estimate_capacity_liters(frame, "92H-03625"),
        model_dir="",
    )

    assert result["CapacityMode"].eq("UNKNOWN").all()
    assert result["CapacitySource"].eq("NONE").all()
    assert result["CapacityEstimate"].isna().all()
    assert result["CleanFuel"].min() > 210.0


def test_dashboard_batches_classifier_without_changing_causal_states():
    class ThresholdModel:
        def predict(self, matrix):
            values = np.asarray(matrix)[:, 0]
            return np.where(values < 299.0, "GRADUAL_CHANGE", "STABLE_JITTER")

    start = datetime(2026, 1, 1, 8, 0)
    frame = pd.DataFrame(
        {
            "FuelTime": [start + timedelta(minutes=2 * index) for index in range(4)],
            "FuelLevel": [300.0, 299.5, 298.5, 298.0],
            "Speed": [20.0] * 4,
        }
    )
    engine = AISmoothTrackingFilter(model_dir="")
    engine.model = ThresholdModel()
    engine.feature_columns = ["FuelLevel"]

    states = _predict_causal_states_batch(frame, engine, "BATCH-CAR", 400.0)

    assert states.tolist() == [
        "INIT",
        "STABLE_JITTER",
        "GRADUAL_CHANGE",
        "GRADUAL_CHANGE",
    ]
