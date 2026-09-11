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
            "OperationalState",
            "CapacityMode",
            "ShadowFuel",
            "TrendEscapeTriggered",
            "InnovationScore",
            "TransitionProgress",
            "KalmanQ",
            "KalmanR",
        }
    ).issubset(result.columns)
    assert result.loc[result.index[2], "QualityFlag"] == "ZERO_DROPOUT_HELD"
    assert result.loc[result.index[2], "CleanFuel"] == result.loc[result.index[1], "CleanFuel"]
    assert "REFUEL" not in result["SignalState"].tolist()


def test_dashboard_capacity_estimate_stays_unknown_without_vehicle_calibration():
    frame = pd.DataFrame({"FuelLevel": [0.0, 20.0, 100.0, 110.0]})
    assert estimate_capacity_liters(frame) is None
    assert estimate_capacity_liters(frame, "29H75028") == 100.0


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


def _causal_frame():
    start = datetime(2026, 1, 1, 8, 0)
    values = [50, 50, 48, 46, 45, 46, 48, 50, 50]
    return pd.DataFrame({
        "VehicleID": ["CAUSAL-CAR"] * len(values),
        "FuelTime": [start + timedelta(minutes=2 * i) for i in range(len(values))],
        "FuelLevel": values,
        "Speed": [0, 0, 20, 30, 0, 0, 20, 30, 0],
        "SegmentID": [1] * len(values),
    })


def test_dashboard_prefix_invariance_proves_no_future_leakage():
    frame = _causal_frame()
    prefix = run_topic1_filter(frame.iloc[:6], "CAUSAL-CAR", 100.0, model_dir="")
    full = run_topic1_filter(frame, "CAUSAL-CAR", 100.0, model_dir="").iloc[:6]
    columns = ["CleanFuel", "OperationalState", "ExpectedFuelRate", "KalmanQ", "KalmanR"]
    pd.testing.assert_frame_equal(prefix[columns].reset_index(drop=True), full[columns].reset_index(drop=True))


def test_dashboard_causal_batch_equals_independent_sample_streaming():
    frame = _causal_frame()
    batch = run_topic1_filter(frame, "CAUSAL-CAR", 100.0, model_dir="")
    streaming = AISmoothTrackingFilter(model_dir="")
    rows = [streaming.process_point(
        "CAUSAL-CAR", row.FuelTime, row.FuelLevel, row.Speed, 100.0,
        segment_id=row.SegmentID,
    ) for row in frame.itertuples(index=False)]
    assert batch["CleanFuel"].tolist() == [row["CleanFuel"] for row in rows]
    assert batch["OperationalState"].tolist() == [row["OperationalState"] for row in rows]
