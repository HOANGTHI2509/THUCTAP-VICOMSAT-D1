from datetime import datetime, timedelta

import pandas as pd

from src.dashboard.dashboard_data import estimate_capacity_liters, run_topic1_filter


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


def test_dashboard_capacity_estimate_is_safe_when_calibration_is_unknown():
    frame = pd.DataFrame({"FuelLevel": [0.0, 20.0, 100.0, 110.0]})
    assert estimate_capacity_liters(frame) >= 200.0
