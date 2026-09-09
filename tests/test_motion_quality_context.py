from datetime import datetime, timedelta

import pandas as pd

from src.core.filters.smooth_tracking import AISmoothTrackingFilter, filter_smooth_tracking_dataframe
from src.service.state_manager import StreamingStateManager


START = datetime(2026, 1, 1, 8, 0)


def _run(speeds, coordinates, values=None):
    engine = AISmoothTrackingFilter(model_dir="")
    values = values or [500.0 - index for index in range(len(speeds))]
    results = []
    for index, (speed, coordinate, value) in enumerate(zip(speeds, coordinates, values)):
        kwargs = {}
        if coordinate is not None:
            kwargs = {"lat": coordinate[0], "lng": coordinate[1]}
        results.append(
            engine.process_point(
                vehicle_id="MOTION_TEST",
                timestamp=START + timedelta(minutes=2 * index),
                raw_fuel=value,
                speed=speed,
                capacity_est=600.0,
                **kwargs,
            )
        )
    return results


def test_low_motion_requires_low_speed_and_clustered_gps():
    coordinates = [
        (21.00000, 105.00000),
        (21.00001, 105.00001),
        (21.00002, 105.00001),
        (21.00001, 105.00002),
    ]
    results = _run([0.0] * len(coordinates), coordinates)

    assert [item["motion_state"] for item in results] == [
        "UNCERTAIN",
        "UNCERTAIN",
        "LOW_MOTION",
        "LOW_MOTION",
    ]
    assert results[-1]["gps_displacement_meters"] < 5.0
    assert results[-1]["motion_confidence"] >= 0.95


def test_moving_requires_high_speed_and_meaningful_gps_displacement():
    coordinates = [
        (21.0000, 105.0000),
        (21.0004, 105.0000),
        (21.0008, 105.0000),
        (21.0012, 105.0000),
    ]
    results = _run([35.0] * len(coordinates), coordinates)

    assert results[-1]["motion_state"] == "MOVING"
    assert results[-1]["gps_displacement_meters"] > 100.0


def test_speed_and_gps_conflict_remains_uncertain():
    static_coordinates = [(21.0, 105.0)] * 4
    moving_coordinates = [
        (21.0000, 105.0000),
        (21.0004, 105.0000),
        (21.0008, 105.0000),
        (21.0012, 105.0000),
    ]

    speed_only = _run([35.0] * 4, static_coordinates)
    gps_only = _run([0.0] * 4, moving_coordinates)

    assert speed_only[-1]["motion_state"] == "UNCERTAIN"
    assert gps_only[-1]["motion_state"] == "UNCERTAIN"


def test_low_motion_uses_a_lighter_tracking_baseline_than_legacy_speed_only():
    clustered_coordinates = [
        (21.00000, 105.00000),
        (21.00001, 105.00001),
        (21.00002, 105.00001),
        (21.00001, 105.00002),
        (21.00002, 105.00002),
    ]
    low_motion = _run([0.0] * 5, clustered_coordinates)
    no_gps = _run([0.0] * 5, [None] * 5)

    assert low_motion[-1]["motion_state"] == "LOW_MOTION"
    assert no_gps[-1]["motion_state"] == "UNCERTAIN"
    assert low_motion[-1]["clean_fuel"] < no_gps[-1]["clean_fuel"]


def test_state_manager_and_dataframe_expose_motion_diagnostics():
    manager = StreamingStateManager(model_dir="")
    coordinates = [
        (21.0000, 105.0000),
        (21.0004, 105.0000),
        (21.0008, 105.0000),
    ]
    outputs = [
        manager.process_point(
            vehicle_id="MANAGER_GPS",
            fuel_time=START + timedelta(minutes=2 * index),
            fuel_level=400.0 - index,
            speed=30.0,
            lat=coordinate[0],
            lng=coordinate[1],
            capacity_est=600.0,
        )
        for index, coordinate in enumerate(coordinates)
    ]
    assert outputs[-1]["motion_state"] == "MOVING"

    frame = pd.DataFrame(
        {
            "VehicleID": ["DATAFRAME_GPS"] * 3,
            "FuelTime": [START + timedelta(minutes=2 * index) for index in range(3)],
            "FuelLevel": [400.0, 399.0, 398.0],
            "Speed": [30.0, 30.0, 30.0],
            "Lat": [coordinate[0] for coordinate in coordinates],
            "Lng": [coordinate[1] for coordinate in coordinates],
        }
    )
    result = filter_smooth_tracking_dataframe(
        frame,
        capacity_est=600.0,
        filter_engine=AISmoothTrackingFilter(model_dir=""),
    )
    assert result["MotionState_SmoothTracking"].tolist()[-1] == "MOVING"
    assert result["GpsDisplacementMeters_SmoothTracking"].iloc[-1] > 50.0
