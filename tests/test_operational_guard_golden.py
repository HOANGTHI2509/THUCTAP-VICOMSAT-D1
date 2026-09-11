from datetime import datetime, timedelta
from pathlib import Path

import pytest

from src.core.filters.smooth_tracking import AISmoothTrackingFilter
from src.dashboard.dashboard_data import _predict_causal_states_batch, load_telemetry_csv


def _run(values, speeds=None, states=None, capacity=100.0):
    speeds = speeds or [0.0] * len(values)
    states = states or ["STABLE_JITTER"] * len(values)
    engine = AISmoothTrackingFilter(model_dir="")
    start = datetime(2026, 1, 1)
    return [
        engine.process_point(
            "GOLDEN", start + timedelta(minutes=2 * index), value,
            speeds[index], capacity, states[index], segment_id="A",
        )
        for index, value in enumerate(values)
    ]


def test_a_u_shape_is_held_and_cancelled_on_rebound():
    output = _run([50, 49, 48, 47, 46, 45, 45, 45, 46, 48, 50])
    assert min(item["clean_fuel"] for item in output) >= 49.5
    assert "REBOUND_RECOVERY" in [item["OperationalState"] for item in output]
    assert "DOWNWARD_CONFIRMED" not in [item["OperationalState"] for item in output]


def test_b_real_down_confirms_new_physical_baseline():
    output = _run([50, 48, 46, 45, 45, 45, 45])
    assert "PENDING_DOWNWARD" in [item["OperationalState"] for item in output]
    assert output[-1]["OperationalState"] == "BASELINE_REACQUISITION"
    assert 45.0 < output[-1]["clean_fuel"] < 50.0


def test_c_continued_down_confirms_in_three_candidate_beats():
    output = _run([50, 48, 46, 44, 42, 40])
    assert output[3]["OperationalState"] == "DOWNWARD_CONFIRMED"
    assert 44.0 < output[3]["clean_fuel"] < 50.0
    assert output[-1]["OperationalState"] == "DOWNWARD_CONFIRMED"
    assert output[-1]["clean_fuel"] == pytest.approx(40.0, abs=1.0)
    confirmed_clean = [item["clean_fuel"] for item in output[3:]]
    assert all(next_value < value for value, next_value in zip(confirmed_clean, confirmed_clean[1:]))


def test_d_overshoot_recovery_does_not_create_false_up():
    output = _run([50, 45, 45, 50, 52, 50])
    states = [item["OperationalState"] for item in output]
    assert "UPWARD_CONFIRMED" not in states
    assert max(item["clean_fuel"] for item in output) < 50.2


def test_e_real_up_confirms_and_tracks_new_baseline():
    output = _run([50, 50, 60, 60, 60])
    assert output[-1]["OperationalState"] == "UPWARD_CONFIRMED"
    assert 50.0 < output[-1]["clean_fuel"] < 60.0
    assert output[-1]["KalmanQ"] == 2.0
    assert output[-1]["KalmanR"] == 6.0


def test_f_gradual_moving_is_not_reclassified_as_pending_down():
    values = [50, 49.8, 49.6, 49.4, 49.2]
    output = _run(values, [40.0] * len(values), ["GRADUAL_CHANGE"] * len(values))
    assert [item["OperationalState"] for item in output[1:]] == ["GRADUAL_TRACKING"] * 4
    assert output[-1]["clean_fuel"] < output[1]["clean_fuel"]


def test_state_resets_on_segment_change_and_gap_over_30_minutes():
    engine = AISmoothTrackingFilter(model_dir="")
    start = datetime(2026, 1, 1)
    engine.process_point("CAR", start, 50, capacity_est=100, segment_id="A")
    engine.process_point("CAR", start + timedelta(minutes=2), 45, capacity_est=100, segment_id="A")
    segment_reset = engine.process_point("CAR", start + timedelta(minutes=4), 70, capacity_est=100, segment_id="B")
    gap_reset = engine.process_point("CAR", start + timedelta(minutes=35), 80, capacity_est=100, segment_id="B")
    assert segment_reset["ai_state"] == "INIT" and segment_reset["clean_fuel"] == 70
    assert gap_reset["ai_state"] == "INIT" and gap_reset["clean_fuel"] == 80


def test_capacity_registry_and_unknown_capacity_mode_are_explicit():
    engine = AISmoothTrackingFilter(model_dir="")
    known = engine.process_point("29H75028", datetime(2026, 1, 1), 50)
    unknown = engine.process_point("NO-CALIBRATION", datetime(2026, 1, 1), 50)
    assert known["CapacityMode"] == "KNOWN_CAPACITY"
    assert known["CapacityEstimate"] == 100.0
    assert unknown["CapacityMode"] == "UNKNOWN_CAPACITY_MODE"
    assert unknown["CapacityEstimate"] == 62.5


def test_innovation_gate_holds_one_sample_spike():
    output = _run([50, 50, 80, 50, 50])
    assert output[2]["InnovationGated"] is True
    assert output[2]["clean_fuel"] == pytest.approx(50.0, abs=0.05)
    assert max(item["clean_fuel"] for item in output) < 50.1


def test_g_real_29h75028_u_shape_reacquires_53_1_in_three_beats():
    source = Path("TienXuLy/29H75028_processed.csv")
    if not source.is_file():
        pytest.skip("Real telemetry fixture is not present")
    frame = load_telemetry_csv(source).iloc[:2213]
    predictor = AISmoothTrackingFilter(model_dir="models/fuel_state_classifier")
    states = _predict_causal_states_batch(frame, predictor, "29H75028:features", 100.0)
    probabilities = states.attrs["probabilities"]
    engine = AISmoothTrackingFilter(model_dir="")
    output = []
    for position, row in enumerate(frame.itertuples(index=False)):
        output.append(engine.process_point(
            "29H75028", row.FuelTime, row.FuelLevel, row.Speed, 100.0,
            states.iloc[position], getattr(row, "Lat", None), getattr(row, "Lng", None),
            row.SegmentID, probabilities.iloc[position],
        ))
    case = output[2192:2213]
    assert min(item["clean_fuel"] for item in case) > 53.0
    assert max(item["clean_fuel"] for item in case[10:14]) < 54.5
    assert case[16]["OperationalState"] == "BASELINE_REACQUISITION"
    assert case[18]["clean_fuel"] == pytest.approx(53.1, abs=0.1)
