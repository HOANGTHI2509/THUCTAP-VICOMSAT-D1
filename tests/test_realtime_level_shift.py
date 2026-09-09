import pandas as pd

from src.core.filters.ai_enhanced_adaptive_realtime import (
    RealtimeAdaptiveKalmanState,
    filter_ai_enhanced_adaptive_realtime,
)


def _frame(values, segments=None):
    times = pd.date_range("2026-01-01 00:00:00", periods=len(values), freq="2min")
    return pd.DataFrame({
        "FuelTime": times,
        "FuelLevel": values,
        "AI_State": ["OSCILLATION_NOISE"] * len(values),
        "flat_jitter_threshold": [0.8] * len(values),
        "noise_sigma_liters": [0.8] * len(values),
        "event_threshold": [5.0] * len(values),
        "RollingStd": [1.0] * len(values),
        "Speed": [0.0] * len(values),
        "capacity_est": [500.0] * len(values),
        "SegmentID": segments or ["A"] * len(values),
    })


def test_short_u_is_not_committed():
    out = filter_ai_enhanced_adaptive_realtime(_frame([200.0, 150.0, 200.0]))
    assert out[0] == 200.0
    assert out[-1] >= 199.5


def test_stable_lower_level_tracks_on_third_low_sample():
    out = filter_ai_enhanced_adaptive_realtime(_frame([200.0, 150.0, 150.0, 150.0]))
    assert out[1] > 195.0
    assert out[2] > 195.0
    assert out[3] == 150.0


def test_small_upward_cluster_is_smoothed_not_committed_as_a_step():
    out = filter_ai_enhanced_adaptive_realtime(_frame([170.0, 173.0, 174.0, 173.0]))
    assert out[-1] > 170.0
    assert out[-1] < 171.0


def test_new_segment_initializes_once_not_on_every_following_point():
    state = RealtimeAdaptiveKalmanState()
    first = _frame([100.0], ["A"])
    second = _frame([200.0], ["B"])
    third = _frame([201.0], ["B"])
    _, state = filter_ai_enhanced_adaptive_realtime(first, state=state, return_state=True)
    _, state = filter_ai_enhanced_adaptive_realtime(second, state=state, return_state=True)
    out, state = filter_ai_enhanced_adaptive_realtime(third, state=state, return_state=True)
    assert state.segment_id == "B"
    assert out[0] < 200.5


def test_large_multistep_drop_tracks_at_third_directional_sample():
    out = filter_ai_enhanced_adaptive_realtime(
        _frame([500.0, 460.0, 370.0, 290.0, 282.0])
    )
    assert out[1] > 450.0
    assert out[2] > 350.0
    assert out[3] == 290.0


def test_medium_persistent_increase_on_large_tank_is_confirmed():
    frame = _frame([500.0, 650.0, 750.0, 815.0, 827.8, 827.9, 827.9])
    frame["capacity_est"] = 830.0
    frame["event_threshold"] = 29.0
    frame["noise_sigma_liters"] = 1.65
    frame["flat_jitter_threshold"] = 2.5
    out = filter_ai_enhanced_adaptive_realtime(frame)
    assert out[3] == 815.0
    assert out[4] < 820.0
    assert out[5] < 820.0
    assert out[6] == 827.9


def test_standalone_medium_hill_is_smoothed_not_committed():
    frame = _frame([279.0, 273.7, 284.6, 293.2, 290.2, 290.0, 282.1])
    frame["capacity_est"] = 537.0
    frame["event_threshold"] = 18.8
    frame["noise_sigma_liters"] = 1.1
    frame["flat_jitter_threshold"] = 1.6
    out = filter_ai_enhanced_adaptive_realtime(frame)
    assert max(out) < 285.0


def test_large_upward_hill_that_has_started_reversing_is_not_committed():
    frame = _frame([775.4, 824.0, 849.9, 847.5, 794.0, 789.6, 784.0, 778.0])
    frame["capacity_est"] = 813.1
    frame["event_threshold"] = 28.46
    frame["noise_sigma_liters"] = 1.63
    frame["flat_jitter_threshold"] = 2.44
    frame["RollingStd"] = [0.0, 17.9, 30.5, 32.9, 30.9, 28.6, 32.8, 28.0]
    frame["Speed"] = [0.0, 0.0, 0.0, 0.0, 7.0, 0.0, 0.0, 0.0]

    out = filter_ai_enhanced_adaptive_realtime(frame)

    assert max(out[1:4]) < 780.0
    assert out[-1] < 780.0


def test_moving_u_returning_on_fourth_sample_is_aborted():
    frame = _frame([302.0, 278.0, 278.0, 278.0, 302.0])
    frame["Speed"] = [0.0, 25.0, 20.0, 10.0, 5.0]
    out = filter_ai_enhanced_adaptive_realtime(frame)
    assert out == [302.0, 302.0, 302.0, 302.0, 302.0]


def test_moving_low_plateau_is_committed_on_fourth_low_sample():
    frame = _frame([302.0, 278.0, 278.0, 278.0, 278.0])
    frame["Speed"] = [0.0, 25.0, 20.0, 10.0, 5.0]
    out = filter_ai_enhanced_adaptive_realtime(frame)
    assert out[:4] == [302.0, 302.0, 302.0, 302.0]
    assert out[4] == 278.0


def test_micro_up_hill_returning_to_anchor_is_fully_held():
    frame = _frame([173.0, 176.0, 175.5, 174.0, 173.0])
    frame["Speed"] = [0.0, 20.0, 15.0, 10.0, 5.0]
    out = filter_ai_enhanced_adaptive_realtime(frame)
    assert max(abs(value - 173.0) for value in out) < 0.05


def test_micro_up_branch_does_not_delay_large_upward_shift():
    frame = _frame([173.0, 176.0, 220.0, 260.0, 300.0])
    out = filter_ai_enhanced_adaptive_realtime(frame)
    assert out[-1] == 300.0


def test_small_continuous_consumption_while_moving_does_not_form_stairs():
    values = [247.0 - 0.1 * i for i in range(60)]
    frame = _frame(values)
    frame["Speed"] = 48.0
    frame["AI_State"] = "GRADUAL_CHANGE"
    frame["RollingStd"] = 0.5
    out = filter_ai_enhanced_adaptive_realtime(frame)
    steps = [out[i] - out[i - 1] for i in range(1, len(out))]

    assert out[-1] < out[0] - 3.0
    assert max(abs(step) for step in steps) < 1.0
    assert sum(step < -0.01 for step in steps) > 40


def test_low_noise_stable_state_tracks_measurement_without_direct_snap():
    frame = _frame([525.8])
    frame["FuelTime"] = [pd.Timestamp("2026-01-01 00:02:00")]
    frame["capacity_est"] = 587.2
    frame["flat_jitter_threshold"] = 1.76
    frame["noise_sigma_liters"] = 1.17
    frame["event_threshold"] = 20.55
    frame["RollingStd"] = 0.83
    frame["AI_State"] = "STABLE_JITTER"

    state = RealtimeAdaptiveKalmanState(
        x=533.0,
        P=10.0,
        last_valid_x=533.0,
        previous_raw=526.0,
        previous_raw_2=526.0,
        segment_id="A",
        fuel_time="2026-01-01T00:00:00",
    )
    out = filter_ai_enhanced_adaptive_realtime(frame, state=state)

    assert 525.8 < out[0] < 532.0


def test_high_noise_stable_label_keeps_noise_scaled_smoothing():
    frame = _frame([525.8])
    frame["FuelTime"] = [pd.Timestamp("2026-01-01 00:02:00")]
    frame["capacity_est"] = 587.2
    frame["flat_jitter_threshold"] = 1.76
    frame["noise_sigma_liters"] = 1.17
    frame["event_threshold"] = 20.55
    frame["RollingStd"] = 3.0
    frame["AI_State"] = "STABLE_JITTER"

    state = RealtimeAdaptiveKalmanState(
        x=533.0,
        P=10.0,
        last_valid_x=533.0,
        previous_raw=526.0,
        previous_raw_2=526.0,
        segment_id="A",
        fuel_time="2026-01-01T00:00:00",
    )
    out = filter_ai_enhanced_adaptive_realtime(frame, state=state)

    assert out[0] > 531.5
