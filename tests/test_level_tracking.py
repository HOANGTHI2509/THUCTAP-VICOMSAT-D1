import numpy as np
import pandas as pd

from src.core.filters.ai_enhanced_adaptive_realtime import (
    RealtimeAdaptiveKalmanState,
    filter_ai_enhanced_adaptive_realtime as run_filter,
)


def frame(levels, label="OSCILLATION_NOISE", speed=35.0):
    return pd.DataFrame({
        "FuelLevel": levels,
        "FuelTime": pd.date_range("2026-08-11", periods=len(levels), freq="2min"),
        "Speed": speed, "AI_State": label, "AI_State_Confidence": 0.95,
        "RollingStd": 10.0, "flat_jitter_threshold": 0.8,
        "noise_sigma_liters": 0.5, "capacity_est": 800.0,
        "event_threshold": 28.0,
    })


def test_moving_u_does_not_pull_output_to_bottom():
    df = frame([300.] * 5 + [292., 286., 280., 282., 290.] + [300.] * 8)
    out = run_filter(df)
    assert min(out) > 298.0


def test_small_stable_rebound_below_refuel_threshold():
    df = frame([270.] * 5 + [277.] * 18)
    out = run_filter(df)
    assert max(out[5:10]) < 271.0
    assert abs(out[-1] - 277.) < 1.0


def test_confirmed_drop_tracks_stable_lower_level():
    df = frame([300.] * 5 + [260.] * 6, label="DOWNWARD_SHIFT")
    df["AI_State"] = ["STABLE_JITTER"] * 5 + ["OSCILLATION_NOISE"] * 3 + ["DOWNWARD_SHIFT"] * 3
    trace = []
    out = run_filter(df, config={"trace_collector": trace})
    assert abs(out[-1] - 260.) < 1.0
    assert any(t["branch_selected"] == "confirmed_downward_level" for t in trace)


def test_streaming_roundtrip_matches_batch():
    df = frame([300.] * 5 + [290., 280., 290.] + [298.] * 8 + [305.] * 16)
    expected, expected_state = run_filter(df, return_state=True)
    state = RealtimeAdaptiveKalmanState()
    actual = []
    for i in range(len(df)):
        values, state = run_filter(df.iloc[i:i+1], state=state, return_state=True)
        actual.extend(values)
        state = RealtimeAdaptiveKalmanState.from_dict(state.to_dict())
    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-10)
    assert state.to_dict() == expected_state.to_dict()
