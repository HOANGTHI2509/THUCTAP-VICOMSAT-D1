from datetime import datetime, timedelta

from src.core.filters.ai_smooth_tracking_filter import AISmoothTrackingFilter


def _run(values, states, speeds, capacity=800.0):
    engine = AISmoothTrackingFilter(model_dir="")
    start = datetime(2026, 1, 1)
    results = []
    for index, (value, state, speed) in enumerate(zip(values, states, speeds)):
        results.append(
            engine.process_point(
                vehicle_id="TEST",
                timestamp=start + timedelta(minutes=2 * index),
                raw_fuel=value,
                speed=speed,
                capacity_est=capacity,
                known_ai_state=state,
            )
        )
    return results


def test_parked_oscillation_valley_does_not_pull_filter_to_bottom():
    values = [360.0, 355.0, 353.0, 347.0, 343.0, 339.0, 355.0, 358.0, 359.0]
    results = _run(values, ["OSCILLATION_NOISE"] * len(values), [0.0] * len(values))
    clean = [item["clean_fuel"] for item in results]

    assert min(clean) > 355.0
    assert abs(clean[-1] - 360.0) < 1.0


def test_small_moving_hill_is_only_followed_slightly():
    values = [360.0, 365.0, 365.0, 365.0, 360.0]
    results = _run(values, ["OSCILLATION_NOISE"] * len(values), [35.0] * len(values))
    clean = [item["clean_fuel"] for item in results]

    assert max(clean) < 361.0


def test_supported_moving_downward_trend_still_tracks_consumption():
    values = [360.0, 358.0, 356.0, 354.0, 352.0, 350.0, 348.0, 346.0]
    results = _run(values, ["GRADUAL_CHANGE"] * len(values), [45.0] * len(values))
    clean = [item["clean_fuel"] for item in results]

    assert clean[-1] < clean[1] - 1.0
    assert all(b <= a + 0.2 for a, b in zip(clean, clean[1:]))


def test_stable_lower_plateau_is_accepted_below_large_shift_threshold():
    values = [500.0, 482.0] + [478.5] * 8
    results = _run(values, ["STABLE_JITTER"] * len(values), [0.0] * len(values), capacity=537.0)
    clean = [item["clean_fuel"] for item in results]

    assert clean[-1] < 480.0
    assert "STABLE_LEVEL_TRACKING" in [item["quality_flag"] for item in results]


def test_noisy_but_directional_consumption_tracks_a_continuous_slope():
    values = [523.0, 513.0, 520.0, 515.0, 514.0, 513.0, 515.0, 511.0, 512.0, 510.0, 506.0, 511.0, 507.0, 504.0, 503.0]
    states = ["STABLE_JITTER"] * len(values)
    speeds = [42.0, 44.0, 36.0, 44.0, 33.0, 42.0, 35.0, 46.0, 40.0, 37.0, 48.0, 45.0, 47.0, 31.0, 41.0]
    results = _run(values, states, speeds, capacity=537.0)
    clean = [item["clean_fuel"] for item in results]

    assert clean[-1] < 510.0
    assert max(abs(b - a) for a, b in zip(clean, clean[1:])) <= 3.0


def test_strong_bidirectional_noise_is_filtered_more_than_mild_jitter():
    strong = [500.0, 510.0, 491.0, 509.0, 492.0, 508.0, 500.0]
    mild = [500.0, 500.8, 499.4, 500.6, 499.6, 500.4, 500.0]
    strong_out = _run(strong, ["OSCILLATION_NOISE"] * len(strong), [30.0] * len(strong))
    mild_out = _run(mild, ["STABLE_JITTER"] * len(mild), [30.0] * len(mild))

    strong_span = max(item["clean_fuel"] for item in strong_out) - min(item["clean_fuel"] for item in strong_out)
    mild_span = max(item["clean_fuel"] for item in mild_out) - min(item["clean_fuel"] for item in mild_out)
    assert strong_span < 1.0
    assert mild_span < 1.0
    assert strong_span / (max(strong) - min(strong)) < mild_span / (max(mild) - min(mild))
