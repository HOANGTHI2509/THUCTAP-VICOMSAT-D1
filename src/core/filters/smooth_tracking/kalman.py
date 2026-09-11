"""Loi Adaptive Kalman cua bo loc Smooth-Tracking."""

import math

from .config import SmoothTrackingConfig
from .state import MotionEvidence, TrendEvidence, VehicleFilterContext, WindowEvidence


def smooth_kalman_update(
    context: VehicleFilterContext,
    raw_fuel: float,
    speed: float,
    dt_minutes: float,
    ai_state: str,
    jitter: float,
    window: WindowEvidence,
    trend: TrendEvidence,
    config: SmoothTrackingConfig,
    motion: MotionEvidence,
) -> float:
    """Cap nhat Kalman trong nhanh khong co chuyen muc da xac nhan."""
    gps_context_ready = motion.gps_sample_count >= config.low_motion_min_gps_points
    is_low_motion = motion.state == "LOW_MOTION" or (
        not gps_context_ready and speed <= config.parked_speed_kmh
    )
    is_moving = motion.state == "MOVING" or (
        not gps_context_ready and speed > config.parked_speed_kmh
    )

    if is_low_motion:
        measurement_noise = config.low_motion_r if motion.state == "LOW_MOTION" else config.parked_r
        process_noise = config.low_motion_q if motion.state == "LOW_MOTION" else config.parked_q
    elif is_moving:
        measurement_noise = config.moving_r
        process_noise = config.moving_q
    else:
        # Speed and GPS disagree after enough GPS samples: filter more
        # conservatively instead of trusting either signal on its own.
        measurement_noise = max(config.parked_r, config.moving_r) * config.gps_conflict_r_multiplier
        process_noise = min(config.parked_q, config.moving_q) * config.gps_conflict_q_multiplier

    strong_bidirectional_noise = (
        ai_state in ("OSCILLATION_NOISE", "SLOSHING")
        and window.local_range >= max(5.0, jitter * 3.0)
        and window.directionality <= config.strong_noise_directionality_max
        and not trend.robust_downtrend
    )
    mild_noise = ai_state == "STABLE_JITTER" and window.local_std <= jitter
    if strong_bidirectional_noise:
        measurement_noise = max(measurement_noise, config.strong_noise_r)
        process_noise = min(process_noise, config.strong_noise_q)
    elif mild_noise:
        measurement_noise = min(measurement_noise, config.mild_noise_r)
        process_noise = max(process_noise, config.mild_noise_q)

    if raw_fuel < float(context.kalman_x) - max(jitter * 0.8, 0.6):
        context.drop_count += 1
    else:
        context.drop_count = 0

    downward_supported = (
        ai_state in ("GRADUAL_CHANGE", "DOWNWARD_SHIFT")
        or (is_moving and context.drop_count >= 3 and window.directional_down)
        or trend.robust_downtrend
    )
    if context.drop_count >= 2 and downward_supported:
        slope_lag = min(4.0, (float(context.kalman_x) - raw_fuel) / max(jitter * 0.5, 0.4))
        measurement_noise = max(10.0, measurement_noise / slope_lag)
        process_noise = max(process_noise, 0.15 * slope_lag)
        if window.directional_down:
            measurement_noise = min(measurement_noise, config.directional_r)
            process_noise = max(process_noise, config.directional_q)
        if trend.robust_downtrend:
            measurement_noise = min(measurement_noise, config.robust_trend_r)
            process_noise = max(process_noise, config.robust_trend_q)

    dt_ratio = max(0.1, min(10.0, dt_minutes / config.nominal_period_minutes))
    predicted_covariance = context.kalman_p + process_noise * dt_ratio
    gain = predicted_covariance / (predicted_covariance + measurement_noise)
    target = min(raw_fuel, trend.target) if trend.robust_downtrend else raw_fuel
    difference = target - float(context.kalman_x)

    if is_moving and difference > 0:
        compressed = min(max(0.4, jitter * 0.25), difference)
    elif (is_moving or trend.robust_downtrend or downward_supported) and difference < 0:
        if downward_supported and (window.directional_down or trend.robust_downtrend or ai_state == "GRADUAL_CHANGE"):
            compressed = difference
        else:
            compressed = max(-max(1.5, jitter * 1.2), difference)
    else:
        shrink_threshold = max(1.5, jitter * 1.5)
        compressed = shrink_threshold * math.tanh(difference / shrink_threshold)

    kalman_step = gain * compressed
    if (is_moving and window.directional_down) or trend.robust_downtrend or (downward_supported and ai_state == "GRADUAL_CHANGE"):
        kalman_step = max(-max(2.5, jitter * 1.8), kalman_step)
    context.kalman_x = float(context.kalman_x) + kalman_step
    context.kalman_p = (1.0 - gain) * predicted_covariance
    return context.kalman_x
