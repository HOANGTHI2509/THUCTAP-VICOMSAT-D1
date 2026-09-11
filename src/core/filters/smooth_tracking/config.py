"""Cau hinh cho bo loc AI Smooth-Tracking."""

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class SmoothTrackingConfig:
    """Toan bo tham so co the tinh chinh ma khong sua loi thuat toan."""

    # None is intentional: an uncalibrated vehicle must enter UNKNOWN_CAPACITY_MODE.
    default_capacity: Optional[float] = None
    minimum_capacity: float = 30.0
    initial_fuel_fallback: float = 50.0
    capacity_headroom: float = 1.02
    inferred_capacity_headroom: float = 1.05
    reset_gap_minutes: float = 30.0
    nominal_period_minutes: float = 2.0
    stopped_speed_kmh: float = 0.5
    parked_speed_kmh: float = 1.0
    moving_speed_kmh: float = 5.0
    motion_window_points: int = 5
    low_motion_min_gps_points: int = 3
    low_motion_radius_meters: float = 30.0
    moving_min_gps_displacement_meters: float = 25.0

    noise_sigma_floor: float = 0.5
    noise_sigma_capacity_ratio: float = 0.002
    jitter_floor: float = 0.8
    jitter_capacity_ratio: float = 0.003
    spike_floor: float = 3.0
    spike_capacity_ratio: float = 0.012
    event_floor: float = 6.0
    event_capacity_ratio: float = 0.035
    level_shift_floor: float = 10.0
    level_shift_capacity_ratio: float = 0.04

    parked_r: float = 35.0
    parked_q: float = 0.03
    moving_r: float = 45.0
    moving_q: float = 0.08
    low_motion_r: float = 28.0
    low_motion_q: float = 0.06
    gps_conflict_r_multiplier: float = 1.30
    gps_conflict_q_multiplier: float = 0.75
    strong_noise_r: float = 250.0
    strong_noise_q: float = 0.01
    mild_noise_r: float = 18.0
    mild_noise_q: float = 0.12
    directional_r: float = 8.0
    directional_q: float = 1.0
    robust_trend_r: float = 6.0
    robust_trend_q: float = 1.5

    upward_hold_steps: int = 4
    downward_confirm_points: int = 3
    stable_level_gain: float = 0.65
    strong_noise_directionality_max: float = 0.50
    downward_directionality_min: float = 0.65

    # OperationalGuard thresholds are ratios of calibrated tank capacity.
    low_motion_deviation_pct: float = 0.012
    moving_deviation_pct: float = 0.020
    uncertain_deviation_pct: float = 0.016
    candidate_cumulative_pct: float = 0.018
    strong_shift_pct: float = 0.045
    confirm_shift_pct: float = 0.035
    stable_range_pct: float = 0.008
    stable_std_pct: float = 0.0035
    stable_slope_pct_per_sample: float = 0.0015
    stable_net_change_pct: float = 0.004
    gradual_max_step_pct: float = 0.0055
    unknown_gradual_max_step_pct: float = 0.0058
    rebound_cancel_ratio: float = 0.60
    recovery_lock_samples: int = 8
    recovery_min_beats: int = 2
    recovery_min_elapsed_minutes: float = 2.0
    new_baseline_accept_minutes: float = 30.0
    weak_confirm_samples: int = 5
    strong_confirm_samples: int = 2
    weak_confirm_elapsed_minutes: float = 6.0
    strong_confirm_elapsed_minutes: float = 4.0
    confirmed_tracking_samples: int = 3
    baseline_moving_gain: float = 0.15
    baseline_stationary_gain: float = 0.04
    profile_min_samples: int = 6
    rate_profile_alpha: float = 0.12
    rate_residual_sigma: float = 3.0
    robust_noise_window: int = 9
    robust_noise_floor: float = 0.10
    deviation_noise_k: float = 4.0
    candidate_noise_k: float = 5.0
    confirm_noise_k: float = 8.0
    innovation_noise_k: float = 6.0
    innovation_capacity_pct: float = 0.03
    shadow_gain: float = 0.35
    unknown_absolute_safety_floor: float = 0.5
    shift_discontinuity_pct: float = 0.015
    shift_discontinuity_noise_k: float = 3.0
    strong_up_pct: float = 0.05
    strong_up_noise_ratio: float = 12.0
    trend_min_samples: int = 3
    trend_directionality_min: float = 0.75
    trend_negative_ratio_min: float = 0.65
    trend_positive_ratio_max: float = 0.20
    trend_rebound_max: float = 0.25
    trend_net_change_pct: float = 0.01
    trend_ema_alpha: float = 0.35

    stable_q: float = 0.05
    stable_r: float = 28.0
    gradual_q: float = 1.2
    gradual_r: float = 4.0
    pending_q: float = 0.015
    pending_r: float = 220.0
    recovery_q: float = 0.008
    recovery_r: float = 250.0
    confirmed_q: float = 20.0
    confirmed_r: float = 0.5
    confirmed_q_start: float = 2.0
    confirmed_r_start: float = 6.0
    confirmed_step_start_pct: float = 0.03
    confirmed_step_end_pct: float = 0.20
    confirmed_transition_minutes: float = 4.0
    reacquisition_q: float = 6.0
    reacquisition_r_start: float = 12.0
    reacquisition_r_end: float = 0.8
    reacquisition_steps: int = 3
