"""Cau hinh cho bo loc AI Smooth-Tracking."""

from dataclasses import dataclass


@dataclass(frozen=True)
class SmoothTrackingConfig:
    """Toan bo tham so co the tinh chinh ma khong sua loi thuat toan."""

    default_capacity: float = 200.0
    minimum_capacity: float = 30.0
    initial_fuel_fallback: float = 50.0
    capacity_headroom: float = 1.02
    inferred_capacity_headroom: float = 1.05
    reset_gap_minutes: float = 120.0
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
