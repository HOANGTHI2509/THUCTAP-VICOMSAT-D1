"""Kieu du lieu state va bang chung causal cua Smooth-Tracking."""

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional


@dataclass
class VehicleFilterContext:
    """Bo nho rieng cua mot xe, khong chia se voi xe khac."""

    vehicle_id: str
    capacity_est: float = 0.0
    capacity_known: bool = False
    capacity_mode: str = "UNKNOWN_CAPACITY_MODE"
    capacity_warning: Optional[str] = "CAPACITY_NOT_CALIBRATED"
    kalman_x: Optional[float] = None
    kalman_p: float = 1.0
    last_clean_fuel: Optional[float] = None
    last_raw_fuel: Optional[float] = None
    last_time: Optional[datetime] = None
    segment_id: Optional[str] = None
    operational_state: str = "STABLE"
    stable_baseline: Optional[float] = None
    excursion_baseline: Optional[float] = None
    excursion_min: Optional[float] = None
    excursion_max: Optional[float] = None
    excursion_active: bool = False
    excursion_direction: int = 0
    excursion_start_time: Optional[datetime] = None
    excursion_elapsed_min: float = 0.0
    excursion_max_step: float = 0.0
    low_plateau_active: bool = False
    pending_direction: int = 0
    pending_samples: int = 0
    pending_elapsed_min: float = 0.0
    pending_stable_samples: int = 0
    max_deviation_pct: float = 0.0
    rebound_ratio: float = 0.0
    pullback_ratio: float = 0.0
    recovery_direction: int = 0
    recovery_samples_left: int = 0
    recovery_active: bool = False
    recovery_start_time: Optional[datetime] = None
    recovery_elapsed_min: float = 0.0
    recovery_beats: int = 0
    recovery_opposite_samples: int = 0
    confirmed_direction: int = 0
    confirmed_samples_left: int = 0
    reacquisition_step: int = 0
    last_kalman_q: float = 0.0
    last_kalman_r: float = 0.0
    last_expected_rate: float = 0.0
    last_observed_rate: float = 0.0
    last_rate_residual: float = 0.0
    classifier_probability: float = 0.0
    robust_noise: float = 0.1
    innovation_gated: bool = False
    shadow_fuel: Optional[float] = None
    shadow_values: deque = field(default_factory=lambda: deque(maxlen=5))
    confirmed_ramp_step: int = 0
    confirm_start_time: Optional[datetime] = None
    transition_progress: float = 0.0
    reacquisition_start_time: Optional[datetime] = None
    reacquisition_samples: int = 0
    trend_active: bool = False
    trend_direction: int = 0
    trend_start_time: Optional[datetime] = None
    trend_elapsed_min: float = 0.0
    trend_samples: int = 0
    trend_confidence: float = 0.0
    trend_net_change_pct: float = 0.0
    trend_directionality_ema: float = 0.0
    trend_negative_ratio_ema: float = 0.0
    trend_positive_ratio_ema: float = 0.0
    trend_rebound_max: float = 0.0
    trend_escape_triggered: bool = False
    trend_deltas: deque = field(default_factory=lambda: deque(maxlen=12))
    innovation: float = 0.0
    innovation_score: float = 0.0
    strong_up_confirmed: bool = False
    recent_upward_steps: int = 0
    pending_downward_count: int = 0
    drop_count: int = 0
    rise_count: int = 0
    upward_anchor: Optional[float] = None
    upward_samples: List[float] = field(default_factory=list)
    history_fuel: deque = field(default_factory=lambda: deque(maxlen=12))
    history_time: deque = field(default_factory=lambda: deque(maxlen=12))
    history_speed: deque = field(default_factory=lambda: deque(maxlen=12))
    history_coordinates: deque = field(default_factory=lambda: deque(maxlen=12))


@dataclass(frozen=True)
class WindowEvidence:
    """Dac trung cua so ngan tai diem hien tai."""

    recent_values: List[float]
    directionality: float
    local_std: float
    local_range: float
    directional_down: bool


@dataclass(frozen=True)
class TrendEvidence:
    """Bang chung xu huong cua so dai hon."""

    robust_downtrend: bool
    target: float


@dataclass(frozen=True)
class MotionEvidence:
    """Causal movement-quality context derived from speed and GPS only."""

    state: str
    confidence: float
    gps_displacement_meters: float
    gps_radius_meters: float
    gps_sample_count: int
    has_gps: bool
