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
    pending_direction: int = 0
    pending_samples: int = 0
    pending_elapsed_min: float = 0.0
    pending_stable_samples: int = 0
    max_deviation_pct: float = 0.0
    rebound_ratio: float = 0.0
    pullback_ratio: float = 0.0
    recovery_direction: int = 0
    recovery_samples_left: int = 0
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
