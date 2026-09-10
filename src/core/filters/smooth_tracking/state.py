"""Kieu du lieu state va bang chung causal cua Smooth-Tracking."""

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional


@dataclass
class VehicleFilterContext:
    """Bo nho rieng cua mot xe, khong chia se voi xe khac."""

    vehicle_id: str
    capacity_est: float = 200.0
    kalman_x: Optional[float] = None
    kalman_p: float = 1.0
    last_clean_fuel: Optional[float] = None
    last_raw_fuel: Optional[float] = None
    last_time: Optional[datetime] = None
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
