"""Public signal and quality contracts for Topic 1 fuel preprocessing.

The classifier and older adapters historically used a few business-event names.
This module keeps those implementation details out of the public denoising API.
"""

from __future__ import annotations

from typing import Final


SIGNAL_STATES: Final[tuple[str, ...]] = (
    "UNINITIALIZED",
    "INIT",
    "STABLE_JITTER",
    "OSCILLATION_NOISE",
    "SLOSHING",
    "GRADUAL_CHANGE",
    "UPWARD_SHIFT",
    "DOWNWARD_SHIFT",
    "SPIKE",
    "UNCERTAIN",
)

QUALITY_FLAGS: Final[tuple[str, ...]] = (
    "INITIAL_INVALID_DISCARDED",
    "VALID",
    "SMOOTH_KALMAN",
    "ZERO_DROPOUT_HELD",
    "SPIKE_HELD",
    "PENDING_UPWARD_SHIFT_HELD",
    "UPWARD_SHIFT_TRACKED",
    "UPWARD_REVERSAL_REJECTED",
    "UPWARD_REVERSAL_RESET",
    "PENDING_DOWNWARD_SHIFT_HELD",
    "DOWNWARD_SHIFT_TRACKED",
    "DOWN_EXCURSION_HELD",
    "REBOUND_RECOVERY_HELD",
    "DOWNWARD_SHIFT_TRANSITION",
    "STABLE_LEVEL_TRACKING",
)

MOTION_STATES: Final[tuple[str, ...]] = (
    "MOVING",
    "LOW_MOTION",
    "UNCERTAIN",
)

_SIGNAL_ALIASES: Final[dict[str, str]] = {
    "NORMAL": "STABLE_JITTER",
    "DROPOUT": "OSCILLATION_NOISE",
    "TRANSIENT_NOISE": "OSCILLATION_NOISE",
    "TRANSIENT_UP_NOISE": "OSCILLATION_NOISE",
    "TRANSIENT_DOWN_NOISE": "OSCILLATION_NOISE",
    "TRANSIENT_CLUSTER_NOISE": "OSCILLATION_NOISE",
    "SPIKE_UP": "SPIKE",
    "SPIKE_DOWN": "SPIKE",
    "IMPULSE_NOISE": "SPIKE",
    # Legacy model/business labels are exposed as neutral level shifts.
    "RISING": "UPWARD_SHIFT",
    "REFUEL": "UPWARD_SHIFT",
    "DRAIN": "DOWNWARD_SHIFT",
}

_QUALITY_ALIASES: Final[dict[str, str]] = {
    "PENDING_REFUEL_HELD": "PENDING_UPWARD_SHIFT_HELD",
    "REFUEL_TRACKED": "UPWARD_SHIFT_TRACKED",
    "REVERSAL_REJECTED": "UPWARD_REVERSAL_REJECTED",
    "REFUEL_REVERSAL_RESET": "UPWARD_REVERSAL_RESET",
    "PENDING_DRAIN_HELD": "PENDING_DOWNWARD_SHIFT_HELD",
    "DRAIN_CONFIRMED": "DOWNWARD_SHIFT_TRACKED",
}


def normalize_signal_state(value: object) -> str:
    """Return one stable Topic 1 signal-state value."""
    state = str(value or "UNCERTAIN").strip().upper()
    state = _SIGNAL_ALIASES.get(state, state)
    return state if state in SIGNAL_STATES else "UNCERTAIN"


def normalize_quality_flag(value: object) -> str:
    """Return one stable Topic 1 filter-action value."""
    quality = str(value or "VALID").strip().upper()
    quality = _QUALITY_ALIASES.get(quality, quality)
    return quality if quality in QUALITY_FLAGS else "VALID"


def normalize_motion_state(value: object) -> str:
    """Return one stable GPS/speed movement-quality value."""
    state = str(value or "UNCERTAIN").strip().upper()
    return state if state in MOTION_STATES else "UNCERTAIN"
