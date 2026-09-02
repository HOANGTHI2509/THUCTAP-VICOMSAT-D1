"""Neutral signal-state vocabulary for Topic 1.

Legacy names are accepted only when importing historical datasets/models.
New training datasets and public outputs use these neutral labels.
"""

LEGACY_TO_SIGNAL = {
    "REFUEL": "UPWARD_SHIFT",
    "DRAIN": "DOWNWARD_SHIFT",
    "CONSUMPTION": "GRADUAL_CHANGE",
    "STABLE_JITTER": "STABLE_JITTER",
    "SLOSHING_NOISE": "OSCILLATION_NOISE",
    "SPIKE": "OSCILLATION_NOISE",
    "UNKNOWN": "UNKNOWN",
}
SIGNAL_TO_LEGACY = {
    "UPWARD_SHIFT": "REFUEL",
    "DOWNWARD_SHIFT": "DRAIN",
    "GRADUAL_CHANGE": "CONSUMPTION",
    "STABLE_JITTER": "STABLE_JITTER",
    "OSCILLATION_NOISE": "SLOSHING_NOISE",
    "UNKNOWN": "UNKNOWN",
}
SIGNAL_LABELS = {
    "UPWARD_SHIFT",
    "DOWNWARD_SHIFT",
    "GRADUAL_CHANGE",
    "STABLE_JITTER",
    "OSCILLATION_NOISE",
    "UNKNOWN",
    "IMPULSE_NOISE",
}


def to_signal_label(value: object) -> str:
    text = str(value).upper().strip()
    return LEGACY_TO_SIGNAL.get(text, text) if text in SIGNAL_LABELS or text in LEGACY_TO_SIGNAL else "UNKNOWN"


def to_legacy_label(value: object) -> str:
    text = str(value).upper().strip()
    return SIGNAL_TO_LEGACY.get(text, text) if text in SIGNAL_TO_LEGACY else "UNKNOWN"
