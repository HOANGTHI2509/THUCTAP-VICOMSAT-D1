"""Compatibility facade for the AI Smooth-Tracking filter.

New code may import from src.core.filters.smooth_tracking. This module keeps
existing dashboard, tests and integrations backward compatible.
"""

from src.core.filters.smooth_tracking import (
    AISmoothTrackingFilter,
    SmoothTrackingConfig,
    TrendEvidence,
    VehicleFilterContext,
    WindowEvidence,
    filter_smooth_tracking_dataframe,
    get_shared_smooth_engine,
)

__all__ = [
    "AISmoothTrackingFilter",
    "SmoothTrackingConfig",
    "TrendEvidence",
    "VehicleFilterContext",
    "WindowEvidence",
    "filter_smooth_tracking_dataframe",
    "get_shared_smooth_engine",
]
