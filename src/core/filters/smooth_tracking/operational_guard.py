"""Causal operational protection for implausibly deep fuel excursions."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import math
import statistics
from typing import Optional

from .config import SmoothTrackingConfig
from .state import VehicleFilterContext


@dataclass(frozen=True)
class GuardDecision:
    """A published-output decision made before the legacy tracking branches."""

    clean_fuel: float
    quality_flag: str


class OperationalGuard:
    """Keep extreme low-positive U excursions out of published CleanFuel.

    The guard is stateless itself. All mutable memory lives in the per-vehicle
    ``VehicleFilterContext``, so engines and vehicles cannot share excursions.
    """

    def __init__(self, config: SmoothTrackingConfig):
        self.config = config

    @staticmethod
    def _robust_noise(context: VehicleFilterContext) -> float:
        values = list(context.history_fuel)
        if len(values) < 3:
            return 0.5
        deltas = [values[index] - values[index - 1] for index in range(1, len(values))]
        median_delta = statistics.median(deltas)
        mad = statistics.median(abs(delta - median_delta) for delta in deltas)
        return max(0.5, 1.4826 * mad)

    @staticmethod
    def reset(context: VehicleFilterContext) -> None:
        context.excursion_active = False
        context.excursion_direction = None
        context.excursion_baseline = None
        context.excursion_min = None
        context.excursion_max = None
        context.excursion_start_time = None
        context.excursion_samples.clear()
        context.excursion_confirmed = False
        context.recovery_active = False
        context.rebound_ratio = 0.0
        context.pending_downward_count = 0
        context.upward_anchor = None
        context.upward_samples.clear()
        context.operational_state = "STABLE" if context.last_clean_fuel is not None else "UNINITIALIZED"

    def evaluate(
        self,
        context: VehicleFilterContext,
        raw_fuel: float,
        timestamp: datetime,
        dt_minutes: float,
        level_shift_threshold: float,
    ) -> Optional[GuardDecision]:
        """Return a hold/transition decision, or ``None`` for normal processing."""
        if not math.isfinite(raw_fuel) or raw_fuel <= 0.0 or context.last_clean_fuel is None:
            return None

        if not context.excursion_active:
            baseline = float(context.last_clean_fuel)
            depth = baseline - raw_fuel
            noise = self._robust_noise(context)
            extreme_threshold = max(
                self.config.extreme_excursion_baseline_ratio * max(abs(baseline), 1.0),
                self.config.extreme_excursion_noise_multiplier * noise,
                2.0 * level_shift_threshold,
            )
            if depth < extreme_threshold:
                return None

            context.excursion_active = True
            context.excursion_direction = "DOWN"
            context.excursion_baseline = baseline
            context.excursion_min = raw_fuel
            context.excursion_max = raw_fuel
            context.excursion_start_time = timestamp
            context.excursion_samples = [raw_fuel]
            context.excursion_confirmed = False
            context.recovery_active = False
            context.rebound_ratio = 0.0
            context.pending_downward_count = 0
            context.upward_anchor = None
            context.upward_samples.clear()
            context.operational_state = "DOWN_EXCURSION"
            return GuardDecision(baseline, "DOWN_EXCURSION_HELD")

        if context.excursion_direction != "DOWN" or context.excursion_baseline is None:
            self.reset(context)
            return None

        baseline = float(context.excursion_baseline)
        context.excursion_min = min(float(context.excursion_min), raw_fuel)
        context.excursion_max = max(float(context.excursion_max), raw_fuel)
        context.excursion_samples.append(raw_fuel)
        context.excursion_samples = context.excursion_samples[-24:]

        depth = max(baseline - float(context.excursion_min), 1e-6)
        depth_ratio = depth / max(abs(baseline), 1.0)
        context.rebound_ratio = max(0.0, (raw_fuel - float(context.excursion_min)) / depth)
        elapsed_minutes = max(
            0.0,
            (timestamp - context.excursion_start_time).total_seconds() / 60.0,
        ) if context.excursion_start_time is not None else 0.0

        # A strong return is part of the same U event, not a new upward shift.
        if context.rebound_ratio >= self.config.excursion_rebound_cancel_ratio:
            context.recovery_active = True
            context.operational_state = "REBOUND_RECOVERY"

        near_old_baseline = abs(raw_fuel - baseline) <= max(
            level_shift_threshold,
            self.config.recovery_baseline_ratio * max(abs(baseline), 1.0),
        )
        if context.recovery_active:
            if near_old_baseline:
                self.reset(context)
                return None
            return GuardDecision(float(context.last_clean_fuel), "REBOUND_RECOVERY_HELD")

        recent = context.excursion_samples[-self.config.excursion_plateau_points :]
        noise = self._robust_noise(context)
        plateau_tolerance = max(
            self.config.excursion_plateau_floor,
            self.config.excursion_plateau_noise_multiplier * noise,
        )
        stable_low_plateau = (
            len(recent) >= self.config.excursion_plateau_points
            and max(recent) - min(recent) <= plateau_tolerance
            and context.rebound_ratio <= self.config.excursion_accept_max_rebound_ratio
        )

        if (
            not context.excursion_confirmed
            # A near-total instantaneous loss is dropout-like. Elapsed time
            # alone cannot turn it into a physical baseline shift; keep the
            # published line protected until rebound or a segment/gap reset.
            and depth_ratio < self.config.dropout_like_depth_ratio
            and elapsed_minutes >= self.config.extreme_excursion_accept_minutes
            and stable_low_plateau
        ):
            context.excursion_confirmed = True
            context.operational_state = "DOWNWARD_CONFIRMED"

        if context.excursion_confirmed:
            target = float(statistics.median(recent))
            gain = min(0.45, max(0.15, dt_minutes / self.config.excursion_transition_minutes))
            clean = float(context.last_clean_fuel) + gain * (target - float(context.last_clean_fuel))
            context.kalman_x = clean
            if abs(clean - target) <= plateau_tolerance:
                self.reset(context)
                context.operational_state = "BASELINE_REACQUISITION"
            return GuardDecision(clean, "DOWNWARD_SHIFT_TRANSITION")

        context.operational_state = "LOW_PLATEAU_UNCERTAIN"
        return GuardDecision(float(context.last_clean_fuel), "DOWN_EXCURSION_HELD")
