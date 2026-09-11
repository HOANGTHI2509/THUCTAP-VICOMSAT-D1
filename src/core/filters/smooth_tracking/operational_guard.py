"""Causal operational guard between signal classification and Kalman filtering."""

from __future__ import annotations

from dataclasses import dataclass, field
import statistics
from typing import Dict, Tuple

from .config import SmoothTrackingConfig
from .state import MotionEvidence, TrendEvidence, VehicleFilterContext, WindowEvidence


OPERATIONAL_STATES = (
    "STABLE",
    "GRADUAL_TRACKING",
    "PENDING_DOWNWARD",
    "PENDING_UPWARD",
    "REBOUND_RECOVERY",
    "OSCILLATION",
    "DOWNWARD_CONFIRMED",
    "UPWARD_CONFIRMED",
    "BASELINE_REACQUISITION",
)


@dataclass
class _RateStats:
    mean: float = 0.0
    count: int = 0

    def update(self, value: float, alpha: float) -> None:
        self.mean = value if self.count == 0 else (1.0 - alpha) * self.mean + alpha * value
        self.count += 1


@dataclass
class FuelRateProfiles:
    """Online, causal speed-conditioned fuel-rate hierarchy."""

    config: SmoothTrackingConfig
    vehicle_bins: Dict[Tuple[str, str], _RateStats] = field(default_factory=dict)
    vehicle_all: Dict[str, _RateStats] = field(default_factory=dict)
    fleet_bins: Dict[str, _RateStats] = field(default_factory=dict)
    fleet_all: _RateStats = field(default_factory=_RateStats)

    @staticmethod
    def speed_bin(recent_speed: float) -> str:
        if recent_speed <= 0.5:
            return "STOPPED"
        if recent_speed <= 20.0:
            return "LOW"
        if recent_speed <= 50.0:
            return "MID"
        return "HIGH"

    def expected(self, vehicle_id: str, speed_bin: str, capacity: float) -> float:
        candidates = (
            self.vehicle_bins.get((vehicle_id, speed_bin)),
            self.vehicle_all.get(vehicle_id),
            self.fleet_bins.get(speed_bin),
            self.fleet_all,
        )
        for stats in candidates:
            if stats is not None and stats.count >= self.config.profile_min_samples:
                return stats.mean
        # Conservative priors, scaled by tank capacity (litres/minute).
        ratio = {"STOPPED": 0.0, "LOW": -0.00010, "MID": -0.00025, "HIGH": -0.00040}[speed_bin]
        return ratio * capacity

    def update(self, vehicle_id: str, speed_bin: str, observed_rate: float, capacity: float) -> None:
        # Do not teach profiles event/noise rates. Legitimate consumption is bounded
        # in capacity-relative units so profiles remain useful across tank sizes.
        lower, upper = -0.003 * capacity, 0.0005 * capacity
        if not lower <= observed_rate <= upper:
            return
        entries = (
            self.vehicle_bins.setdefault((vehicle_id, speed_bin), _RateStats()),
            self.vehicle_all.setdefault(vehicle_id, _RateStats()),
            self.fleet_bins.setdefault(speed_bin, _RateStats()),
            self.fleet_all,
        )
        for stats in entries:
            stats.update(observed_rate, self.config.rate_profile_alpha)


@dataclass(frozen=True)
class GuardDecision:
    state: str
    target: float
    q: float
    r: float
    guard_active: bool
    quality_flag: str
    deviation_pct: float
    expected_rate: float
    observed_rate: float
    rate_residual: float


class OperationalGuard:
    """Own the physical state machine; classifier output is advisory evidence only."""

    def __init__(self, config: SmoothTrackingConfig):
        self.config = config
        self.rate_profiles = FuelRateProfiles(config)

    def reset(self, context: VehicleFilterContext, raw: float) -> None:
        context.operational_state = "STABLE"
        context.stable_baseline = raw
        self._clear_excursion(context)
        context.recovery_direction = 0
        context.recovery_samples_left = 0
        context.confirmed_direction = 0
        context.confirmed_samples_left = 0
        context.reacquisition_step = 0

    @staticmethod
    def _clear_excursion(context: VehicleFilterContext) -> None:
        context.excursion_baseline = None
        context.excursion_min = None
        context.excursion_max = None
        context.pending_direction = 0
        context.pending_samples = 0
        context.pending_elapsed_min = 0.0
        context.pending_stable_samples = 0
        context.max_deviation_pct = 0.0
        context.rebound_ratio = 0.0
        context.pullback_ratio = 0.0
        context.shadow_fuel = None
        context.shadow_values.clear()

    def _robust_noise(self, context: VehicleFilterContext, raw: float) -> float:
        values = list(context.history_fuel)[-self.config.robust_noise_window:] + [raw]
        differences = [values[index] - values[index - 1] for index in range(1, len(values))]
        if len(differences) < 3:
            return self.config.robust_noise_floor
        center = statistics.median(differences)
        mad = statistics.median(abs(value - center) for value in differences)
        return max(self.config.robust_noise_floor, 1.4826 * mad)

    def _stable_window(self, window: WindowEvidence, capacity: float) -> bool:
        values = window.recent_values[-3:]
        if len(values) < 3:
            return False
        slope = (values[-1] - values[0]) / (len(values) - 1)
        return (
            max(values) - min(values) <= self.config.stable_range_pct * capacity
            and statistics.pstdev(values) <= self.config.stable_std_pct * capacity
            and abs(slope) <= self.config.stable_slope_pct_per_sample * capacity
            and abs(values[-1] - values[0]) <= self.config.stable_net_change_pct * capacity
        )

    def _mapping(self, state: str, context: VehicleFilterContext) -> Tuple[float, float]:
        c = self.config
        if state == "STABLE":
            return c.stable_q, c.stable_r
        if state == "GRADUAL_TRACKING":
            return c.gradual_q, c.gradual_r
        if state in ("PENDING_DOWNWARD", "PENDING_UPWARD"):
            return c.pending_q, c.pending_r
        if state in ("OSCILLATION", "REBOUND_RECOVERY"):
            return c.recovery_q, c.recovery_r
        if state in ("DOWNWARD_CONFIRMED", "UPWARD_CONFIRMED"):
            fraction = min(1.0, context.confirmed_ramp_step / max(1, c.confirmed_tracking_samples - 1))
            q = c.confirmed_q_start + fraction * (c.confirmed_q - c.confirmed_q_start)
            r = c.confirmed_r_start + fraction * (c.confirmed_r - c.confirmed_r_start)
            return q, r
        step = min(context.reacquisition_step, max(1, c.reacquisition_steps - 1))
        fraction = step / max(1, c.reacquisition_steps - 1)
        return c.reacquisition_q, c.reacquisition_r_start + fraction * (c.reacquisition_r_end - c.reacquisition_r_start)

    def _start_pending(self, context: VehicleFilterContext, direction: int, raw: float) -> None:
        context.pending_direction = direction
        context.pending_samples = 1
        context.pending_elapsed_min = 0.0
        context.pending_stable_samples = 0
        context.excursion_baseline = float(context.stable_baseline)
        context.excursion_min = raw
        context.excursion_max = raw
        context.shadow_fuel = raw
        context.shadow_values.clear()
        context.shadow_values.append(raw)
        context.max_deviation_pct = abs(raw - context.excursion_baseline) / context.capacity_est
        context.rebound_ratio = 0.0
        context.pullback_ratio = 0.0
        context.operational_state = "PENDING_DOWNWARD" if direction < 0 else "PENDING_UPWARD"

    def evaluate(
        self,
        context: VehicleFilterContext,
        raw: float,
        speed: float,
        dt_minutes: float,
        model_state: str,
        model_probability: float,
        motion: MotionEvidence,
        window: WindowEvidence,
        trend: TrendEvidence,
    ) -> GuardDecision:
        c = self.config
        capacity = context.capacity_est
        baseline = float(context.stable_baseline if context.stable_baseline is not None else context.last_clean_fuel)
        deviation = raw - baseline
        deviation_pct = deviation / capacity
        recent_speeds = list(context.history_speed)[-2:] + [speed]
        recent_speed = float(statistics.median(recent_speeds))
        speed_bin = self.rate_profiles.speed_bin(recent_speed)
        observed_rate = (raw - float(context.last_raw_fuel)) / dt_minutes if context.last_raw_fuel is not None else 0.0
        expected_rate = self.rate_profiles.expected(context.vehicle_id, speed_bin, capacity)
        rate_residual = observed_rate - expected_rate
        motion_state = motion.state if motion.state != "UNCERTAIN" else ("LOW_MOTION" if recent_speed <= 0.5 else "MOVING")
        robust_noise = self._robust_noise(context, raw)
        context.robust_noise = robust_noise
        percent_threshold = (c.low_motion_deviation_pct if motion_state == "LOW_MOTION" else c.moving_deviation_pct) * capacity if context.capacity_known else 0.0
        threshold = max(percent_threshold, c.deviation_noise_k * robust_noise)
        if motion.state == "UNCERTAIN":
            threshold = max(c.uncertain_deviation_pct * capacity if context.capacity_known else 0.0, c.deviation_noise_k * robust_noise)
        candidate_threshold = max(c.candidate_cumulative_pct * capacity if context.capacity_known else 0.0, c.candidate_noise_k * robust_noise)
        confirm_threshold = max(c.confirm_shift_pct * capacity if context.capacity_known else 0.0, c.confirm_noise_k * robust_noise)
        strong_threshold = max(c.strong_shift_pct * capacity if context.capacity_known else 0.0, c.confirm_noise_k * 1.25 * robust_noise)
        innovation_threshold = max(c.innovation_capacity_pct * capacity if context.capacity_known else 0.0, c.innovation_noise_k * robust_noise)
        context.innovation_gated = (
            abs(raw - float(context.last_clean_fuel)) > innovation_threshold
            and abs(raw - float(context.last_raw_fuel)) > innovation_threshold
        )
        stable_window = self._stable_window(window, capacity)

        persistent_moving_down = (
            sum(value > c.stopped_speed_kmh for value in recent_speeds) >= 2
            and trend.robust_downtrend
        )

        if persistent_moving_down and (
            context.pending_direction < 0 or context.recovery_samples_left > 0
        ):
            self._clear_excursion(context)
            context.recovery_samples_left = 0
            context.operational_state = "GRADUAL_TRACKING"
            q, r = self._mapping(context.operational_state, context)
            return GuardDecision(context.operational_state, raw, q, r, False, "SMOOTH_KALMAN", deviation_pct, expected_rate, observed_rate, rate_residual)

        if context.operational_state in ("DOWNWARD_CONFIRMED", "UPWARD_CONFIRMED"):
            continuation = context.confirmed_direction * (raw - baseline) >= -c.stable_net_change_pct * capacity
            if continuation and not stable_window and context.confirmed_samples_left > 0:
                context.confirmed_samples_left -= 1
                context.confirmed_ramp_step += 1
                context.stable_baseline = raw
                q, r = self._mapping(context.operational_state, context)
                return GuardDecision(context.operational_state, raw, q, r, False, "DOWNWARD_SHIFT_TRACKED" if context.confirmed_direction < 0 else "UPWARD_SHIFT_TRACKED", deviation_pct, expected_rate, observed_rate, rate_residual)
            context.operational_state = "BASELINE_REACQUISITION"
            context.reacquisition_step = 0

        if context.operational_state == "BASELINE_REACQUISITION":
            if stable_window:
                q, r = self._mapping("BASELINE_REACQUISITION", context)
                cluster_target = float(statistics.median(window.recent_values[-3:]))
                return GuardDecision("BASELINE_REACQUISITION", cluster_target, q, r, False, "STABLE_LEVEL_TRACKING", deviation_pct, expected_rate, observed_rate, rate_residual)
            # A noisy reacquisition stays guarded, but R is stepped down on each
            # subsequent stable window rather than being pinned at strong-noise R.
            context.operational_state = "OSCILLATION"

        # Recovery suppresses the opposite event created by U-shape overshoot.
        if context.recovery_samples_left > 0:
            context.recovery_samples_left -= 1
            context.operational_state = "REBOUND_RECOVERY"
            recovery_side_ok = (
                context.recovery_direction < 0
                and raw <= baseline + c.stable_net_change_pct * capacity
            ) or (
                context.recovery_direction > 0
                and raw >= baseline - c.stable_net_change_pct * capacity
            )
            if stable_window and recovery_side_ok:
                context.operational_state = "BASELINE_REACQUISITION"
                context.reacquisition_step = min(context.reacquisition_step + 1, c.reacquisition_steps - 1)
            elif context.recovery_samples_left == 0:
                context.operational_state = "OSCILLATION"
            recovery_target = float(statistics.median(window.recent_values[-3:])) if context.operational_state == "BASELINE_REACQUISITION" else baseline
            return self._decision(context, recovery_target, deviation_pct, expected_rate, observed_rate, rate_residual, True)

        direction = -1 if deviation < 0 else 1
        cumulative_trigger = abs(deviation) >= candidate_threshold
        rate_capacity_term = 0.0025 * capacity if context.capacity_known else 0.0
        rate_trigger = abs(rate_residual) > max(rate_capacity_term, c.candidate_noise_k * robust_noise / max(dt_minutes, 0.1), abs(expected_rate) * c.rate_residual_sigma)
        classifier_support = model_probability >= 0.60 and model_state in ("DOWNWARD_SHIFT", "UPWARD_SHIFT")
        candidate_trigger = abs(deviation) >= threshold or cumulative_trigger or rate_trigger or classifier_support

        if context.pending_direction:
            context.pending_samples += 1
            context.pending_elapsed_min += dt_minutes
            context.shadow_values.append(raw)
            shadow_target = float(statistics.median(list(context.shadow_values)[-3:]))
            context.shadow_fuel = (
                shadow_target if context.shadow_fuel is None
                else (1.0 - c.shadow_gain) * context.shadow_fuel + c.shadow_gain * shadow_target
            )
            stable_step_threshold = max(c.stable_range_pct * capacity if context.capacity_known else 0.0, 2.0 * robust_noise)
            if abs(raw - float(context.last_raw_fuel)) <= stable_step_threshold:
                context.pending_stable_samples += 1
            else:
                context.pending_stable_samples = 0
            context.excursion_min = min(float(context.excursion_min), raw)
            context.excursion_max = max(float(context.excursion_max), raw)
            context.max_deviation_pct = max(context.max_deviation_pct, abs(raw - float(context.excursion_baseline)) / capacity)
            if context.pending_direction < 0:
                depth = float(context.excursion_baseline) - float(context.excursion_min)
                context.rebound_ratio = max(0.0, (raw - float(context.excursion_min)) / depth) if depth > 0 else 0.0
                cancelled = context.rebound_ratio >= c.rebound_cancel_ratio
            else:
                height = float(context.excursion_max) - float(context.excursion_baseline)
                context.pullback_ratio = max(0.0, (float(context.excursion_max) - raw) / height) if height > 0 else 0.0
                cancelled = context.pullback_ratio >= c.rebound_cancel_ratio
            if cancelled:
                old_direction = context.pending_direction
                context.operational_state = "REBOUND_RECOVERY"
                context.recovery_direction = old_direction
                context.recovery_samples_left = c.recovery_lock_samples
                context.reacquisition_step = 0
                context.pending_direction = 0
                q, r = self._mapping(context.operational_state, context)
                return GuardDecision(context.operational_state, baseline, q, r, True, "SMOOTH_KALMAN", deviation_pct, expected_rate, observed_rate, rate_residual)

            absolute_shift = abs(raw - float(context.excursion_baseline))
            magnitude = absolute_shift / capacity
            directionality = window.directionality
            no_rebound = max(context.rebound_ratio, context.pullback_ratio) < 0.35
            if (
                absolute_shift < confirm_threshold
                and stable_window
                and context.pending_stable_samples >= 3
                and context.pending_elapsed_min >= c.weak_confirm_elapsed_minutes
            ):
                minor_direction = context.pending_direction
                context.operational_state = "BASELINE_REACQUISITION" if minor_direction < 0 else "STABLE"
                context.reacquisition_step = 0
                context.pending_direction = 0
                q, r = self._mapping(context.operational_state, context)
                target = float(context.shadow_fuel) if minor_direction < 0 else baseline
                quality = "STABLE_LEVEL_TRACKING" if minor_direction < 0 else "SMOOTH_KALMAN"
                return GuardDecision(context.operational_state, target, q, r, False, quality, deviation_pct, expected_rate, observed_rate, rate_residual)
            directional_step = context.pending_direction * (raw - float(context.last_raw_fuel))
            strong = (
                absolute_shift >= strong_threshold
                and directionality >= 0.80
                and no_rebound
                and model_state != "SPIKE"
                and (directional_step >= max(0.015 * capacity if context.capacity_known else 0.0, 3.0 * robust_noise) or absolute_shift >= 1.75 * strong_threshold)
            )
            weak = (
                absolute_shift >= confirm_threshold
                and no_rebound
                and stable_window
                and context.pending_stable_samples >= 3
            )
            confirm = (
                strong and context.pending_samples >= c.strong_confirm_samples and context.pending_elapsed_min >= c.strong_confirm_elapsed_minutes
            ) or (
                weak and context.pending_samples >= c.weak_confirm_samples and context.pending_elapsed_min >= c.weak_confirm_elapsed_minutes
            )
            if confirm:
                confirmed_direction = context.pending_direction
                context.operational_state = "DOWNWARD_CONFIRMED" if confirmed_direction < 0 else "UPWARD_CONFIRMED"
                context.stable_baseline = float(context.shadow_fuel)
                context.confirmed_direction = confirmed_direction
                context.confirmed_samples_left = c.confirmed_tracking_samples
                context.confirmed_ramp_step = 0
                context.reacquisition_step = 0
                context.pending_direction = 0
                return self._decision(context, float(context.shadow_fuel), deviation_pct, expected_rate, observed_rate, rate_residual, False)
            context.operational_state = "PENDING_DOWNWARD" if context.pending_direction < 0 else "PENDING_UPWARD"
            # Capacity-relative expected-rate soft hold. A candidate can move only
            # with plausible vehicle consumption, never jump directly to Raw.
            plausible_rate = expected_rate
            if context.pending_direction > 0:
                plausible_rate = max(0.0, expected_rate)
            target = float(context.last_clean_fuel) + plausible_rate * dt_minutes
            q, r = self._mapping(context.operational_state, context)
            return GuardDecision(context.operational_state, target, q, r, True, "PENDING_DOWNWARD_SHIFT_HELD" if context.pending_direction < 0 else "PENDING_UPWARD_SHIFT_HELD", deviation_pct, expected_rate, observed_rate, rate_residual)

        gradual = (
            recent_speed > c.stopped_speed_kmh
            and sum(value > c.stopped_speed_kmh for value in recent_speeds) >= 2
            and deviation <= 0.0
            and not context.innovation_gated
            and abs(observed_rate) <= max(0.003 * capacity, abs(expected_rate) * 5.0)
            and (
                model_state == "GRADUAL_CHANGE"
                or context.operational_state == "GRADUAL_TRACKING"
                or trend.robust_downtrend
                or (observed_rate < 0 and abs(deviation) < threshold)
            )
        )
        if gradual:
            context.operational_state = "GRADUAL_TRACKING"
        elif candidate_trigger:
            self._start_pending(context, direction, raw)
            target = float(context.last_clean_fuel) + (expected_rate if direction < 0 else max(0.0, expected_rate)) * dt_minutes
            q, r = self._mapping(context.operational_state, context)
            return GuardDecision(context.operational_state, target, q, r, True, "PENDING_DOWNWARD_SHIFT_HELD" if direction < 0 else "PENDING_UPWARD_SHIFT_HELD", deviation_pct, expected_rate, observed_rate, rate_residual)
        elif model_state in ("OSCILLATION_NOISE", "SLOSHING", "SPIKE") and window.local_range >= threshold:
            context.operational_state = "OSCILLATION"
        else:
            context.operational_state = "STABLE"

        if context.operational_state in ("STABLE", "GRADUAL_TRACKING"):
            self.rate_profiles.update(context.vehicle_id, speed_bin, observed_rate, capacity)
        q, r = self._mapping(context.operational_state, context)
        if context.operational_state == "STABLE" and motion.state == "UNCERTAIN" and recent_speed <= c.stopped_speed_kmh:
            q, r = self.config.parked_q, self.config.parked_r
        elif context.operational_state == "STABLE" and motion.state == "UNCERTAIN":
            r = max(r, self.config.moving_r * self.config.gps_conflict_r_multiplier)
        elif context.operational_state == "STABLE" and motion.state == "LOW_MOTION":
            r = min(r, self.config.low_motion_r)
        return GuardDecision(context.operational_state, raw, q, r, False, "SMOOTH_KALMAN", deviation_pct, expected_rate, observed_rate, rate_residual)

    def after_update(self, context: VehicleFilterContext, clean: float, window: WindowEvidence, motion: MotionEvidence) -> None:
        stable_window = self._stable_window(window, context.capacity_est)
        if context.operational_state == "BASELINE_REACQUISITION":
            context.reacquisition_step += 1
            if context.reacquisition_step >= self.config.reacquisition_steps:
                context.stable_baseline = clean
                context.operational_state = "STABLE"
                self._clear_excursion(context)
        may_update_baseline = (
            context.operational_state in ("STABLE", "GRADUAL_TRACKING", "BASELINE_REACQUISITION")
            and (motion.state == "MOVING" or stable_window)
        )
        if may_update_baseline:
            gain = self.config.baseline_moving_gain if motion.state == "MOVING" else self.config.baseline_stationary_gain
            context.stable_baseline = float(context.stable_baseline) + gain * (clean - float(context.stable_baseline))

    def _decision(self, context, target, deviation_pct, expected, observed, residual, active):
        q, r = self._mapping(context.operational_state, context)
        quality = {
            "DOWNWARD_CONFIRMED": "DOWNWARD_SHIFT_TRACKED",
            "UPWARD_CONFIRMED": "UPWARD_SHIFT_TRACKED",
        }.get(context.operational_state, "SMOOTH_KALMAN")
        return GuardDecision(context.operational_state, target, q, r, active, quality, deviation_pct, expected, observed, residual)
