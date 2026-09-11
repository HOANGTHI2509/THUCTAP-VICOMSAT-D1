"""Causal OperationalGuard between Random Forest evidence and Kalman."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
import statistics
from typing import Dict, Tuple

from .config import SmoothTrackingConfig
from .state import MotionEvidence, TrendEvidence, VehicleFilterContext, WindowEvidence

OPERATIONAL_STATES = (
    "STABLE", "DOWN_EXCURSION", "GRADUAL_TRACKING", "PENDING_DOWNWARD",
    "PENDING_UPWARD", "LOW_PLATEAU_UNCERTAIN", "REBOUND_RECOVERY",
    "U_SHAPE_CONFIRMED", "OSCILLATION", "PERSISTENT_TREND_ESCAPE",
    "DOWNWARD_CONFIRMED", "UPWARD_CONFIRMED", "BASELINE_REACQUISITION",
)


@dataclass
class _RateStats:
    mean: float = 0.0
    count: int = 0

    def update(self, value: float, alpha: float) -> None:
        self.mean = value if self.count == 0 else (1-alpha)*self.mean + alpha*value
        self.count += 1


@dataclass
class FuelRateProfiles:
    config: SmoothTrackingConfig
    vehicle_bins: Dict[Tuple[str, str], _RateStats] = field(default_factory=dict)
    vehicle_all: Dict[str, _RateStats] = field(default_factory=dict)
    fleet_bins: Dict[str, _RateStats] = field(default_factory=dict)
    fleet_all: _RateStats = field(default_factory=_RateStats)

    @staticmethod
    def speed_bin(speed: float) -> str:
        return "STOPPED" if speed <= .5 else "LOW" if speed <= 20 else "MID" if speed <= 50 else "HIGH"

    def expected(self, vehicle_id: str, speed_bin: str, scale: float) -> float:
        for stats in (self.vehicle_bins.get((vehicle_id, speed_bin)), self.vehicle_all.get(vehicle_id),
                      self.fleet_bins.get(speed_bin), self.fleet_all):
            if stats is not None and stats.count >= self.config.profile_min_samples:
                return stats.mean
        return {"STOPPED": 0., "LOW": -.00010, "MID": -.00025, "HIGH": -.00040}[speed_bin] * scale

    def update(self, vehicle_id: str, speed_bin: str, rate: float, scale: float) -> None:
        if not -.003*scale <= rate <= .0005*scale:
            return
        for stats in (self.vehicle_bins.setdefault((vehicle_id, speed_bin), _RateStats()),
                      self.vehicle_all.setdefault(vehicle_id, _RateStats()),
                      self.fleet_bins.setdefault(speed_bin, _RateStats()), self.fleet_all):
            stats.update(rate, self.config.rate_profile_alpha)


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
    """Stateful physical guard. Classifier output is advisory evidence only."""
    def __init__(self, config: SmoothTrackingConfig):
        self.config = config
        self.rate_profiles = FuelRateProfiles(config)

    def reset(self, ctx: VehicleFilterContext, raw: float) -> None:
        ctx.operational_state, ctx.stable_baseline = "STABLE", raw
        self._clear_excursion(ctx)
        ctx.recovery_active, ctx.recovery_direction = False, 0
        ctx.confirmed_direction, ctx.confirm_start_time, ctx.transition_progress = 0, None, 0.
        ctx.reacquisition_start_time, ctx.reacquisition_samples = None, 0
        self._reset_trend(ctx)

    @staticmethod
    def _clear_excursion(ctx: VehicleFilterContext) -> None:
        ctx.excursion_active, ctx.excursion_direction = False, 0
        ctx.excursion_baseline = ctx.excursion_min = ctx.excursion_max = None
        ctx.excursion_start_time, ctx.excursion_elapsed_min, ctx.excursion_max_step = None, 0., 0.
        ctx.pending_direction, ctx.pending_samples, ctx.pending_elapsed_min = 0, 0, 0.
        ctx.pending_stable_samples, ctx.max_deviation_pct = 0, 0.
        ctx.rebound_ratio = ctx.pullback_ratio = 0.
        ctx.shadow_fuel = None
        ctx.shadow_values.clear()
        ctx.low_plateau_active = False

    @staticmethod
    def _reset_trend(ctx: VehicleFilterContext) -> None:
        ctx.trend_active, ctx.trend_direction = False, 0
        ctx.trend_start_time, ctx.trend_elapsed_min, ctx.trend_samples = None, 0., 0
        ctx.trend_confidence = ctx.trend_net_change_pct = 0.
        ctx.trend_directionality_ema = ctx.trend_negative_ratio_ema = ctx.trend_positive_ratio_ema = 0.
        ctx.trend_rebound_max, ctx.trend_escape_triggered = 0., False
        ctx.trend_deltas.clear()

    def _robust_noise(self, ctx, raw):
        values = list(ctx.history_fuel)[-self.config.robust_noise_window:] + [raw]
        deltas = [values[i]-values[i-1] for i in range(1, len(values))]
        if len(deltas) < 3:
            return self.config.robust_noise_floor
        center = statistics.median(deltas)
        return max(self.config.robust_noise_floor, 1.4826*statistics.median(abs(v-center) for v in deltas))

    def _stable_values(self, values, scale):
        values = list(values)[-3:]
        if len(values) < 3:
            return False
        slope = (values[-1]-values[0])/2
        c = self.config
        return (max(values)-min(values) <= c.stable_range_pct*scale
                and statistics.pstdev(values) <= c.stable_std_pct*scale
                and abs(slope) <= c.stable_slope_pct_per_sample*scale
                and abs(values[-1]-values[0]) <= c.stable_net_change_pct*scale)

    def _mapping(self, state, ctx):
        c = self.config
        if state == "STABLE": return c.stable_q, c.stable_r
        if state in ("GRADUAL_TRACKING", "PERSISTENT_TREND_ESCAPE"): return c.gradual_q, c.gradual_r
        if state in ("PENDING_DOWNWARD", "PENDING_UPWARD", "DOWN_EXCURSION", "LOW_PLATEAU_UNCERTAIN"): return c.pending_q, c.pending_r
        if state in ("OSCILLATION", "REBOUND_RECOVERY", "U_SHAPE_CONFIRMED"): return c.recovery_q, c.recovery_r
        if state in ("DOWNWARD_CONFIRMED", "UPWARD_CONFIRMED"):
            p = max(0., min(1., ctx.transition_progress))
            return c.confirmed_q_start+p*(c.confirmed_q-c.confirmed_q_start), c.confirmed_r_start+p*(c.confirmed_r-c.confirmed_r_start)
        p = min(1., ctx.reacquisition_samples/max(1, c.reacquisition_steps-1))
        return c.reacquisition_q, c.reacquisition_r_start+p*(c.reacquisition_r_end-c.reacquisition_r_start)

    def _start_pending(self, ctx, direction, raw, timestamp):
        ctx.excursion_active, ctx.excursion_direction = True, direction
        ctx.excursion_baseline, ctx.excursion_min, ctx.excursion_max = float(ctx.stable_baseline), raw, raw
        ctx.excursion_start_time, ctx.excursion_elapsed_min = timestamp, 0.
        ctx.excursion_max_step = abs(raw-float(ctx.last_raw_fuel))
        ctx.pending_direction, ctx.pending_samples, ctx.pending_elapsed_min = direction, 1, 0.
        ctx.pending_stable_samples = 0
        ctx.max_deviation_pct = abs(raw-ctx.excursion_baseline)/max(ctx.capacity_est, 1e-9)
        ctx.rebound_ratio = ctx.pullback_ratio = 0.
        ctx.shadow_fuel = raw
        ctx.shadow_values.clear(); ctx.shadow_values.append(raw)
        ctx.low_plateau_active = False
        ctx.operational_state = "PENDING_DOWNWARD" if direction < 0 else "PENDING_UPWARD"

    def _update_trend(self, ctx, raw, timestamp, scale):
        delta = raw-float(ctx.last_raw_fuel)
        ctx.trend_deltas.append(delta)
        ds = list(ctx.trend_deltas)
        sig = [d for d in ds if abs(d) > ctx.robust_noise*.25]
        neg = sum(d < 0 for d in sig)/len(sig) if sig else 0.
        pos = sum(d > 0 for d in sig)/len(sig) if sig else 0.
        net, variation = sum(ds), sum(abs(d) for d in ds)
        direct = abs(net)/variation if variation > 1e-9 else 0.
        a = self.config.trend_ema_alpha
        ctx.trend_directionality_ema = (1-a)*ctx.trend_directionality_ema+a*direct
        ctx.trend_negative_ratio_ema = (1-a)*ctx.trend_negative_ratio_ema+a*neg
        ctx.trend_positive_ratio_ema = (1-a)*ctx.trend_positive_ratio_ema+a*pos
        ctx.trend_samples, ctx.trend_direction = len(ds), (-1 if net < 0 else 1 if net > 0 else 0)
        ctx.trend_net_change_pct = net/max(scale, 1e-9)
        if ctx.trend_start_time is None: ctx.trend_start_time = timestamp
        ctx.trend_elapsed_min = max(0., (timestamp-ctx.trend_start_time).total_seconds()/60)
        persistence = min(1., len(ds)/5)
        strength = min(1., abs(net)/max(5*ctx.robust_noise, 1e-9))
        ctx.trend_confidence = max(0., min(1., .30*direct+.25*neg+.20*strength+.15*persistence+.10*(1-min(1., ctx.rebound_ratio))))
        threshold = max(self.config.trend_net_change_pct*scale if ctx.capacity_known else self.config.unknown_absolute_safety_floor, 5*ctx.robust_noise)
        persistent = (len(ds) >= self.config.trend_min_samples and net <= -threshold
                      and direct >= self.config.trend_directionality_min and neg >= self.config.trend_negative_ratio_min
                      and pos <= self.config.trend_positive_ratio_max and ctx.rebound_ratio <= self.config.trend_rebound_max)
        ctx.trend_active = persistent
        return persistent

    def _decision(self, ctx, state, target, dev_pct, expected, observed, residual, active=True):
        ctx.operational_state = state
        q, r = self._mapping(state, ctx)
        quality = {"DOWNWARD_CONFIRMED":"DOWNWARD_SHIFT_TRACKED", "UPWARD_CONFIRMED":"UPWARD_SHIFT_TRACKED",
                   "PENDING_DOWNWARD":"PENDING_DOWNWARD_SHIFT_HELD", "PENDING_UPWARD":"PENDING_UPWARD_SHIFT_HELD"}.get(state, "SMOOTH_KALMAN")
        if state == "BASELINE_REACQUISITION":
            quality = "STABLE_LEVEL_TRACKING"
        return GuardDecision(state, target, q, r, active, quality, dev_pct, expected, observed, residual)

    def _confirm(self, ctx, direction, timestamp, target):
        ctx.operational_state = "DOWNWARD_CONFIRMED" if direction < 0 else "UPWARD_CONFIRMED"
        ctx.confirmed_direction, ctx.confirm_start_time, ctx.transition_progress = direction, timestamp, 0.
        ctx.stable_baseline, ctx.pending_direction, ctx.recovery_active = target, 0, False

    def evaluate(self, ctx: VehicleFilterContext, raw: float, speed: float, dt_minutes: float,
                 model_state: str, model_probability: float, motion: MotionEvidence,
                 window: WindowEvidence, trend: TrendEvidence, timestamp: datetime) -> GuardDecision:
        c = self.config
        scale = ctx.capacity_est if ctx.capacity_known else max(abs(float(ctx.stable_baseline)), c.minimum_capacity)
        baseline = float(ctx.stable_baseline if ctx.stable_baseline is not None else ctx.last_clean_fuel)
        deviation, dev_pct = raw-baseline, (raw-baseline)/max(scale, 1e-9)
        recent_speeds = list(ctx.history_speed)[-2:]+[speed]
        recent_speed = float(statistics.median(recent_speeds)); speed_bin = self.rate_profiles.speed_bin(recent_speed)
        observed = (raw-float(ctx.last_raw_fuel))/max(dt_minutes, .1)
        expected = self.rate_profiles.expected(ctx.vehicle_id, speed_bin, scale); residual = observed-expected
        noise = self._robust_noise(ctx, raw); ctx.robust_noise = noise
        floor = 0. if ctx.capacity_known else c.unknown_absolute_safety_floor
        motion_state = motion.state if motion.state != "UNCERTAIN" else ("LOW_MOTION" if recent_speed <= .5 else "MOVING")
        threshold = max((c.low_motion_deviation_pct if motion_state == "LOW_MOTION" else c.moving_deviation_pct)*scale if ctx.capacity_known else floor, c.deviation_noise_k*noise)
        candidate_threshold = max(c.candidate_cumulative_pct*scale if ctx.capacity_known else floor, c.candidate_noise_k*noise)
        confirm_threshold = max(c.confirm_shift_pct*scale if ctx.capacity_known else max(floor, .01*scale), c.confirm_noise_k*noise)
        strong_threshold = max(c.strong_shift_pct*scale if ctx.capacity_known else max(floor, .05*scale), 1.25*c.confirm_noise_k*noise)
        discontinuity_threshold = max(c.shift_discontinuity_pct*scale if ctx.capacity_known else floor, c.shift_discontinuity_noise_k*noise)
        innovation_threshold = max(c.innovation_capacity_pct*scale if ctx.capacity_known else floor, c.innovation_noise_k*noise)
        ctx.innovation = raw-float(ctx.kalman_x); ctx.innovation_score = abs(ctx.innovation)/max(noise, 1e-9)
        ctx.innovation_gated = abs(raw-float(ctx.last_clean_fuel)) > innovation_threshold and abs(raw-float(ctx.last_raw_fuel)) > innovation_threshold
        stable_window = self._stable_values(window.recent_values, scale)
        shadow_stable = self._stable_values(ctx.shadow_values, scale)
        persistent_down = self._update_trend(ctx, raw, timestamp, scale)

        if ctx.operational_state in ("DOWNWARD_CONFIRMED", "UPWARD_CONFIRMED"):
            elapsed = max(0., (timestamp-ctx.confirm_start_time).total_seconds()/60) if ctx.confirm_start_time else 0.
            ctx.transition_progress = min(1., elapsed/max(c.confirmed_transition_minutes, .1))
            if ctx.transition_progress < 1. or not stable_window:
                ctx.stable_baseline = float(ctx.stable_baseline)+.35*(raw-float(ctx.stable_baseline))
                return self._decision(ctx, ctx.operational_state, raw, dev_pct, expected, observed, residual, False)
            ctx.operational_state, ctx.reacquisition_start_time, ctx.reacquisition_samples = "BASELINE_REACQUISITION", timestamp, 0

        if ctx.operational_state == "BASELINE_REACQUISITION":
            if trend.robust_downtrend and recent_speed > c.stopped_speed_kmh:
                self._clear_excursion(ctx)
                ctx.trend_escape_triggered = True
                return self._decision(ctx, "PERSISTENT_TREND_ESCAPE", trend.target, dev_pct, expected, observed, residual, False)
            if stable_window:
                ctx.reacquisition_samples += 1
                return self._decision(ctx, "BASELINE_REACQUISITION", float(statistics.median(window.recent_values[-3:])), dev_pct, expected, observed, residual, False)
            return self._decision(ctx, "BASELINE_REACQUISITION", baseline, dev_pct, expected, observed, residual, True)

        if ctx.recovery_active:
            ctx.recovery_beats += 1
            ctx.recovery_elapsed_min = max(0., (timestamp-ctx.recovery_start_time).total_seconds()/60) if ctx.recovery_start_time else 0.
            recovery_downtrend = (persistent_down or trend.robust_downtrend or (
                ctx.trend_samples >= 5 and ctx.trend_net_change_pct <= -c.trend_net_change_pct
                and ctx.trend_directionality_ema >= .65
                and ctx.trend_negative_ratio_ema >= c.trend_negative_ratio_min
                and ctx.trend_positive_ratio_ema <= c.trend_positive_ratio_max
            ))
            if (ctx.recovery_direction < 0 and recovery_downtrend
                    and (recent_speed > c.stopped_speed_kmh or ctx.recovery_elapsed_min >= c.weak_confirm_elapsed_minutes)):
                ctx.recovery_active = False
                self._clear_excursion(ctx)
                ctx.trend_escape_triggered = True
                gain = .20+.65*ctx.trend_confidence
                target = float(ctx.last_clean_fuel)+gain*(raw-float(ctx.last_clean_fuel))
                return self._decision(ctx, "PERSISTENT_TREND_ESCAPE", target, dev_pct, expected, observed, residual, False)
            opposite_distance = raw-baseline if ctx.recovery_direction < 0 else baseline-raw
            strong_opposite = opposite_distance >= (c.strong_up_pct*scale if ctx.capacity_known else max(c.unknown_absolute_safety_floor, .05*scale, c.strong_up_noise_ratio*noise))
            ctx.recovery_opposite_samples = ctx.recovery_opposite_samples+1 if strong_opposite else 0
            if ctx.recovery_opposite_samples >= 2:
                direction = 1 if ctx.recovery_direction < 0 else -1
                self._confirm(ctx, direction, timestamp, raw); ctx.strong_up_confirmed = direction > 0
                return self._decision(ctx, ctx.operational_state, raw, dev_pct, expected, observed, residual, False)
            cluster = float(statistics.median(window.recent_values[-3:])) if len(window.recent_values) >= 3 else raw
            recovery_tolerance = threshold if ctx.capacity_known else max(c.unknown_absolute_safety_floor, .005*scale)
            near_old_baseline = abs(cluster-baseline) <= recovery_tolerance
            if stable_window and near_old_baseline and ctx.recovery_beats >= c.recovery_min_beats and ctx.recovery_elapsed_min >= c.recovery_min_elapsed_minutes:
                ctx.recovery_active = False; ctx.reacquisition_start_time = timestamp; ctx.reacquisition_samples = 1
                return self._decision(ctx, "BASELINE_REACQUISITION", cluster, dev_pct, expected, observed, residual, False)
            return self._decision(ctx, "REBOUND_RECOVERY", baseline, dev_pct, expected, observed, residual, True)

        if ctx.excursion_active and ctx.pending_direction:
            ctx.excursion_elapsed_min = max(0., (timestamp-ctx.excursion_start_time).total_seconds()/60)
            ctx.pending_elapsed_min = ctx.excursion_elapsed_min; ctx.pending_samples += 1
            ctx.excursion_max_step = max(ctx.excursion_max_step, abs(raw-float(ctx.last_raw_fuel)))
            ctx.excursion_min, ctx.excursion_max = min(float(ctx.excursion_min), raw), max(float(ctx.excursion_max), raw)
            ctx.shadow_values.append(raw); median3 = float(statistics.median(list(ctx.shadow_values)[-3:]))
            ctx.shadow_fuel = median3 if ctx.shadow_fuel is None else (1-c.shadow_gain)*ctx.shadow_fuel+c.shadow_gain*median3
            ctx.max_deviation_pct = max(ctx.max_deviation_pct, abs(raw-float(ctx.excursion_baseline))/max(scale, 1e-9))
            step_stable = abs(raw-float(ctx.last_raw_fuel)) <= max(c.stable_range_pct*scale, 2*noise)
            ctx.pending_stable_samples = ctx.pending_stable_samples+1 if step_stable else 0
            if ctx.pending_direction < 0:
                depth = float(ctx.excursion_baseline)-float(ctx.excursion_min)
                ctx.rebound_ratio = max(0., (raw-float(ctx.excursion_min))/max(depth, 1e-9)); cancelled = ctx.rebound_ratio >= c.rebound_cancel_ratio
            else:
                height = float(ctx.excursion_max)-float(ctx.excursion_baseline)
                ctx.pullback_ratio = max(0., (float(ctx.excursion_max)-raw)/max(height, 1e-9)); cancelled = ctx.pullback_ratio >= c.rebound_cancel_ratio
            if cancelled:
                direction = ctx.pending_direction; ctx.pending_direction = 0
                ctx.recovery_active, ctx.recovery_direction, ctx.recovery_start_time = True, direction, timestamp
                ctx.recovery_elapsed_min, ctx.recovery_beats, ctx.recovery_opposite_samples = 0., 1, 0
                return self._decision(ctx, "U_SHAPE_CONFIRMED", baseline, dev_pct, expected, observed, residual, True)

            shift = abs(raw-float(ctx.excursion_baseline)); no_pullback = max(ctx.rebound_ratio, ctx.pullback_ratio) <= .25
            has_discontinuity = ctx.excursion_max_step >= discontinuity_threshold
            strong_up = ctx.pending_direction > 0 and no_pullback and ctx.pending_samples >= 2 and shift >= (c.strong_up_pct*scale if ctx.capacity_known else max(c.unknown_absolute_safety_floor, .05*scale, c.strong_up_noise_ratio*noise))
            if strong_up:
                self._confirm(ctx, 1, timestamp, float(ctx.shadow_fuel)); ctx.strong_up_confirmed = True
                return self._decision(ctx, "UPWARD_CONFIRMED", float(ctx.shadow_fuel), dev_pct, expected, observed, residual, False)

            smooth_path = ctx.excursion_max_step < discontinuity_threshold
            trend_context = (
                ctx.trend_escape_triggered
                or any(value > c.stopped_speed_kmh for value in recent_speeds)
                or model_state == "GRADUAL_CHANGE"
            )
            if ctx.pending_direction < 0 and persistent_down and smooth_path and trend_context:
                ctx.trend_escape_triggered = True
                gain = .20+.65*ctx.trend_confidence
                target = float(ctx.last_clean_fuel)+gain*(raw-float(ctx.last_clean_fuel))
                return self._decision(ctx, "PERSISTENT_TREND_ESCAPE", target, dev_pct, expected, observed, residual, False)

            directional = window.directionality >= .75 and no_pullback
            directional_step = ctx.pending_direction * (raw-float(ctx.last_raw_fuel))
            innovation_has_persistence = not ctx.innovation_gated or ctx.pending_samples >= 3
            continued = (shift >= strong_threshold and directional and has_discontinuity and innovation_has_persistence
                         and directional_step >= discontinuity_threshold
                         and ctx.pending_samples >= c.strong_confirm_samples)
            plateau_shift = (shift >= confirm_threshold and has_discontinuity and stable_window and shadow_stable
                             and ctx.pending_stable_samples >= 2 and ctx.pending_elapsed_min >= c.weak_confirm_elapsed_minutes)
            if continued or plateau_shift:
                direction = ctx.pending_direction
                self._confirm(ctx, direction, timestamp, float(ctx.shadow_fuel))
                return self._decision(ctx, ctx.operational_state, float(ctx.shadow_fuel), dev_pct, expected, observed, residual, False)

            if ctx.pending_direction < 0 and stable_window and shadow_stable:
                ctx.low_plateau_active = True
                if ctx.excursion_elapsed_min >= c.new_baseline_accept_minutes:
                    ctx.stable_baseline = float(ctx.shadow_fuel); ctx.pending_direction = 0
                    ctx.reacquisition_start_time, ctx.reacquisition_samples = timestamp, 1
                    return self._decision(ctx, "BASELINE_REACQUISITION", float(ctx.shadow_fuel), dev_pct, expected, observed, residual, False)
                state = "LOW_PLATEAU_UNCERTAIN"
            else:
                state = "PENDING_DOWNWARD" if ctx.pending_direction < 0 else "PENDING_UPWARD"
            plausible = expected if ctx.pending_direction < 0 else max(0., expected)
            return self._decision(ctx, state, float(ctx.last_clean_fuel)+plausible*dt_minutes, dev_pct, expected, observed, residual, True)

        rate_trigger = abs(residual) > max(.0025*scale if ctx.capacity_known else 0., c.candidate_noise_k*noise/max(dt_minutes, .1), abs(expected)*c.rate_residual_sigma)
        classifier_support = model_probability >= .60 and model_state in ("DOWNWARD_SHIFT", "UPWARD_SHIFT")
        candidate = abs(deviation) >= threshold or abs(deviation) >= candidate_threshold or rate_trigger or classifier_support
        plausible_gradual_innovation = (not ctx.innovation_gated or (
            model_state == "GRADUAL_CHANGE"
            and abs(raw-float(ctx.last_raw_fuel)) <= max(.01*scale, 6*noise)
        ))
        gradual = (recent_speed > c.stopped_speed_kmh and sum(s > c.stopped_speed_kmh for s in recent_speeds) >= 2
                   and deviation <= 0 and plausible_gradual_innovation
                   and abs(observed) <= max(.003*scale, abs(expected)*5)
                   and (model_state == "GRADUAL_CHANGE" or trend.robust_downtrend or (observed < 0 and abs(deviation) < threshold)))
        if gradual: state = "GRADUAL_TRACKING"
        elif candidate:
            direction = -1 if deviation < 0 else 1
            self._start_pending(ctx, direction, raw, timestamp)
            plausible = expected if direction < 0 else max(0., expected)
            return self._decision(ctx, ctx.operational_state, float(ctx.last_clean_fuel)+plausible*dt_minutes, dev_pct, expected, observed, residual, True)
        elif model_state in ("OSCILLATION_NOISE", "SLOSHING", "SPIKE") and window.local_range >= threshold: state = "OSCILLATION"
        else: state = "STABLE"
        ctx.operational_state = state
        if state in ("STABLE", "GRADUAL_TRACKING"): self.rate_profiles.update(ctx.vehicle_id, speed_bin, observed, scale)
        q, r = self._mapping(state, ctx)
        if state == "STABLE" and motion.state == "UNCERTAIN" and recent_speed <= c.stopped_speed_kmh: q, r = c.parked_q, c.parked_r
        elif state == "STABLE" and motion.state == "UNCERTAIN": r = max(r, c.moving_r*c.gps_conflict_r_multiplier)
        elif state == "STABLE" and motion.state == "LOW_MOTION": r = min(r, c.low_motion_r)
        return GuardDecision(state, raw, q, r, False, "SMOOTH_KALMAN", dev_pct, expected, observed, residual)

    def after_update(self, ctx, clean, window, motion):
        scale = ctx.capacity_est if ctx.capacity_known else max(abs(float(ctx.stable_baseline)), self.config.minimum_capacity)
        stable = self._stable_values(window.recent_values, scale)
        if ctx.operational_state == "BASELINE_REACQUISITION" and ctx.reacquisition_samples >= self.config.reacquisition_steps:
            ctx.stable_baseline, ctx.operational_state = clean, "STABLE"
            ctx.transition_progress = 0.
            self._clear_excursion(ctx); self._reset_trend(ctx)
            return
        if ctx.operational_state in ("STABLE", "GRADUAL_TRACKING", "PERSISTENT_TREND_ESCAPE") and (motion.state == "MOVING" or stable):
            gain = self.config.baseline_moving_gain if motion.state == "MOVING" else self.config.baseline_stationary_gain
            ctx.stable_baseline = float(ctx.stable_baseline)+gain*(clean-float(ctx.stable_baseline))
