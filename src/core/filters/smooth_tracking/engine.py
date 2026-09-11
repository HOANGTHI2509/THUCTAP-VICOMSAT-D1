"""Causal purple fuel filter: classifier evidence -> OperationalGuard -> Kalman."""

from __future__ import annotations

import json
import math
import os
import pickle
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from .config import SmoothTrackingConfig
from .capacity import capacity_for_vehicle
from .contracts import normalize_quality_flag, normalize_signal_state
from .features import CausalFeatureExtractor
from .kalman import adaptive_kalman_update
from .operational_guard import OperationalGuard
from .state import MotionEvidence, VehicleFilterContext


class AISmoothTrackingFilter:
    """Keep independent causal filter and guard state for every vehicle."""

    def __init__(self, model_dir: str = "models/fuel_state_classifier", config: Optional[SmoothTrackingConfig] = None):
        self.config = config or SmoothTrackingConfig()
        self.feature_extractor = CausalFeatureExtractor(self.config)
        self.guard = OperationalGuard(self.config)
        self.model = None
        self.metadata = None
        self.feature_columns: List[str] = []
        self._load_model(model_dir)
        self.contexts: Dict[str, VehicleFilterContext] = {}

    def _load_model(self, model_dir: str) -> None:
        if not model_dir:
            return
        model_path = os.path.join(model_dir, "fuel_state_classifier.pkl")
        meta_path = os.path.join(model_dir, "metadata.json")
        if os.path.exists(model_path) and os.path.exists(meta_path):
            try:
                with open(meta_path, "r", encoding="utf-8") as stream:
                    self.metadata = json.load(stream)
                with open(model_path, "rb") as stream:
                    self.model = pickle.load(stream)
                self.feature_columns = self.metadata.get("feature_columns", [])
            except Exception:
                self.model = None
                self.metadata = None

    def get_or_create_context(self, vehicle_id: str, capacity_est: Optional[float] = None) -> VehicleFilterContext:
        if vehicle_id not in self.contexts:
            calibrated = capacity_est if capacity_est and capacity_est > self.config.minimum_capacity else capacity_for_vehicle(vehicle_id)
            known = calibrated is not None
            self.contexts[vehicle_id] = VehicleFilterContext(
                vehicle_id=vehicle_id,
                capacity_est=float(calibrated or 0.0),
                capacity_known=known,
                capacity_mode="KNOWN_CAPACITY" if known else "UNKNOWN_CAPACITY_MODE",
                capacity_warning=None if known else "CAPACITY_NOT_CALIBRATED",
            )
        return self.contexts[vehicle_id]

    def reset_context(self, vehicle_id: str) -> None:
        self.contexts.pop(vehicle_id, None)

    @staticmethod
    def _normalize_timestamp(value: Union[str, datetime]) -> datetime:
        if isinstance(value, datetime):
            return value
        for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%d/%m/%Y %H:%M:%S", "%d/%m/%Y %H:%M"):
            try:
                return datetime.strptime(str(value).split(".")[0], fmt)
            except (TypeError, ValueError):
                continue
        return datetime.now()

    @staticmethod
    def _normalize_number(value: Optional[float], default: float) -> float:
        try:
            number = float(value)
        except (TypeError, ValueError):
            return default
        return default if math.isnan(number) else number

    def _result(self, context: VehicleFilterContext, clean_fuel: float, fuel_rate: float, speed: float, ai_state: str, quality_flag: str, motion: MotionEvidence, guard_active: bool = False, operational_state: Optional[str] = None) -> Dict[str, Any]:
        def rounded(value: Optional[float], digits: int = 4):
            return None if value is None else round(float(value), digits)

        return {
            "VehicleID": context.vehicle_id, "SegmentID": context.segment_id,
            "clean_fuel": round(clean_fuel, 2), "fuel_rate": round(fuel_rate, 4),
            "is_stopped": 1 if speed <= self.config.stopped_speed_kmh else 0,
            "ai_state": normalize_signal_state(ai_state), "signal_state": normalize_signal_state(ai_state),
            "quality_flag": normalize_quality_flag(quality_flag), "motion_state": motion.state,
            "motion_confidence": round(motion.confidence, 3), "gps_displacement_meters": round(motion.gps_displacement_meters, 2),
            "OperationalState": operational_state or context.operational_state,
            "ModelState": normalize_signal_state(ai_state),
            "ModelProbabilities": json.dumps({normalize_signal_state(ai_state): round(context.classifier_probability, 6)}),
            "StableBaseline": rounded(context.stable_baseline, 3),
            "ExcursionActive": bool(context.excursion_active),
            "ExcursionDirection": context.excursion_direction,
            "ExcursionBaseline": rounded(context.excursion_baseline, 3),
            "ExcursionMin": rounded(context.excursion_min, 3), "ExcursionMax": rounded(context.excursion_max, 3),
            "ExcursionElapsedMin": rounded(context.excursion_elapsed_min, 3),
            "DeviationPct": rounded((context.last_raw_fuel - context.stable_baseline) / context.capacity_est if context.last_raw_fuel is not None and context.stable_baseline is not None else 0.0, 6),
            "ExpectedFuelRate": rounded(context.last_expected_rate, 5), "ObservedFuelRate": rounded(context.last_observed_rate, 5),
            "RateResidual": rounded(context.last_rate_residual, 5), "ReboundRatio": rounded(context.rebound_ratio, 4),
            "PullbackRatio": rounded(context.pullback_ratio, 4), "PendingSamples": context.pending_samples,
            "PendingElapsedMin": rounded(context.pending_elapsed_min, 3), "GuardActive": bool(guard_active),
            "KalmanQ": rounded(context.last_kalman_q, 4), "KalmanR": rounded(context.last_kalman_r, 4),
            "CleanFuel": round(clean_fuel, 2),
            "CapacityMode": context.capacity_mode,
            "CapacityWarning": context.capacity_warning,
            "CapacityEstimate": rounded(context.capacity_est, 3),
            "RobustNoise": rounded(context.robust_noise, 4),
            "InnovationGated": bool(context.innovation_gated),
            "Innovation": rounded(context.innovation, 4),
            "InnovationScore": rounded(context.innovation_score, 4),
            "ShadowFuel": rounded(context.shadow_fuel, 3),
            "RecoveryActive": bool(context.recovery_active),
            "TrendActive": bool(context.trend_active),
            "TrendDirection": context.trend_direction,
            "TrendConfidence": rounded(context.trend_confidence, 4),
            "TrendSamples": context.trend_samples,
            "TrendElapsedMin": rounded(context.trend_elapsed_min, 3),
            "TrendNetChangePct": rounded(context.trend_net_change_pct, 6),
            "TrendDirectionality": rounded(context.trend_directionality_ema, 4),
            "TrendEscapeTriggered": bool(context.trend_escape_triggered),
            "TransitionProgress": rounded(context.transition_progress, 4),
        }

    def _resolve_ai_state(self, context: VehicleFilterContext, raw_fuel: float, speed: float, dt_minutes: float, known_ai_state: Optional[str], motion: MotionEvidence, known_probability: Optional[float]) -> Tuple[str, float]:
        known_state = str(known_ai_state).strip()
        if known_ai_state is not None and known_state not in ("", "nan", "None", "MODEL_NOT_FOUND"):
            return known_state.upper(), self._normalize_number(known_probability, 0.0)
        if self.model is not None and self.feature_columns:
            features = self.feature_extractor.model_features(context, raw_fuel, speed, dt_minutes, motion)
            try:
                vector = np.array([[features.get(column, 0.0) for column in self.feature_columns]], dtype=float)
                prediction = str(self.model.predict(vector)[0])
                probability = float(np.max(self.model.predict_proba(vector)[0])) if hasattr(self.model, "predict_proba") else 0.0
                return prediction, probability
            except Exception:
                pass
        delta = raw_fuel - float(context.last_clean_fuel)
        shift = self.config.level_shift_capacity_ratio * context.capacity_est
        if delta >= shift:
            return "UPWARD_SHIFT", 0.5
        if delta <= -shift:
            return "DOWNWARD_SHIFT", 0.5
        if speed > self.config.stopped_speed_kmh and delta < -0.05:
            return "GRADUAL_CHANGE", 0.5
        return "STABLE_JITTER", 0.5

    def process_point(self, vehicle_id: str, timestamp: Union[str, datetime], raw_fuel: Optional[float], speed: float = 0.0, capacity_est: Optional[float] = None, known_ai_state: Optional[str] = None, lat: Optional[float] = None, lng: Optional[float] = None, segment_id: Optional[object] = None, known_ai_probability: Optional[float] = None) -> Dict[str, Any]:
        timestamp = self._normalize_timestamp(timestamp)
        speed = self._normalize_number(speed, 0.0)
        raw = self._normalize_number(raw_fuel, np.nan)
        coordinate = self.feature_extractor._valid_coordinate(lat, lng)
        context = self.get_or_create_context(vehicle_id, capacity_est)
        segment_key = None if segment_id is None else str(segment_id)
        time_gap = (timestamp - context.last_time).total_seconds() / 60.0 if context.last_time else 0.0
        must_reset = context.last_clean_fuel is not None and ((segment_key is not None and context.segment_id is not None and segment_key != context.segment_id) or time_gap > self.config.reset_gap_minutes)
        if must_reset:
            self.reset_context(vehicle_id)
            context = self.get_or_create_context(vehicle_id, capacity_est)
        context.segment_id = segment_key
        motion = self.feature_extractor.motion_evidence(context, speed, lat, lng)

        if context.last_clean_fuel is None:
            initial = self.config.initial_fuel_fallback if math.isnan(raw) or raw <= 0.0 else raw
            if context.capacity_known and initial > context.capacity_est * self.config.inferred_capacity_headroom:
                context.capacity_known = False
                context.capacity_mode = "UNKNOWN_CAPACITY_MODE"
                context.capacity_warning = "DECLARED_CAPACITY_BELOW_RAW"
            if not context.capacity_known:
                context.capacity_est = max(self.config.minimum_capacity, initial * 1.25)
            context.last_clean_fuel = initial; context.kalman_x = initial; context.kalman_p = 1.0
            context.last_time = timestamp; context.last_raw_fuel = raw
            context.history_fuel.append(initial); context.history_time.append(timestamp)
            context.history_speed.append(speed); context.history_coordinates.append(coordinate)
            self.guard.reset(context, initial)
            return self._result(context, initial, 0.0, speed, "INIT", "VALID", motion)

        dt_minutes = max(time_gap if time_gap >= 0.0 else self.config.nominal_period_minutes, 0.1)
        if context.capacity_known and raw > context.capacity_est * self.config.inferred_capacity_headroom:
            context.capacity_known = False
            context.capacity_mode = "UNKNOWN_CAPACITY_MODE"
            context.capacity_warning = "DECLARED_CAPACITY_BELOW_RAW"
            context.capacity_est = raw * 1.25
        elif not context.capacity_known and raw > context.capacity_est:
            context.capacity_est = raw * 1.25
        ai_state, probability = self._resolve_ai_state(context, raw, speed, dt_minutes, known_ai_state, motion, known_ai_probability)
        ai_state = normalize_signal_state(ai_state)
        if math.isnan(raw) or raw <= 0.0:
            context.classifier_probability = probability
            context.last_time = timestamp; context.history_time.append(timestamp)
            context.history_speed.append(speed); context.history_coordinates.append(coordinate)
            return self._result(context, float(context.last_clean_fuel), 0.0, speed, ai_state, "ZERO_DROPOUT_HELD", motion, True)

        jitter = max(self.config.jitter_floor, self.config.jitter_capacity_ratio * context.capacity_est)
        window = self.feature_extractor.window_evidence(context, raw, jitter)
        trend = self.feature_extractor.trend_evidence(context, raw, speed, jitter)
        decision = self.guard.evaluate(context, raw, speed, dt_minutes, ai_state, probability, motion, window, trend, timestamp)
        context.classifier_probability = probability
        context.last_expected_rate = decision.expected_rate; context.last_observed_rate = decision.observed_rate
        context.last_rate_residual = decision.rate_residual
        clean = adaptive_kalman_update(context, decision.target, dt_minutes, decision.q, decision.r, self.config)
        if decision.state in ("GRADUAL_TRACKING", "PERSISTENT_TREND_ESCAPE"):
            step_scale = context.capacity_est if context.capacity_known else max(abs(float(context.last_clean_fuel)), self.config.minimum_capacity)
            step_pct = self.config.gradual_max_step_pct if context.capacity_known else self.config.unknown_gradual_max_step_pct
            max_step = step_pct * step_scale
            lower = float(context.last_clean_fuel) - max_step
            upper = float(context.last_clean_fuel) + max_step
            clean = max(lower, min(upper, clean))
            context.kalman_x = clean
        elif decision.state in ("DOWNWARD_CONFIRMED", "UPWARD_CONFIRMED"):
            fraction = context.transition_progress
            step_pct = self.config.confirmed_step_start_pct + fraction * (
                self.config.confirmed_step_end_pct - self.config.confirmed_step_start_pct
            )
            if context.capacity_known:
                max_step = step_pct * context.capacity_est
            else:
                level_scale = max(abs(float(context.last_clean_fuel)), self.config.minimum_capacity)
                max_step = self.config.unknown_gradual_max_step_pct * level_scale
            lower = float(context.last_clean_fuel) - max_step
            upper = float(context.last_clean_fuel) + max_step
            clean = max(lower, min(upper, clean))
            context.kalman_x = clean
        clean = max(0.0, min(clean, context.capacity_est * self.config.inferred_capacity_headroom))
        fuel_rate = (clean - float(context.last_clean_fuel)) / dt_minutes
        context.last_clean_fuel = clean; context.last_raw_fuel = raw; context.last_time = timestamp
        context.history_fuel.append(raw); context.history_time.append(timestamp); context.history_speed.append(speed)
        context.history_coordinates.append(coordinate)
        self.guard.after_update(context, clean, window, motion)
        return self._result(context, clean, fuel_rate, speed, ai_state, decision.quality_flag, motion, decision.guard_active, decision.state)
