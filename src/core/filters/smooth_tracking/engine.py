"""Bo loc causal AI Smooth-Tracking cho tang tien xu ly nhien lieu.

Module chi tra muc nhien lieu sach va trang thai chat luong tin hieu. Y nghia
nghiep vu nhu nap nhien lieu hay rut trom thuoc tang phan tich phia sau.
"""
from __future__ import annotations

import json
import math
import os
import pickle
import statistics
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

import numpy as np

from .config import SmoothTrackingConfig
from .contracts import normalize_quality_flag, normalize_signal_state
from .features import CausalFeatureExtractor
from .kalman import smooth_kalman_update
from .operational_guard import OperationalGuard
from .state import MotionEvidence, VehicleFilterContext


class AISmoothTrackingFilter:
    """Dieu phoi tracking, adaptive Kalman va state causal cua tung xe."""

    def __init__(
        self,
        model_dir: str = "models/fuel_state_classifier",
        config: Optional[SmoothTrackingConfig] = None,
    ):
        self.config = config or SmoothTrackingConfig()
        self.feature_extractor = CausalFeatureExtractor(self.config)
        self.operational_guard = OperationalGuard(self.config)
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
                with open(meta_path, "r", encoding="utf-8") as f:
                    self.metadata = json.load(f)
                with open(model_path, "rb") as f:
                    self.model = pickle.load(f)
                self.feature_columns = self.metadata.get("feature_columns", [])
            except Exception:
                self.model = None
                self.metadata = None

    def get_or_create_context(
        self,
        vehicle_id: str,
        capacity_est: Optional[float] = None,
        capacity_source: str = "REQUEST",
    ) -> VehicleFilterContext:
        valid_capacity = self._valid_capacity(capacity_est)
        if vehicle_id not in self.contexts:
            self.contexts[vehicle_id] = VehicleFilterContext(
                vehicle_id=vehicle_id,
                capacity_est=float(capacity_est) if valid_capacity else None,
                capacity_mode="KNOWN" if valid_capacity else "UNKNOWN",
                capacity_source=capacity_source if valid_capacity else "NONE",
            )
        elif valid_capacity:
            context = self.contexts[vehicle_id]
            context.capacity_est = float(capacity_est)
            context.capacity_mode = "KNOWN"
            context.capacity_source = capacity_source
        return self.contexts[vehicle_id]

    @staticmethod
    def _valid_capacity(value: Optional[float]) -> bool:
        try:
            return float(value) > 0.0 and math.isfinite(float(value))
        except (TypeError, ValueError):
            return False

    @staticmethod
    def _capacity_threshold(ctx: VehicleFilterContext, floor: float, ratio: float) -> float:
        return max(floor, ratio * float(ctx.capacity_est)) if ctx.capacity_est is not None else floor

    def reset_context(self, vehicle_id: str) -> None:
        """Xoa trang context cua mot xe (khi chuyen phan doan moi)."""
        if vehicle_id in self.contexts:
            del self.contexts[vehicle_id]

    @staticmethod
    def _normalize_timestamp(value: Union[str, datetime]) -> datetime:
        if isinstance(value, datetime):
            return value
        for fmt in (
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%dT%H:%M:%S",
            "%d/%m/%Y %H:%M:%S",
            "%d/%m/%Y %H:%M",
        ):
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
        return default if not math.isfinite(number) else number

    def _result(
        self,
        clean_fuel: Optional[float],
        fuel_rate: float,
        speed: float,
        ai_state: str,
        quality_flag: str,
        motion: MotionEvidence,
        context: VehicleFilterContext,
    ) -> Dict[str, Any]:
        return {
            "clean_fuel": None if clean_fuel is None else round(clean_fuel, 2),
            "fuel_rate": round(fuel_rate, 4),
            "is_stopped": 1 if speed <= self.config.stopped_speed_kmh else 0,
            "ai_state": normalize_signal_state(ai_state),
            "signal_state": normalize_signal_state(ai_state),
            "quality_flag": normalize_quality_flag(quality_flag),
            "motion_state": motion.state,
            "motion_confidence": round(motion.confidence, 3),
            "gps_displacement_meters": round(motion.gps_displacement_meters, 2),
            "capacity_est": context.capacity_est,
            "capacity_mode": context.capacity_mode,
            "capacity_source": context.capacity_source,
            "operational_state": context.operational_state,
        }

    def _resolve_ai_state(
        self,
        ctx: VehicleFilterContext,
        raw_fuel: float,
        speed: float,
        dt_minutes: float,
        known_ai_state: Optional[str],
        motion: MotionEvidence,
    ) -> str:
        known_state = str(known_ai_state).strip()
        if known_ai_state is not None and known_state not in (
            "",
            "nan",
            "None",
            "MODEL_NOT_FOUND",
        ):
            return known_state.upper()

        if self.model is not None and self.feature_columns:
            features = self.feature_extractor.model_features(
                ctx,
                raw_fuel,
                speed,
                dt_minutes,
                motion,
            )
            try:
                vector = np.array(
                    [[features.get(column, 0.0) for column in self.feature_columns]],
                    dtype=float,
                )
                return str(self.model.predict(vector)[0])
            except Exception:
                return "STABLE_JITTER"

        delta_raw = raw_fuel - float(ctx.last_clean_fuel)
        shift_threshold = self._capacity_threshold(
            ctx, self.config.level_shift_floor, self.config.level_shift_capacity_ratio
        )
        if delta_raw >= shift_threshold:
            return "UPWARD_SHIFT"
        if delta_raw <= -shift_threshold:
            return "DOWNWARD_SHIFT"
        if speed > self.config.moving_speed_kmh and delta_raw < -0.3:
            return "GRADUAL_CHANGE"
        if abs(delta_raw) > 2.0:
            return "OSCILLATION_NOISE"
        return "STABLE_JITTER"

    def process_point(
        self,
        vehicle_id: str,
        timestamp: Union[str, datetime],
        raw_fuel: Optional[float],
        speed: float = 0.0,
        capacity_est: Optional[float] = None,
        known_ai_state: Optional[str] = None,
        lat: Optional[float] = None,
        lng: Optional[float] = None,
        capacity_source: str = "REQUEST",
    ) -> Dict[str, Any]:
        """
        Xu ly 1 diem do duy nhat theo thoi gian thuc (Causal).
        Tra ve: clean_fuel, fuel_rate (L/min), is_stopped (0/1), ai_state, quality_flag.
        """
        # 1. Chuan hoa kieu du lieu
        timestamp = self._normalize_timestamp(timestamp)
        speed = self._normalize_number(speed, 0.0)
        raw_val = self._normalize_number(raw_fuel, np.nan)
        coordinate = self.feature_extractor._valid_coordinate(lat, lng)

        ctx = self.get_or_create_context(vehicle_id, capacity_est, capacity_source)
        motion = self.feature_extractor.motion_evidence(ctx, speed, lat, lng)

        # 2. Xu ly diem khoi tao dau tien
        if ctx.last_clean_fuel is None:
            if not math.isfinite(raw_val) or raw_val <= 0.0:
                return self._result(
                    None, 0.0, speed, "UNINITIALIZED",
                    "INITIAL_INVALID_DISCARDED", motion, ctx,
                )
            init_val = raw_val

            ctx.last_clean_fuel = init_val
            ctx.kalman_x = init_val
            ctx.kalman_p = 1.0
            ctx.last_time = timestamp
            ctx.last_raw_fuel = raw_val
            ctx.history_fuel.append(init_val)
            ctx.history_time.append(timestamp)
            ctx.history_speed.append(speed)
            ctx.history_coordinates.append(coordinate)
            ctx.operational_state = "STABLE"

            return self._result(init_val, 0.0, speed, "INIT", "VALID", motion, ctx)

        # 3. Tinh khoang cach thoi gian dt
        dt_sec = (timestamp - ctx.last_time).total_seconds() if ctx.last_time else 120.0
        if dt_sec < 0:
            dt_sec = 120.0
        dt_minutes = max(dt_sec / 60.0, 0.1)

        # Mat ket noi qua 120 phut -> Reset Kalman de tranh keo lech sau khoang trong dai
        if dt_minutes > self.config.reset_gap_minutes and not math.isnan(raw_val) and raw_val > 0:
            ctx.kalman_x = raw_val
            ctx.kalman_p = 1.0
            ctx.last_clean_fuel = raw_val
            ctx.recent_upward_steps = 0
            ctx.pending_downward_count = 0
            ctx.history_fuel.clear()
            ctx.history_time.clear()
            ctx.history_speed.clear()
            ctx.history_coordinates.clear()
            self.operational_guard.reset(ctx)

        # 5. Xac dinh trang thai AI
        ai_state = normalize_signal_state(self._resolve_ai_state(
            ctx=ctx,
            raw_fuel=raw_val,
            speed=speed,
            dt_minutes=dt_minutes,
            known_ai_state=known_ai_state,
            motion=motion,
        ))

        # 6. Kiem tra va loai bo loi cam bien: NaN, rot ve 0, Spike xung
        if math.isnan(raw_val) or raw_val <= 0.0 or ai_state in ("SPIKE", "IMPULSE_NOISE"):
            ctx.last_time = timestamp
            ctx.history_time.append(timestamp)
            ctx.history_speed.append(speed)
            ctx.history_coordinates.append(coordinate)
            quality_flag = (
                "SPIKE_HELD"
                if ai_state in ("SPIKE", "IMPULSE_NOISE")
                else "ZERO_DROPOUT_HELD"
            )
            return self._result(
                float(ctx.last_clean_fuel),
                0.0,
                speed,
                ai_state,
                quality_flag,
                motion,
                ctx,
            )

        # 7. LOI THUAT TOAN THICH NGHI 2 CHE DO (BAM SAT & LAM MUOT)
        delta_from_clean = raw_val - ctx.last_clean_fuel
        level_shift_threshold = self._capacity_threshold(
            ctx, self.config.level_shift_floor, self.config.level_shift_capacity_ratio
        )

        guard_decision = self.operational_guard.evaluate(
            context=ctx,
            raw_fuel=raw_val,
            timestamp=timestamp,
            dt_minutes=dt_minutes,
            level_shift_threshold=level_shift_threshold,
        )
        if guard_decision is not None:
            clean_fuel = max(0.0, float(guard_decision.clean_fuel))
            if ctx.capacity_est is not None:
                clean_fuel = min(
                    clean_fuel,
                    ctx.capacity_est * self.config.inferred_capacity_headroom,
                )
            fuel_rate = (clean_fuel - ctx.last_clean_fuel) / dt_minutes
            ctx.last_clean_fuel = clean_fuel
            ctx.last_raw_fuel = raw_val
            ctx.last_time = timestamp
            ctx.history_fuel.append(raw_val)
            ctx.history_time.append(timestamp)
            ctx.history_speed.append(speed)
            ctx.history_coordinates.append(coordinate)
            return self._result(
                clean_fuel,
                fuel_rate,
                speed,
                ai_state,
                guard_decision.quality_flag,
                motion,
                ctx,
            )

        # Giam thoi gian bao ve sau mot dich chuyen mat bang tang.
        if ctx.recent_upward_steps > 0:
            ctx.recent_upward_steps -= 1

        quality_flag = "VALID"

        jitter = self._capacity_threshold(
            ctx, self.config.jitter_floor, self.config.jitter_capacity_ratio
        )
        window = self.feature_extractor.window_evidence(ctx, raw_val, jitter)
        trend = self.feature_extractor.trend_evidence(ctx, raw_val, speed, jitter)
        recent_four = window.recent_values[-4:]
        stable_lower_level = (
            len(recent_four) >= 4
            and max(recent_four) - min(recent_four) <= max(1.0, jitter * 0.8)
            and float(statistics.median(recent_four))
            <= ctx.last_clean_fuel - max(0.8, jitter * 0.5)
        )

        # --- A. XAC NHAN DICH CHUYEN MAT BANG TANG (causal plateau/reversal guard) ---
        is_upward_shift = False
        plateau_tol = self._capacity_threshold(ctx, 3.0, 0.006)

        # 1. Theo doi ung vien dich chuyen tang dang cho xac nhan.
        if ctx.upward_anchor is not None:
            # Neu tut ve gan anchor cu -> DAO CHIEU (Reversal): Huy bo ngay lap tuc!
            if raw_val <= ctx.upward_anchor + level_shift_threshold * 0.4:
                ctx.upward_anchor = None
                ctx.upward_samples.clear()
                ctx.rise_count = 0
                quality_flag = "UPWARD_REVERSAL_REJECTED"
            else:
                ctx.upward_samples.append(raw_val)
                # Candidate memory is bounded. Confirmation uses only recent
                # causal evidence; an old ramp/reversal must not poison the
                # candidate forever.
                ctx.upward_samples = ctx.upward_samples[-12:]
                recent_upward = ctx.upward_samples[-3:]
                spread = max(recent_upward) - min(recent_upward)
                # Tăng tiến liên tục (Ramp >= 3 nhịp) HOẶC tạo mặt bằng ổn định trên cao (Plateau >= 3 nhịp)
                is_advancing = (len(recent_upward) >= 3 and
                                all(recent_upward[k] >= recent_upward[k-1] - 1.5 for k in range(1, len(recent_upward))))
                is_stable_plateau = (len(recent_upward) >= 3 and spread <= plateau_tol * 1.5)
                # A very large rise sustained for two measurements is enough
                # for the strong causal path. Recovery from a deep U is handled
                # by OperationalGuard before this branch.
                recent_two = ctx.upward_samples[-2:]
                strong_persistent_up = (
                    len(recent_two) >= 2
                    and min(recent_two) >= ctx.upward_anchor + 2.0 * level_shift_threshold
                    and recent_two[-1] >= recent_two[-2] - plateau_tol
                )
                # Hoặc có AI xác nhận rõ ràng UPWARD_SHIFT / RISING từ nhịp 2
                ai_supported = (len(ctx.upward_samples) >= 2 and ai_state == "UPWARD_SHIFT")

                if strong_persistent_up or is_advancing or is_stable_plateau or ai_supported:
                    is_upward_shift = True
                    ctx.upward_anchor = None
                    ctx.upward_samples.clear()
                else:
                    clean_fuel = ctx.last_clean_fuel
                    quality_flag = "PENDING_UPWARD_SHIFT_HELD"
        elif delta_from_clean >= level_shift_threshold:
            if ctx.recent_upward_steps > 0:
                is_upward_shift = True
            else:
                ctx.upward_anchor = ctx.last_clean_fuel
                ctx.upward_samples = [raw_val]
                clean_fuel = ctx.last_clean_fuel
                quality_flag = "PENDING_UPWARD_SHIFT_HELD"
        elif ctx.recent_upward_steps > 0 and delta_from_clean >= 1.0:
            is_upward_shift = True
        else:
            ctx.upward_anchor = None
            ctx.upward_samples.clear()
            ctx.rise_count = 0

        if is_upward_shift:
            ctx.rise_count = 0
            ctx.upward_anchor = None
            ctx.upward_samples.clear()
            clean_fuel = raw_val
            ctx.kalman_x = raw_val
            ctx.kalman_p = 4.0
            ctx.recent_upward_steps = self.config.upward_hold_steps
            ctx.pending_downward_count = 0
            ctx.drop_count = 0
            quality_flag = "UPWARD_SHIFT_TRACKED"
            ctx.operational_state = "UPWARD_CONFIRMED"

        # Neu dang bao ve sau dich chuyen tang ma raw quay lai gan muc cu: hoan tac.
        elif ctx.recent_upward_steps > 0 and delta_from_clean <= -level_shift_threshold * 0.5:
            ctx.recent_upward_steps = 0
            ctx.kalman_x = raw_val
            clean_fuel = raw_val
            quality_flag = "UPWARD_REVERSAL_RESET"

        # --- B. XAC NHAN DICH CHUYEN MAT BANG GIAM (3 nhip causal) ---
        elif delta_from_clean <= -level_shift_threshold and quality_flag != "PENDING_UPWARD_SHIFT_HELD":
            ctx.pending_downward_count += 1
            if ctx.pending_downward_count >= self.config.downward_confirm_points:
                # Xac nhan mat bang thap moi sau 3 nhip lien tiep.
                clean_fuel = raw_val
                ctx.kalman_x = raw_val
                ctx.kalman_p = 4.0
                ctx.pending_downward_count = 0
                ctx.drop_count = 0
                quality_flag = "DOWNWARD_SHIFT_TRACKED"
                ctx.operational_state = "DOWNWARD_CONFIRMED"
            else:
                clean_fuel = ctx.last_clean_fuel
                quality_flag = "PENDING_DOWNWARD_SHIFT_HELD"

        # Muc thap khong du nguong su kien lon van duoc chap nhan khi bon
        # phep do lien tiep tao thanh mot mat bang hep. Day la thay doi muc
        # cua tin hieu, khong gan y nghia nghiep vu cho thay doi nay.
        elif stable_lower_level and quality_flag != "PENDING_UPWARD_SHIFT_HELD":
            stable_target = float(statistics.median(recent_four))
            stable_step = self.config.stable_level_gain * (stable_target - ctx.kalman_x)
            stable_step_limit = max(4.0, jitter * 3.0)
            stable_step = max(-stable_step_limit, min(stable_step_limit, stable_step))
            clean_fuel = ctx.kalman_x + stable_step
            ctx.kalman_x = clean_fuel
            ctx.kalman_p = 2.0
            ctx.pending_downward_count = 0
            ctx.drop_count = 0
            quality_flag = "STABLE_LEVEL_TRACKING"
            ctx.operational_state = "STABLE"

        # --- C. CHE DO LAM MUOT DOI XUNG & BAM DOC TIEU HAO (TREND-ADAPTIVE) ---
        elif quality_flag != "PENDING_UPWARD_SHIFT_HELD":
            ctx.pending_downward_count = 0
            clean_fuel = smooth_kalman_update(
                context=ctx,
                raw_fuel=raw_val,
                speed=speed,
                dt_minutes=dt_minutes,
                ai_state=ai_state,
                jitter=jitter,
                window=window,
                trend=trend,
                config=self.config,
                motion=motion,
            )
            quality_flag = "SMOOTH_KALMAN"
            ctx.operational_state = (
                "GRADUAL_TRACKING" if ai_state == "GRADUAL_CHANGE" else "STABLE"
            )

        if quality_flag == "PENDING_UPWARD_SHIFT_HELD":
            ctx.operational_state = "PENDING_UPWARD"
        elif quality_flag == "PENDING_DOWNWARD_SHIFT_HELD":
            ctx.operational_state = "PENDING_DOWNWARD"

        # 8. Gioi han vat ly
        clean_fuel = max(0.0, float(clean_fuel))
        if ctx.capacity_est is not None:
            clean_fuel = min(clean_fuel, ctx.capacity_est * self.config.inferred_capacity_headroom)

        fuel_rate = (clean_fuel - ctx.last_clean_fuel) / dt_minutes

        # Cap nhat bo nho cho nhip tiep theo
        ctx.last_clean_fuel = clean_fuel
        ctx.last_raw_fuel = raw_val
        ctx.last_time = timestamp
        ctx.history_fuel.append(raw_val)
        ctx.history_time.append(timestamp)
        ctx.history_speed.append(speed)
        ctx.history_coordinates.append(coordinate)

        return self._result(clean_fuel, fuel_rate, speed, ai_state, quality_flag, motion, ctx)
