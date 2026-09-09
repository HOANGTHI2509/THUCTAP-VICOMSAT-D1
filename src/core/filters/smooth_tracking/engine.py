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
from .features import CausalFeatureExtractor
from .kalman import smooth_kalman_update
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
    ) -> VehicleFilterContext:
        if vehicle_id not in self.contexts:
            cap = (
                capacity_est
                if capacity_est and capacity_est > self.config.minimum_capacity
                else self.config.default_capacity
            )
            self.contexts[vehicle_id] = VehicleFilterContext(
                vehicle_id=vehicle_id,
                capacity_est=cap,
            )
        return self.contexts[vehicle_id]

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
        return default if math.isnan(number) else number

    def _result(
        self,
        clean_fuel: float,
        fuel_rate: float,
        speed: float,
        ai_state: str,
        quality_flag: str,
        motion: MotionEvidence,
    ) -> Dict[str, Any]:
        return {
            "clean_fuel": round(clean_fuel, 2),
            "fuel_rate": round(fuel_rate, 4),
            "is_stopped": 1 if speed <= self.config.stopped_speed_kmh else 0,
            "ai_state": ai_state,
            "quality_flag": quality_flag,
            "motion_state": motion.state,
            "motion_confidence": round(motion.confidence, 3),
            "gps_displacement_meters": round(motion.gps_displacement_meters, 2),
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
        shift_threshold = max(
            self.config.level_shift_floor,
            self.config.level_shift_capacity_ratio * ctx.capacity_est,
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

        ctx = self.get_or_create_context(vehicle_id, capacity_est)
        motion = self.feature_extractor.motion_evidence(ctx, speed, lat, lng)

        # 2. Xu ly diem khoi tao dau tien
        if ctx.last_clean_fuel is None:
            if math.isnan(raw_val) or raw_val <= 0.0:
                init_val = self.config.initial_fuel_fallback
            else:
                init_val = raw_val
                if capacity_est is None:
                    ctx.capacity_est = max(
                        init_val * self.config.inferred_capacity_headroom,
                        self.config.default_capacity,
                    )

            ctx.last_clean_fuel = init_val
            ctx.kalman_x = init_val
            ctx.kalman_p = 1.0
            ctx.last_time = timestamp
            ctx.last_raw_fuel = raw_val
            ctx.history_fuel.append(init_val)
            ctx.history_time.append(timestamp)
            ctx.history_speed.append(speed)
            ctx.history_coordinates.append(coordinate)

            return self._result(init_val, 0.0, speed, "INIT", "VALID", motion)

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
            ctx.recent_refuel_steps = 0
            ctx.pending_drain_count = 0
            ctx.history_fuel.clear()
            ctx.history_time.clear()
            ctx.history_speed.clear()
            ctx.history_coordinates.clear()

        # 4. Tu dong cap nhat tran dung tich neu nhien lieu do duoc cao hon
        if not math.isnan(raw_val) and raw_val > ctx.capacity_est:
            ctx.capacity_est = float(raw_val * self.config.capacity_headroom)

        # 5. Xac dinh trang thai AI
        ai_state = self._resolve_ai_state(
            ctx=ctx,
            raw_fuel=raw_val,
            speed=speed,
            dt_minutes=dt_minutes,
            known_ai_state=known_ai_state,
            motion=motion,
        )

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
            )

        # 7. LOI THUAT TOAN THICH NGHI 2 CHE DO (BAM SAT & LAM MUOT)
        delta_from_clean = raw_val - ctx.last_clean_fuel
        refuel_threshold = max(
            self.config.level_shift_floor,
            self.config.level_shift_capacity_ratio * ctx.capacity_est,
        )

        # Giam thoi gian bao ve sau nap
        if ctx.recent_refuel_steps > 0:
            ctx.recent_refuel_steps -= 1

        quality_flag = "VALID"

        jitter = max(
            self.config.jitter_floor,
            self.config.jitter_capacity_ratio * ctx.capacity_est,
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

        # --- A. PHAT HIEN NAP NHIEN LIEU (Plateau & Reversal Guard - Triet tieu 100% Doi Ao) ---
        is_refuel = False
        plateau_tol = max(3.0, 0.006 * ctx.capacity_est)

        # 1. Kiem tra ung vien nap dang cho xac nhan (Candidate Tracking)
        if ctx.refuel_anchor is not None:
            # Neu tut ve gan anchor cu -> DAO CHIEU (Reversal): Huy bo ngay lap tuc!
            if raw_val <= ctx.refuel_anchor + refuel_threshold * 0.4:
                ctx.refuel_anchor = None
                ctx.refuel_samples.clear()
                ctx.rise_count = 0
                quality_flag = "REVERSAL_REJECTED"
            else:
                ctx.refuel_samples.append(raw_val)
                spread = max(ctx.refuel_samples) - min(ctx.refuel_samples)
                # Tăng tiến liên tục (Ramp >= 3 nhịp) HOẶC tạo mặt bằng ổn định trên cao (Plateau >= 3 nhịp)
                is_advancing = (len(ctx.refuel_samples) >= 3 and 
                                all(ctx.refuel_samples[k] >= ctx.refuel_samples[k-1] - 1.5 for k in range(1, len(ctx.refuel_samples))))
                is_stable_plateau = (len(ctx.refuel_samples) >= 3 and spread <= plateau_tol * 1.5)
                # Hoặc có AI xác nhận rõ ràng UPWARD_SHIFT / RISING từ nhịp 2
                ai_supported = (len(ctx.refuel_samples) >= 2 and ai_state in ("UPWARD_SHIFT", "RISING"))

                if is_advancing or is_stable_plateau or ai_supported:
                    is_refuel = True
                    ctx.refuel_anchor = None
                    ctx.refuel_samples.clear()
                else:
                    clean_fuel = ctx.last_clean_fuel
                    quality_flag = "PENDING_REFUEL_HELD"
        elif delta_from_clean >= refuel_threshold:
            if ctx.recent_refuel_steps > 0:
                is_refuel = True
            else:
                ctx.refuel_anchor = ctx.last_clean_fuel
                ctx.refuel_samples = [raw_val]
                clean_fuel = ctx.last_clean_fuel
                quality_flag = "PENDING_REFUEL_HELD"
        elif ctx.recent_refuel_steps > 0 and delta_from_clean >= 1.0:
            is_refuel = True
        else:
            ctx.refuel_anchor = None
            ctx.refuel_samples.clear()
            ctx.rise_count = 0

        if is_refuel:
            ctx.rise_count = 0
            ctx.refuel_anchor = None
            ctx.refuel_samples.clear()
            clean_fuel = raw_val
            ctx.kalman_x = raw_val
            ctx.kalman_p = 4.0
            ctx.recent_refuel_steps = self.config.upward_hold_steps
            ctx.pending_drain_count = 0
            ctx.drop_count = 0
            quality_flag = "REFUEL_TRACKED"

        # Neu dang trong cua so bao ve sau nap ma muc do tut sau ve lai gan muc cu -> DAO CHIEU HOAN TAC:
        elif ctx.recent_refuel_steps > 0 and delta_from_clean <= -refuel_threshold * 0.5:
            ctx.recent_refuel_steps = 0
            ctx.kalman_x = raw_val
            clean_fuel = raw_val
            quality_flag = "REFUEL_REVERSAL_RESET"

        # --- B. PHAT HIEN RUT TROM NHIEN LIEU (Drain Detection - Xac nhan 3 nhip) ---
        elif delta_from_clean <= -refuel_threshold and quality_flag != "PENDING_REFUEL_HELD":
            ctx.pending_drain_count += 1
            if ctx.pending_drain_count >= self.config.downward_confirm_points:
                # Xac nhan rut trom that sau 3 nhip lien tiep duy tri muc thap
                clean_fuel = raw_val
                ctx.kalman_x = raw_val
                ctx.kalman_p = 4.0
                ctx.pending_drain_count = 0
                ctx.drop_count = 0
                quality_flag = "DRAIN_CONFIRMED"
            else:
                clean_fuel = ctx.last_clean_fuel
                quality_flag = "PENDING_DRAIN_HELD"

        # Muc thap khong du nguong su kien lon van duoc chap nhan khi bon
        # phep do lien tiep tao thanh mot mat bang hep. Day la thay doi muc
        # cua tin hieu, khong gan nghia nap/rut cho su kien.
        elif stable_lower_level and quality_flag != "PENDING_REFUEL_HELD":
            stable_target = float(statistics.median(recent_four))
            stable_step = self.config.stable_level_gain * (stable_target - ctx.kalman_x)
            stable_step_limit = max(4.0, jitter * 3.0)
            stable_step = max(-stable_step_limit, min(stable_step_limit, stable_step))
            clean_fuel = ctx.kalman_x + stable_step
            ctx.kalman_x = clean_fuel
            ctx.kalman_p = 2.0
            ctx.pending_drain_count = 0
            ctx.drop_count = 0
            quality_flag = "STABLE_LEVEL_TRACKING"

        # --- C. CHE DO LAM MUOT DOI XUNG & BAM DOC TIEU HAO (TREND-ADAPTIVE) ---
        elif quality_flag != "PENDING_REFUEL_HELD":
            ctx.pending_drain_count = 0
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

        # 8. Gioi han vat ly
        clean_fuel = max(
            0.0,
            min(float(clean_fuel), ctx.capacity_est * self.config.inferred_capacity_headroom),
        )

        fuel_rate = (clean_fuel - ctx.last_clean_fuel) / dt_minutes

        # Cap nhat bo nho cho nhip tiep theo
        ctx.last_clean_fuel = clean_fuel
        ctx.last_raw_fuel = raw_val
        ctx.last_time = timestamp
        ctx.history_fuel.append(raw_val)
        ctx.history_time.append(timestamp)
        ctx.history_speed.append(speed)
        ctx.history_coordinates.append(coordinate)

        return self._result(clean_fuel, fuel_rate, speed, ai_state, quality_flag, motion)
