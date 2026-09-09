"""
AI Smooth-Tracking Realtime Filter (Bo loc thich nghi 2 che do: Bam sat & Lam muot)
Chuyen dung lam tang Tien xu ly & Lam sach tin hieu muc nhien lieu theo thoi gian thuc,
ban giao du lieu sach (clean_fuel, fuel_rate, is_stopped) cho De tai 2 (Phat hien Nap/Rut).
"""
from __future__ import annotations

import os
import json
import math
import pickle
import statistics
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd


@dataclass
class VehicleFilterContext:
    """Bo nho trang thai nhan qua theo thoi gian thuc cua tung xe."""
    vehicle_id: str
    capacity_est: float = 200.0
    r_base: float = 64.0
    
    # Trang thai Kalman 1D
    kalman_x: Optional[float] = None
    kalman_p: float = 1.0
    
    # Gia tri dau ra truoc do
    last_clean_fuel: Optional[float] = None
    last_raw_fuel: Optional[float] = None
    last_time: Optional[datetime] = None
    
    # Quan ly trang thai nap / rut / do
    recent_refuel_steps: int = 0
    pending_drain_count: int = 0
    drop_count: int = 0
    rise_count: int = 0
    refuel_anchor: Optional[float] = None
    refuel_samples: List[float] = field(default_factory=list)

    # Hang doi truot qua khu de tinh dac trung nhan qua (toi da 12 diem)
    history_fuel: deque = field(default_factory=lambda: deque(maxlen=12))
    history_time: deque = field(default_factory=lambda: deque(maxlen=12))
    history_speed: deque = field(default_factory=lambda: deque(maxlen=12))


class AISmoothTrackingFilter:
    """
    Bo loc thich nghi 2 che do (Dual-Mode Causal Filter):
      - Che do 1: BAM SAT (Tracking Mode - Khi co bien co that):
          + Nap nhien lieu (UPWARD_SHIFT hoac delta >= nguong): Bam sat ngay lap tuc
            khong bi tre 30 phut nhu Kalman truyen thong. Duy tri cua so bao ve 4 buoc de tiep tuc nap.
          + Rut trom nhien lieu: Xac nhan giam lien tuc >= 3 nhip de loai bo loi cam bien sụt ap / spike.
      - Che do 2: LAM MUOT LIEN TUC & CAN BANG TRUNG TAM (Smooth Mode - Khi chay & do):
          + Cap nhat Kalman doi xung quanh trung tam dam may du lieu (nam giua nhu Kalman chuan).
          + Khong kep cat bat doi xung: loai bo hoan toan hien tuong bi giam vao day nhieu va
            triet tieu hien tuong dong bang mot muc suot 4-5 chu ky chay.
          + Triet tieu hien tuong nhich len vai lit do song sanh: R_base tu dong co gian theo dung tich binh
            de hap thu hoan toan cac xung song sanh nho.
          + Khang nhieu gai (Spike/Dropout rejection): Giu nguyen muc hop le truoc do khi gap loi SPIKE hoac tut ve 0L.
    """

    def __init__(self, model_dir: str = "models/fuel_state_classifier"):
        self.model = None
        self.metadata = None
        self.feature_columns: List[str] = []
        self._load_model(model_dir)
        self.contexts: Dict[str, VehicleFilterContext] = {}

    def _load_model(self, model_dir: str):
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

    @staticmethod
    def calculate_r_base(capacity: float) -> float:
        """Tinh R do luong chuan hoa theo Dung tich binh."""
        cap = max(capacity, 30.0)
        return max(36.0, float((0.04 * cap) ** 2))

    def get_or_create_context(self, vehicle_id: str, capacity_est: Optional[float] = None) -> VehicleFilterContext:
        if vehicle_id not in self.contexts:
            cap = capacity_est if (capacity_est and capacity_est > 30.0) else 200.0
            r_base = self.calculate_r_base(cap)
            self.contexts[vehicle_id] = VehicleFilterContext(
                vehicle_id=vehicle_id,
                capacity_est=cap,
                r_base=r_base,
            )
        return self.contexts[vehicle_id]

    def reset_context(self, vehicle_id: str):
        """Xoa trang context cua mot xe (khi chuyen phan doan moi)."""
        if vehicle_id in self.contexts:
            del self.contexts[vehicle_id]

    def _extract_features(
        self,
        ctx: VehicleFilterContext,
        raw_fuel: float,
        speed: float,
        dt_minutes: float,
    ) -> Dict[str, float]:
        """Trich xuat 28 dac trung nhan qua dua tren hang doi qua khu."""
        valid_hist = [f for f in ctx.history_fuel if f is not None and not math.isnan(f) and f > 0]
        prev_valid = valid_hist[-1] if valid_hist else raw_fuel

        delta_fuel = raw_fuel - prev_valid
        abs_delta = abs(delta_fuel)
        cap = ctx.capacity_est
        sigma = max(0.5, 0.002 * cap)

        all_valid = valid_hist + [raw_fuel]
        rolling_std_12 = float(statistics.pstdev(all_valid)) if len(all_valid) >= 3 else 0.0

        last3 = all_valid[-3:] if all_valid else [raw_fuel]
        prev_median3 = float(statistics.median(last3))

        last5 = all_valid[-5:] if len(all_valid) >= 5 else all_valid
        local_range5 = float(max(last5) - min(last5)) if last5 else 0.0

        last7 = all_valid[-7:] if len(all_valid) >= 7 else all_valid
        local_range7 = float(max(last7) - min(last7)) if last7 else local_range5

        flat_jitter = max(0.8, 0.003 * cap)
        spike_thresh = max(3.0, 0.012 * cap)
        event_thresh = max(6.0, 0.035 * cap)

        return {
            "FuelLevel": raw_fuel,
            "FuelPct": raw_fuel / cap if cap else 0.0,
            "Speed": speed,
            "MotionSpeedKmh": speed,
            "TimeGapMinutes": dt_minutes,
            "DeltaFuel": delta_fuel,
            "DeltaPct": delta_fuel / cap if cap else 0.0,
            "AbsDeltaFuel": abs_delta,
            "DeltaOverNoise": delta_fuel / sigma if sigma else 0.0,
            "RollingStd12": rolling_std_12,
            "RollingStdPct": rolling_std_12 / cap if cap else 0.0,
            "DistanceMeters": 0.0,
            "GpsSpeedKmh": speed,
            "HasGPS": 0.0,
            "capacity_est": cap,
            "noise_sigma_liters": sigma,
            "flat_jitter_threshold": flat_jitter,
            "spike_threshold": spike_thresh,
            "event_threshold": event_thresh,
            "PrevMedian3": prev_median3,
            "FutureMedian3": prev_median3,
            "FutureMedian5": prev_median3,
            "ReturnToPrevLevel": 0.0,
            "LocalRange5": local_range5,
            "LocalRange7": local_range7,
            "PeakReversalFlag": 0,
            "ValleyReversalFlag": 0,
            "TransientScore": 0.0,
        }

    def process_point(
        self,
        vehicle_id: str,
        timestamp: Union[str, datetime],
        raw_fuel: Optional[float],
        speed: float = 0.0,
        capacity_est: Optional[float] = None,
        known_ai_state: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Xu ly 1 diem do duy nhat theo thoi gian thuc (Causal).
        Tra ve: clean_fuel, fuel_rate (L/min), is_stopped (0/1), ai_state, quality_flag.
        """
        # 1. Chuan hoa kieu du lieu
        if isinstance(timestamp, str):
            for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%d/%m/%Y %H:%M:%S", "%d/%m/%Y %H:%M"):
                try:
                    timestamp = datetime.strptime(timestamp.split(".")[0], fmt)
                    break
                except Exception:
                    pass
            if isinstance(timestamp, str):
                timestamp = datetime.now()

        speed = float(speed) if (speed is not None and not math.isnan(speed)) else 0.0
        raw_val = float(raw_fuel) if (raw_fuel is not None and not math.isnan(raw_fuel)) else np.nan

        ctx = self.get_or_create_context(vehicle_id, capacity_est)

        # 2. Xu ly diem khoi tao dau tien
        if ctx.last_clean_fuel is None:
            if math.isnan(raw_val) or raw_val <= 0.0:
                init_val = 50.0
            else:
                init_val = raw_val
                if capacity_est is None:
                    ctx.capacity_est = max(init_val * 1.05, 200.0)
                    ctx.r_base = self.calculate_r_base(ctx.capacity_est)

            ctx.last_clean_fuel = init_val
            ctx.kalman_x = init_val
            ctx.kalman_p = 1.0
            ctx.last_time = timestamp
            ctx.last_raw_fuel = raw_val
            ctx.history_fuel.append(init_val)
            ctx.history_time.append(timestamp)
            ctx.history_speed.append(speed)

            return {
                "clean_fuel": round(init_val, 2),
                "fuel_rate": 0.0,
                "is_stopped": 1 if speed <= 0.5 else 0,
                "ai_state": "INIT",
                "quality_flag": "VALID",
            }

        # 3. Tinh khoang cach thoi gian dt
        dt_sec = (timestamp - ctx.last_time).total_seconds() if ctx.last_time else 120.0
        if dt_sec < 0:
            dt_sec = 120.0
        dt_minutes = max(dt_sec / 60.0, 0.1)

        # Mat ket noi qua 120 phut -> Reset Kalman de tranh keo lech sau khoang trong dai
        if dt_minutes > 120.0 and not math.isnan(raw_val) and raw_val > 0:
            ctx.kalman_x = raw_val
            ctx.kalman_p = 1.0
            ctx.last_clean_fuel = raw_val
            ctx.recent_refuel_steps = 0
            ctx.pending_drain_count = 0
            ctx.history_fuel.clear()

        # 4. Tu dong cap nhat tran dung tich neu nhien lieu do duoc cao hon
        if not math.isnan(raw_val) and raw_val > ctx.capacity_est:
            ctx.capacity_est = float(raw_val * 1.02)
            ctx.r_base = self.calculate_r_base(ctx.capacity_est)

        # 5. Xac dinh trang thai AI
        if known_ai_state is not None and str(known_ai_state).strip() not in ("", "nan", "None", "MODEL_NOT_FOUND"):
            ai_state = str(known_ai_state).strip().upper()
        elif self.model is not None and self.feature_columns:
            feat_dict = self._extract_features(ctx, raw_val, speed, dt_minutes)
            try:
                feat_vector = np.array([[feat_dict.get(c, 0.0) for c in self.feature_columns]], dtype=float)
                ai_state = str(self.model.predict(feat_vector)[0])
            except Exception:
                ai_state = "STABLE_JITTER"
        else:
            delta_raw = raw_val - ctx.last_clean_fuel
            if delta_raw >= max(10.0, 0.04 * ctx.capacity_est):
                ai_state = "UPWARD_SHIFT"
            elif delta_raw <= -max(10.0, 0.04 * ctx.capacity_est):
                ai_state = "DOWNWARD_SHIFT"
            elif speed > 5.0 and delta_raw < -0.3:
                ai_state = "GRADUAL_CHANGE"
            elif abs(delta_raw) > 2.0:
                ai_state = "OSCILLATION_NOISE"
            else:
                ai_state = "STABLE_JITTER"

        # 6. Kiem tra va loai bo loi cam bien: NaN, rot ve 0, Spike xung
        if math.isnan(raw_val) or raw_val <= 0.0 or ai_state in ("SPIKE", "IMPULSE_NOISE"):
            ctx.last_time = timestamp
            return {
                "clean_fuel": round(ctx.last_clean_fuel, 2),
                "fuel_rate": 0.0,
                "is_stopped": 1 if speed <= 0.5 else 0,
                "ai_state": ai_state,
                "quality_flag": "SPIKE_HELD" if ai_state in ("SPIKE", "IMPULSE_NOISE") else "ZERO_DROPOUT_HELD",
            }

        # 7. LOI THUAT TOAN THICH NGHI 2 CHE DO (BAM SAT & LAM MUOT)
        delta_from_clean = raw_val - ctx.last_clean_fuel
        refuel_threshold = max(10.0, 0.04 * ctx.capacity_est)

        # Giam thoi gian bao ve sau nap
        if ctx.recent_refuel_steps > 0:
            ctx.recent_refuel_steps -= 1

        quality_flag = "VALID"

        jitter = max(0.8, 0.003 * ctx.capacity_est)
        recent_values = list(ctx.history_fuel)[-5:] + [raw_val]
        recent_changes = [
            recent_values[k] - recent_values[k - 1]
            for k in range(1, len(recent_values))
        ]
        recent_total_variation = sum(abs(change) for change in recent_changes)
        recent_net_change = (
            recent_values[-1] - recent_values[0]
            if len(recent_values) >= 2
            else 0.0
        )
        directionality = (
            abs(recent_net_change) / recent_total_variation
            if recent_total_variation > 1e-6
            else 0.0
        )
        local_std = (
            float(statistics.pstdev(recent_values))
            if len(recent_values) >= 3
            else 0.0
        )
        local_range = (
            float(max(recent_values) - min(recent_values))
            if recent_values
            else 0.0
        )
        directional_down = (
            recent_net_change <= -max(1.5, jitter * 0.75)
            and directionality >= 0.65
        )

        # Cua so dai hon de thay xu huong tieu hao bi che boi rang cua
        # ngan han. So sanh median hai nua va hoi quy tuyen tinh causal.
        trend_values = list(ctx.history_fuel)[-9:] + [raw_val]
        trend_speeds = list(ctx.history_speed)[-9:] + [speed]
        robust_downtrend = False
        robust_trend_target = raw_val
        trend_motion_supported = False
        if len(trend_values) >= 6:
            trend_mid = len(trend_values) // 2
            first_half_level = float(statistics.median(trend_values[:trend_mid]))
            second_half_level = float(statistics.median(trend_values[trend_mid:]))
            robust_level_drop = first_half_level - second_half_level
            moving_ratio = sum(value > 5.0 for value in trend_speeds) / len(trend_speeds)
            trend_motion_supported = moving_ratio >= 0.5

            mean_index = (len(trend_values) - 1) * 0.5
            mean_level = sum(trend_values) / len(trend_values)
            slope_denominator = sum(
                (index - mean_index) ** 2
                for index in range(len(trend_values))
            )
            trend_slope = (
                sum(
                    (index - mean_index) * (value - mean_level)
                    for index, value in enumerate(trend_values)
                ) / slope_denominator
                if slope_denominator > 0.0
                else 0.0
            )
            robust_trend_target = mean_level + trend_slope * (
                len(trend_values) - 1 - mean_index
            )
            robust_downtrend = (
                trend_motion_supported
                and robust_level_drop >= max(2.0, jitter * 1.25)
                and trend_slope <= -max(0.2, jitter * 0.12)
            )

        recent_four = recent_values[-4:]
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
            ctx.recent_refuel_steps = 4
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
            if ctx.pending_drain_count >= 3:
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
            stable_step = 0.65 * (stable_target - ctx.kalman_x)
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
            is_parked = (speed <= 1.0)

            # 1. Thong so Kalman can bang co ban (Doi xung hai chieu trong dai song sanh)
            if is_parked:
                R = 35.0
                Q = 0.03
            else:
                R = 45.0
                Q = 0.08

            # Nhan dien theo hinh dang cua so: dao dong manh hai chieu thi
            # tin raw it hon; song sanh nhe thi chi lam muot vua phai.
            strong_bidirectional_noise = (
                ai_state in ("OSCILLATION_NOISE", "SLOSHING")
                and local_range >= max(5.0, jitter * 3.0)
                and directionality <= 0.50
                and not robust_downtrend
            )
            mild_noise = (
                ai_state == "STABLE_JITTER"
                and local_std <= jitter
            )
            if strong_bidirectional_noise:
                R = max(R, 250.0)
                Q = min(Q, 0.01)
            elif mild_noise:
                R = min(R, 18.0)
                Q = max(Q, 0.12)

            # 2. Phat hien xu huong giam dac trung (Tieu hao xe chay hoac do no may)
            if raw_val < ctx.kalman_x - max(jitter * 0.8, 0.6):
                ctx.drop_count += 1
            else:
                ctx.drop_count = 0

            # 3. Chi tang gain bam giam khi co bang chung xu huong that.
            # Khong keo cuong buc trang thai xuong raw: thao tac do lam duong loc
            # roi vao day chu U truoc khi biet raw co quay lai hay khong.
            downward_trend_supported = (
                ai_state in ("GRADUAL_CHANGE", "DOWNWARD_SHIFT", "DRAIN")
                or (not is_parked and ctx.drop_count >= 3 and directional_down)
                or robust_downtrend
            )

            # 4. Thich nghi do doc: khong dung rieng chuoi diem thap khi AI
            # dang bao OSCILLATION_NOISE de ket luan day la xu huong giam.
            if ctx.drop_count >= 2 and downward_trend_supported:
                slope_lag = min(4.0, (ctx.kalman_x - raw_val) / max(jitter * 0.5, 0.4))
                R = max(10.0, R / slope_lag)
                Q = max(Q, 0.15 * slope_lag)
                if directional_down:
                    R = min(R, 8.0)
                    Q = max(Q, 1.0)
                if robust_downtrend:
                    R = min(R, 6.0)
                    Q = max(Q, 1.5)

            # 5. Cap nhat Kalman doi xung hai chieu
            dt_ratio = max(0.1, min(10.0, dt_minutes / 2.0))
            P_pred = ctx.kalman_p + Q * dt_ratio
            K = P_pred / (P_pred + R)
            measurement_target = (
                min(raw_val, robust_trend_target)
                if robust_downtrend
                else raw_val
            )
            diff = measurement_target - ctx.kalman_x

            # Khi xe dang chay, chi cho phep hoi tang rat cham trong nhanh
            # lam muot. Tang muc lon van phai qua Candidate Tracking o tren.
            if not is_parked and diff > 0:
                max_step_up = max(0.4, jitter * 0.25)
                diff_compressed = min(max_step_up, diff)
            elif (not is_parked or robust_downtrend) and diff < 0:
                if downward_trend_supported and (directional_down or robust_downtrend):
                    diff_compressed = diff
                else:
                    # Chua du tinh dinh huong: chi cho phep giam tung buoc nho.
                    max_step_burn = max(1.5, jitter * 1.2)
                    diff_compressed = max(-max_step_burn, diff)
            else:
                shrink_threshold = max(1.5, jitter * 1.5)
                diff_compressed = shrink_threshold * math.tanh(diff / shrink_threshold)

            kalman_step = K * diff_compressed
            if (not is_parked and directional_down) or robust_downtrend:
                kalman_step = max(-max(2.5, jitter * 1.8), kalman_step)
            ctx.kalman_x = ctx.kalman_x + kalman_step
            ctx.kalman_p = (1.0 - K) * P_pred

            clean_fuel = ctx.kalman_x

            quality_flag = "SMOOTH_KALMAN"

        # 8. Gioi han vat ly
        clean_fuel = max(0.0, min(float(clean_fuel), ctx.capacity_est * 1.05))

        fuel_rate = (clean_fuel - ctx.last_clean_fuel) / dt_minutes
        is_stopped = 1 if speed <= 0.5 else 0

        # Cap nhat bo nho cho nhip tiep theo
        ctx.last_clean_fuel = clean_fuel
        ctx.last_raw_fuel = raw_val
        ctx.last_time = timestamp
        ctx.history_fuel.append(raw_val)
        ctx.history_time.append(timestamp)
        ctx.history_speed.append(speed)

        return {
            "clean_fuel": round(clean_fuel, 2),
            "fuel_rate": round(fuel_rate, 4),
            "is_stopped": is_stopped,
            "ai_state": ai_state,
            "quality_flag": quality_flag,
        }


# Module-level cached engine
_CACHED_FILTER_ENGINE = None

def get_shared_smooth_engine(model_dir: str = "models/fuel_state_classifier") -> AISmoothTrackingFilter:
    global _CACHED_FILTER_ENGINE
    if _CACHED_FILTER_ENGINE is None:
        _CACHED_FILTER_ENGINE = AISmoothTrackingFilter(model_dir=model_dir)
    return _CACHED_FILTER_ENGINE


def filter_smooth_tracking_dataframe(
    df: pd.DataFrame,
    vehicle_id: Optional[str] = None,
    capacity_est: Optional[float] = None,
    filter_engine: Optional[AISmoothTrackingFilter] = None,
    model_dir: str = "models/fuel_state_classifier",
) -> pd.DataFrame:
    """
    Ham tien ich cho Dashboard hoac xu ly Batch theo DataFrame / File.
    Dau ra san sang chuyen giao cho De tai 2:
      - CleanFuel_SmoothTracking: Muc nhien lieu da loc bam sat & lam muot.
      - FuelRate_SmoothTracking: Toc do bien thien (L/min) de phan lop Nap/Rut khi dung.
      - IsStopped: 1 neu xe dung (Speed <= 0.5), 0 neu xe chay.
      - QualityFlag_SmoothTracking: Co chat luong danh dau trang thai loc.
    """
    df_out = df.copy()
    if vehicle_id is None:
        if "VehicleID" in df_out.columns and len(df_out) > 0:
            vehicle_id = str(df_out["VehicleID"].iloc[0])
        else:
            vehicle_id = "Car_Default"

    if filter_engine is None:
        if "AI_State" in df_out.columns:
            filter_engine = AISmoothTrackingFilter(model_dir="")
        else:
            filter_engine = get_shared_smooth_engine(model_dir=model_dir)

    # Reset context cho phan doan / lan chay moi
    filter_engine.reset_context(vehicle_id)

    clean_fuels = []
    fuel_rates = []
    is_stopped_list = []
    ai_states = []
    quality_flags = []

    time_col = "FuelTime" if "FuelTime" in df_out.columns else df_out.columns[0]
    fuel_col = "FuelLevel" if "FuelLevel" in df_out.columns else "Nhiên liệu"
    speed_col = "Speed" if "Speed" in df_out.columns else ("Vận tốc" if "Vận tốc" in df_out.columns else None)
    has_precomputed_ai = "AI_State" in df_out.columns

    for row in df_out.itertuples():
        t = getattr(row, time_col, None)
        f = getattr(row, fuel_col, np.nan)
        s = getattr(row, speed_col, 0.0) if speed_col else 0.0
        known_ai = getattr(row, "AI_State", None) if has_precomputed_ai else None

        res = filter_engine.process_point(
            vehicle_id=vehicle_id,
            timestamp=t,
            raw_fuel=f,
            speed=s,
            capacity_est=capacity_est,
            known_ai_state=known_ai,
        )
        clean_fuels.append(res["clean_fuel"])
        fuel_rates.append(res["fuel_rate"])
        is_stopped_list.append(res["is_stopped"])
        ai_states.append(res["ai_state"])
        quality_flags.append(res["quality_flag"])

    df_out["CleanFuel_SmoothTracking"] = clean_fuels
    df_out["FuelRate_SmoothTracking"] = fuel_rates
    df_out["IsStopped"] = is_stopped_list
    df_out["AI_State_SmoothTracking"] = ai_states
    df_out["QualityFlag_SmoothTracking"] = quality_flags

    return df_out
