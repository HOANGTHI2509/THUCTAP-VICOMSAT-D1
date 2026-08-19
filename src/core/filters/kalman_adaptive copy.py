import glob
import os
import sys
from dataclasses import dataclass
from typing import Optional
import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")


@dataclass
class VehicleFuelProfile:
    vehicle_id: str
    capacity_est: float
    noise_sigma_liters: float
    noise_sigma_pct: float
    sample_gap_minutes: float
    flat_jitter_threshold: float
    spike_threshold: float
    event_threshold: float
    max_drop_rate_lpm: float


def _safe_float(value, fallback: float) -> float:
    value = pd.to_numeric(value, errors="coerce")
    if pd.isna(value):
        return float(fallback)
    return float(value)


def estimate_vehicle_profile(df: pd.DataFrame, vehicle_id: str = "") -> VehicleFuelProfile:
    fuel = pd.to_numeric(df.get("FuelLevel"), errors="coerce")
    fuel = fuel[(fuel > 0) & fuel.notna()]
    if fuel.empty:
        capacity_est = 200.0
        noise_sigma = 1.0
    else:
        capacity_est = float(fuel.quantile(0.995))
        if pd.isna(capacity_est) or capacity_est < 10.0:
            capacity_est = float(max(fuel.quantile(0.99), 50.0))

        diff_abs = fuel.diff().abs().dropna()
        if diff_abs.empty:
            noise_sigma = max(0.5, 0.002 * capacity_est)
        else:
            small_motion = diff_abs[diff_abs <= diff_abs.quantile(0.60)]
            if small_motion.empty:
                small_motion = diff_abs
            noise_sigma = float(1.4826 * small_motion.median())
            if pd.isna(noise_sigma) or noise_sigma <= 0:
                noise_sigma = max(0.5, 0.002 * capacity_est)

    gaps = pd.to_numeric(df.get("TimeGapMinutes", pd.Series(dtype=float)), errors="coerce")
    valid_gaps = gaps[(gaps > 0) & gaps.notna()]
    sample_gap = float(valid_gaps.median()) if not valid_gaps.empty else 5.0
    if pd.isna(sample_gap) or sample_gap <= 0:
        sample_gap = 5.0

    noise_sigma = max(float(noise_sigma), 0.0015 * capacity_est, 0.15)
    flat_jitter = max(2.5 * noise_sigma, 0.003 * capacity_est, 0.8)
    spike_threshold = max(4.5 * noise_sigma, 0.012 * capacity_est, flat_jitter * 1.8)
    event_threshold = max(8.0 * noise_sigma, 0.035 * capacity_est, spike_threshold * 1.8)
    max_drop_rate_lpm = max(0.2, 0.0015 * capacity_est / max(sample_gap, 1.0))

    return VehicleFuelProfile(
        vehicle_id=str(vehicle_id),
        capacity_est=round(capacity_est, 3),
        noise_sigma_liters=round(noise_sigma, 3),
        noise_sigma_pct=round(noise_sigma / capacity_est, 6) if capacity_est > 0 else 0.0,
        sample_gap_minutes=round(sample_gap, 3),
        flat_jitter_threshold=round(flat_jitter, 3),
        spike_threshold=round(spike_threshold, 3),
        event_threshold=round(event_threshold, 3),
        max_drop_rate_lpm=round(max_drop_rate_lpm, 4),
    )


def profile_from_series(row: pd.Series, fallback: VehicleFuelProfile) -> VehicleFuelProfile:
    capacity = _safe_float(row.get("capacity_est", row.get("Dung tích (L)", row.get("Dung tÃ­ch (L)"))), fallback.capacity_est)
    noise_sigma = _safe_float(
        row.get("noise_sigma_liters", row.get("Nhiễu trung bình (L)", row.get("Nhiá»…u trung bÃ¬nh (L)"))),
        fallback.noise_sigma_liters,
    )
    sample_gap = _safe_float(row.get("sample_gap_minutes"), fallback.sample_gap_minutes)

    flat_jitter = _safe_float(row.get("flat_jitter_threshold"), max(2.5 * noise_sigma, 0.003 * capacity, 0.8))
    spike_threshold = _safe_float(row.get("spike_threshold"), max(4.5 * noise_sigma, 0.012 * capacity, flat_jitter * 1.8))
    event_threshold = _safe_float(row.get("event_threshold"), max(8.0 * noise_sigma, 0.035 * capacity, spike_threshold * 1.8))
    max_drop_rate_lpm = _safe_float(row.get("max_drop_rate_lpm"), max(0.2, 0.0015 * capacity / max(sample_gap, 1.0)))

    return VehicleFuelProfile(
        vehicle_id=str(row.get("VehicleID", row.get("Mã Xe", row.get("MÃ£ Xe", fallback.vehicle_id)))),
        capacity_est=capacity,
        noise_sigma_liters=noise_sigma,
        noise_sigma_pct=noise_sigma / capacity if capacity > 0 else 0.0,
        sample_gap_minutes=sample_gap,
        flat_jitter_threshold=flat_jitter,
        spike_threshold=spike_threshold,
        event_threshold=event_threshold,
        max_drop_rate_lpm=max_drop_rate_lpm,
    )


def apply_profile_noise_gate(
    df: pd.DataFrame,
    profile: VehicleFuelProfile,
    source_col: str = "CleanedFuel",
    output_col: str = "ProfileCleanFuel",
    lookback: int = 7,
    lookahead: int = 3,
) -> pd.DataFrame:
    if source_col not in df.columns:
        source_col = "FuelLevel"

    result = df.sort_values("FuelTime").copy()
    raw = pd.to_numeric(result[source_col], errors="coerce").to_numpy(dtype=float)
    speed = pd.to_numeric(result.get("Speed", 0.0), errors="coerce").fillna(0.0).to_numpy(dtype=float)
    times = pd.to_datetime(result.get("FuelTime"), errors="coerce")
    cleaned = np.full(len(result), np.nan, dtype=float)
    flags = np.array(["NORMAL"] * len(result), dtype=object)

    accepted: list[float] = []
    last_time = None

    for i, z in enumerate(raw):
        if pd.isna(z) or z <= 0:
            flags[i] = "INVALID"
            continue

        previous_clean = accepted[-1] if accepted else float(z)
        baseline = float(np.median(accepted[-lookback:])) if accepted else float(z)
        future = raw[i + 1 : i + 1 + lookahead]
        future = future[~np.isnan(future)]
        future_median = float(np.median(future)) if len(future) else float(z)

        current_time = times.iloc[i] if hasattr(times, "iloc") else None
        if last_time is not None and pd.notna(current_time):
            gap_minutes = max((current_time - last_time).total_seconds() / 60.0, profile.sample_gap_minutes)
        else:
            gap_minutes = profile.sample_gap_minutes

        delta_base = float(z - baseline)
        delta_future_to_base = abs(future_median - baseline)
        delta_future_to_z = abs(future_median - z)
        max_physical_drop = max(profile.max_drop_rate_lpm * gap_minutes, profile.flat_jitter_threshold)
        current_speed = float(speed[i]) if i < len(speed) else 0.0
        recent_speed = speed[max(0, i - 2) : i + 1]
        future_speed = speed[i + 1 : i + 1 + lookahead]
        speed_context = np.concatenate([recent_speed, future_speed]) if len(future_speed) else recent_speed
        recent_moving = bool(len(recent_speed) and np.nanmax(recent_speed) > 3.0)
        moving_context = bool(len(speed_context) and np.nanmax(speed_context) > 3.0)

        returns_to_baseline = delta_future_to_base <= profile.flat_jitter_threshold
        holds_new_level = delta_future_to_z <= max(profile.flat_jitter_threshold, 0.35 * abs(delta_base))
        is_large_change = abs(delta_base) >= profile.spike_threshold
        is_refuel_candidate = delta_base > 0 and not recent_moving
        is_drop_candidate = delta_base < 0
        is_event = (
            abs(delta_base) >= profile.event_threshold
            and holds_new_level
            and (is_drop_candidate or is_refuel_candidate)
        )
        is_moving_up_spike = (
            moving_context
            and not holds_new_level
            and delta_base > profile.flat_jitter_threshold
        )
        is_short_spike = (is_large_change and returns_to_baseline or is_moving_up_spike) and not is_event
        is_unphysical_drop = (
            delta_base < -profile.event_threshold
            and abs(delta_base) > max(3.0 * max_physical_drop, profile.event_threshold)
            and returns_to_baseline
            and not is_event
        )
        is_flat_jitter = abs(float(z) - previous_clean) <= profile.flat_jitter_threshold

        if is_short_spike or is_unphysical_drop:
            cleaned[i] = previous_clean
            flags[i] = "SPIKE_SUPPRESSED" if is_short_spike else "DROP_RATE_SUPPRESSED"
        elif is_flat_jitter:
            cleaned[i] = float(z) if not accepted else previous_clean + 0.35 * (float(z) - previous_clean)
            flags[i] = "FLAT_JITTER"
        else:
            cleaned[i] = float(z)
            flags[i] = "EVENT" if is_event else "TREND"

        accepted.append(float(cleaned[i]))
        if pd.notna(current_time):
            last_time = current_time

    result[output_col] = cleaned
    result["ProfileNoiseFlag"] = flags
    result["SignalMode"] = flags
    return result


def classify_signal_modes(
    df: pd.DataFrame,
    profile: VehicleFuelProfile,
    source_col: str = "FuelLevel",
    output_col: str = "ProfileCleanFuel",
    mode_col: str = "SignalMode",
    lookback: int = 7,
    lookahead: int = 5,
) -> pd.DataFrame:
    if source_col not in df.columns:
        source_col = "FuelLevel"

    result = df.sort_values("FuelTime").copy()
    raw = pd.to_numeric(result[source_col], errors="coerce").to_numpy(dtype=float)
    speed = pd.to_numeric(result.get("Speed", 0.0), errors="coerce").fillna(0.0).to_numpy(dtype=float)
    rolling_std = pd.to_numeric(result.get("RollingStd", 0.0), errors="coerce").fillna(0.0).to_numpy(dtype=float)
    delta_feature = pd.to_numeric(result.get("DeltaFuel", np.nan), errors="coerce").to_numpy(dtype=float)
    quality_reason = result.get("QualityReason", pd.Series([""] * len(result), index=result.index)).astype(str).to_numpy()

    cleaned = np.full(len(result), np.nan, dtype=float)
    modes = np.array(["NORMAL"] * len(result), dtype=object)
    accepted: list[float] = []
    raw_prev = np.nan

    for i, z in enumerate(raw):
        if pd.isna(z) or z <= 0:
            modes[i] = "INVALID"
            raw_prev = z
            continue

        previous_clean = accepted[-1] if accepted else float(z)
        baseline = float(np.median(accepted[-lookback:])) if accepted else float(z)
        future = raw[i + 1 : i + 1 + lookahead]
        future = future[~np.isnan(future)]
        future_median = float(np.median(future)) if len(future) else float(z)

        delta_base = float(z - baseline)
        delta_raw = float(delta_feature[i]) if i < len(delta_feature) and not pd.isna(delta_feature[i]) else (
            0.0 if pd.isna(raw_prev) else float(z - raw_prev)
        )
        delta_future_to_base = abs(future_median - baseline)
        delta_future_to_z = abs(future_median - z)

        recent_speed = speed[max(0, i - 2) : i + 1]
        future_speed = speed[i + 1 : i + 1 + lookahead]
        recent_moving = bool(len(recent_speed) and np.nanmax(recent_speed) > 3.0)
        moving_context = bool(
            (len(recent_speed) and np.nanmax(recent_speed) > 3.0)
            or (len(future_speed) and np.nanmax(future_speed) > 3.0)
        )

        returns_to_baseline = delta_future_to_base <= profile.flat_jitter_threshold
        holds_new_level = delta_future_to_z <= max(profile.flat_jitter_threshold, 0.35 * abs(delta_base))
        large_by_profile = abs(delta_base) >= profile.spike_threshold or abs(delta_raw) >= profile.spike_threshold
        event_by_profile = abs(delta_base) >= profile.event_threshold or abs(delta_raw) >= profile.event_threshold
        high_local_noise = rolling_std[i] > max(profile.flat_jitter_threshold, profile.noise_sigma_liters * 3.0)
        has_large_delta_flag = "LARGE_DELTA" in quality_reason[i].upper()

        if delta_base > 0 and event_by_profile and holds_new_level and not recent_moving:
            mode = "REFUEL"
            clean_value = float(z)
        elif delta_base < 0 and event_by_profile and holds_new_level and not returns_to_baseline:
            mode = "DROP_EVENT"
            clean_value = float(z)
        elif delta_base > profile.flat_jitter_threshold and moving_context and not holds_new_level:
            mode = "SPIKE_UP"
            clean_value = previous_clean
        elif large_by_profile and returns_to_baseline:
            mode = "SPIKE_UP" if delta_base > 0 else "SPIKE_DOWN"
            clean_value = previous_clean
        elif has_large_delta_flag and high_local_noise and not holds_new_level:
            mode = "SPIKE_UP" if delta_base > 0 else "SPIKE_DOWN"
            clean_value = previous_clean
        elif abs(float(z) - previous_clean) <= profile.flat_jitter_threshold:
            mode = "STABLE" if rolling_std[i] <= profile.noise_sigma_liters * 2.0 else "JITTER"
            alpha = 0.15 if mode == "JITTER" else 0.35
            clean_value = previous_clean + alpha * (float(z) - previous_clean)
        elif delta_base < -profile.flat_jitter_threshold:
            mode = "TREND_DOWN"
            clean_value = float(z)
        else:
            mode = "NORMAL"
            clean_value = float(z)

        cleaned[i] = clean_value
        modes[i] = mode
        accepted.append(float(clean_value))
        raw_prev = z

    result[output_col] = cleaned
    result[mode_col] = modes
    result["ProfileNoiseFlag"] = modes
    return result


class BoLocKalmanThichNghi1D:
    def __init__(
        self,
        trang_thai_ban_dau: float,
        capacity: float = 200.0,
        sai_so_uoc_luong_ban_dau: float = 4.0,
        nhieu_qua_trinh: float = 0.1,
        r_co_ban: float = 0.0,
        nhip_cho_xac_nhan: int = 3,
        nguong_bat_nhay: Optional[float] = None,
        nguong_bat_nhay_co_ban: Optional[float] = None,
        nguong_toi_da: Optional[float] = None,
        r_nhieu_dot_bien: Optional[float] = None,
        muc_tieu_thu_100km: Optional[float] = None,
    ) -> None:
        self.x = float(trang_thai_ban_dau)
        self.P = float(sai_so_uoc_luong_ban_dau)
        self.Q = float(nhieu_qua_trinh)
        self.capacity = float(capacity)

        if r_co_ban <= 0:
            self.r_base = max(64.0, (0.04 * self.capacity) ** 2)
        else:
            self.r_base = float(r_co_ban)

        threshold_candidates = [
            max(3.0, 0.02 * self.capacity),
            float(nguong_bat_nhay) if nguong_bat_nhay is not None else 0.0,
            float(nguong_bat_nhay_co_ban) if nguong_bat_nhay_co_ban is not None else 0.0,
        ]
        self.threshold = max(threshold_candidates)
        self.max_threshold = float(nguong_toi_da) if nguong_toi_da is not None else max(25.0, 0.125 * self.capacity)
        self.r_spike = float(r_nhieu_dot_bien) if r_nhieu_dot_bien is not None else self.r_base * 8.0
        self.consumption_hint = float(muc_tieu_thu_100km) if muc_tieu_thu_100km is not None else None

        self.persistence_required = max(1, int(nhip_cho_xac_nhan))
        self.stable_epsilon = max(0.5, 0.003 * self.capacity)
        self.refuel_epsilon = max(2.0, 0.01 * self.capacity)
        self.refuel_snap_threshold = max(12.0, 0.04 * self.capacity)
        self.refuel_confirm_threshold = max(6.0, 0.02 * self.capacity)
        self.dropout_floor = max(3.0, 0.03 * self.capacity)

        self.z_prev = self.x
        self.last_residual_sign = 0
        self.so_nhip_tang_manh = 0
        self.so_nhip_giam_manh = 0
        self.so_nhip_on_dinh = 0
        self.so_nhip_zigzag = 0
        self.so_nhip_giam_on_dinh = 0
        self.so_nhip_giam_lien_tuc = 0

        self.ema_z = self.x
        self.prev_ema_z = self.x
        self.ema_v = 0.0
        self.z_buffer = []

    def cap_nhat(
        self,
        gia_tri_do: float,
        ty_le_dt: float = 1.0,
        trang_thai_chuyen_dong: int = 1,
        gia_toc: float = 0.0,
        rolling_std: float = 0.0,
        van_toc: float = 0.0,
        signal_mode: str = "NORMAL",
    ) -> float:
        dt_scale = max(float(ty_le_dt), 0.1)
        rolling_std = 0.0 if pd.isna(rolling_std) else float(rolling_std)
        van_toc = 0.0 if pd.isna(van_toc) else float(van_toc)
        gia_toc = 0.0 if pd.isna(gia_toc) else float(gia_toc)
        signal_mode = str(signal_mode or "NORMAL").upper()

        # BỨC TƯỜNG BẢO VỆ MEDIAN: Chặn đứng nhiễu 1 nhịp (200 -> 2 -> 200)
        self.z_buffer.append(float(gia_tri_do))
        if len(self.z_buffer) > 3:
            self.z_buffer.pop(0)
            
        if len(self.z_buffer) == 3:
            z = sorted(self.z_buffer)[1]
        else:
            z = float(gia_tri_do)

        if self.z_prev is None:
            self.z_prev = z
            self.ema_z = z
            self.prev_ema_z = z
            self.ema_v = 0.0

        delta_z = z - self.z_prev
        self.z_prev = z

        # STABILITY TRACKER: Nhận diện khi mặt xăng tĩnh lặng tuyệt đối (xe đỗ hoặc đi đường siêu bằng phẳng)
        if abs(delta_z) <= 0.3:
            self.so_nhip_on_dinh = getattr(self, 'so_nhip_on_dinh', 0) + 1
        else:
            self.so_nhip_on_dinh = 0

        if delta_z < -max(0.5, self.stable_epsilon * 0.5):
            self.so_nhip_giam_lien_tuc = getattr(self, 'so_nhip_giam_lien_tuc', 0) + 1
        elif delta_z > max(0.5, self.stable_epsilon * 0.5):
            self.so_nhip_giam_lien_tuc = 0

        # 1. EMA lọc nhiễu vị trí trước
        self.ema_z = 0.9 * self.ema_z + 0.1 * z
        raw_v = self.ema_z - self.prev_ema_z
        self.prev_ema_z = self.ema_z

        # 2. EMA lọc nhiễu vận tốc
        self.ema_v = 0.9 * self.ema_v + 0.1 * raw_v
        # Chỉ cho phép học thói quen GIẢM (tiêu thụ nhiên liệu)
        self.ema_v = min(0.0, self.ema_v)

        x_du_doan = self.x
        phan_du_x = z - x_du_doan

        # FAST-JUMP TRACKER (Chống nhiễu đột biến 1 nhịp như V-shape)
        # Giới hạn ngưỡng động (cap) để không bị "nổ" tung khi chính dữ liệu đang nhảy bơm xăng
        nguong_toi_da = max(25.0, 0.05 * self.capacity)
        dynamic_threshold = min(nguong_toi_da, max(self.threshold, 2.0 * rolling_std))
        
        dot_bien_tang = getattr(self, 'so_nhip_dot_bien_tang', 0)
        dot_bien_giam = getattr(self, 'so_nhip_dot_bien_giam', 0)

        if phan_du_x > dynamic_threshold:
            self.so_nhip_dot_bien_tang = dot_bien_tang + 1
            self.so_nhip_dot_bien_giam = 0
        elif phan_du_x < -dynamic_threshold:
            self.so_nhip_dot_bien_giam = dot_bien_giam + 1
            self.so_nhip_dot_bien_tang = 0
        else:
            self.so_nhip_dot_bien_tang = 0
            self.so_nhip_dot_bien_giam = 0

        R_thich_nghi = self.r_base
        Q_thich_nghi = self.Q * max(dt_scale, 0.1) * (1.0 + van_toc / 40.0)

        if signal_mode in {"JITTER", "STABLE"}:
            R_thich_nghi *= 4.0 if signal_mode == "JITTER" else 2.0
            Q_thich_nghi *= 0.35
        elif signal_mode == "TREND_DOWN":
            R_thich_nghi = max(1.0, self.r_base * 0.22)
            Q_thich_nghi = max(Q_thich_nghi * 6.0, self.Q * 3.0)
        elif signal_mode in {"REFUEL", "DROP_EVENT"}:
            R_thich_nghi = max(1.0, self.r_base * 0.10)
            Q_thich_nghi *= 10.0
        elif signal_mode.startswith("SPIKE"):
            R_thich_nghi *= 10.0
            Q_thich_nghi *= 0.2

        if rolling_std > self.stable_epsilon:
            noise_ratio = min(6.0, rolling_std / max(self.stable_epsilon, 0.1))
            R_thich_nghi *= 1.0 + noise_ratio ** 2
            Q_thich_nghi *= 0.5

        if van_toc > 3.0 or trang_thai_chuyen_dong == 1:
            R_thich_nghi *= 3.0
            Q_thich_nghi *= 0.5

        trend_giam_that = (
            getattr(self, 'so_nhip_giam_lien_tuc', 0) >= 2
            and z < self.x - self.stable_epsilon
            and not (z <= self.dropout_floor and self.x > max(30.0, 0.25 * self.capacity))
        )
        if trend_giam_that:
            R_thich_nghi = max(1.0, self.r_base * 0.18)
            Q_thich_nghi = max(Q_thich_nghi * 8.0, self.Q * 4.0)

        # Nếu cảm biến siêu ổn định trong 3 nhịp liên tiếp (ví dụ 203, 203, 203)
        # Ta tin tưởng cảm biến 100%, giảm R để hội tụ nhanh và xóa bỏ quán tính ảo (overshoot)
        if getattr(self, 'so_nhip_on_dinh', 0) >= 3:
            R_thich_nghi = max(1.0, self.r_base * 0.12)
            Q_thich_nghi = max(Q_thich_nghi, self.Q)
            self.ema_v = 0.0

        # Nhờ bức tường Median bảo vệ, ta không sợ nhiễu 1 nhịp nữa!
        # Bơm xăng (nhảy lên) -> Bám sát ngay tức thì!
        nhip_xac_nhan_tang = 1 
        # Rút trộm (nhảy xuống) vẫn nên chờ theo cấu hình để chống rớt cảm biến nhiều nhịp
        nhip_xac_nhan_giam = max(2, self.persistence_required)

        event_dang_xay_ra = False

        # Giải quyết TỨC THÌ các sự kiện bơm/rút
        if signal_mode in {"REFUEL", "DROP_EVENT"} or getattr(self, 'so_nhip_dot_bien_tang', 0) >= nhip_xac_nhan_tang or getattr(self, 'so_nhip_dot_bien_giam', 0) >= nhip_xac_nhan_giam:
            # Đã xác nhận đây là Bơm/Rút lượng lớn
            event_dang_xay_ra = True
            R_thich_nghi = max(1.0, self.r_base * 0.08)
            Q_thich_nghi *= 12.0
            self.ema_z = z
            # CHÌA KHÓA CHỐNG BẬC THANG: Phải bảo toàn ema_v khi Snap để bộ lọc không bị mù hướng!
            self.prev_ema_z = z
            self.ema_v = 0.0
            self.so_nhip_dot_bien_tang = 0
            self.so_nhip_dot_bien_giam = 0
            self.so_nhip_giam_lien_tuc = 0
        else:
            # Rung lắc mạnh thì phạt nhẹ R
            if abs(gia_toc) > 0.1:
                R_thich_nghi *= 3.0

        P_du_doan = self.P + Q_thich_nghi
        K = P_du_doan / (P_du_doan + R_thich_nghi)
        if signal_mode in {"JITTER", "STABLE"}:
            K = min(K, 0.22 if signal_mode == "JITTER" else 0.30)
        elif signal_mode == "TREND_DOWN":
            K = min(max(K, 0.35), 0.62)
        elif signal_mode in {"REFUEL", "DROP_EVENT"}:
            K = min(max(K, 0.45), 0.72)
        elif trend_giam_that:
            K = min(max(K, 0.45), 0.75)
        else:
            K = min(K, 0.65 if event_dang_xay_ra else 0.35)

        phan_du_z = z - x_du_doan
        self.x = x_du_doan + K * phan_du_z
        self.x = max(0.0, self.x)
        self.P = (1.0 - K) ** 2 * P_du_doan + K ** 2 * R_thich_nghi

        return round(self.x, 1)


def is_valid_measurement(fuel, feature_status) -> bool:
    fuel_val = pd.to_numeric(fuel, errors="coerce")
    if pd.isna(fuel_val) or fuel_val <= 0:
        return False

    status = str(feature_status).strip().upper()
    invalid_statuses = {"FUEL_ZERO", "PREVIOUS_INVALID", "INVALID"}
    return status not in invalid_statuses


def chay_kalman_thich_nghi_cho_tat_ca_xe(
    mau_ten_file: str = "CarFuelHistory_Processed_*.csv",
) -> None:
    print("=== BAT DAU CHAY ADAPTIVE KALMAN FILTER (REALTIME GATING) ===")
    danh_sach_file = sorted(glob.glob(mau_ten_file))
    if not danh_sach_file:
        return

    for duong_dan_file in danh_sach_file:
        ma_xe = os.path.basename(duong_dan_file).replace("CarFuelHistory_Processed_", "").replace(".csv", "")
        print(f"Dang chay Adaptive Kalman cho xe {ma_xe}...")

        df = pd.read_csv(duong_dan_file)

        df["FuelTime"] = pd.to_datetime(df["FuelTime"], errors="coerce")
        df["FuelLevel"] = pd.to_numeric(df["FuelLevel"], errors="coerce")
        df["_OriginalOrder"] = np.arange(len(df))
        df = df.sort_values(["SegmentID", "FuelTime", "_OriginalOrder"], kind="stable")

        df["Kalman_Adaptive"] = np.nan

        if "Speed" in df.columns:
            time_gap_sec = df["TimeGapMinutes"].fillna(5.0) * 60.0
            safe_time_gap = time_gap_sec.replace(0, 1.0)
            df["Acceleration"] = (df.groupby("SegmentID", dropna=False)["Speed"].diff().fillna(0.0) / 3.6) / safe_time_gap
        else:
            df["Acceleration"] = 0.0

        valid_gaps = df["TimeGapMinutes"].dropna()
        valid_gaps = valid_gaps[valid_gaps > 0]
        thoi_gian_chuan_phut = valid_gaps.median() if not valid_gaps.empty else 5.0

        for ma_doan, nhom in df.groupby("SegmentID", sort=False, dropna=False):
            kf = None
            khoang_thoi_gian_tich_luy = 0.0
            for dong in nhom.itertuples():
                vi_tri = dong.Index

                gap = getattr(dong, "TimeGapMinutes", thoi_gian_chuan_phut)
                if pd.isna(gap) or gap <= 0:
                    gap = thoi_gian_chuan_phut

                if not is_valid_measurement(getattr(dong, "FuelLevel", None), getattr(dong, "FeatureStatus", "")):
                    khoang_thoi_gian_tich_luy += gap
                    df.loc[vi_tri, "Kalman_Adaptive"] = np.nan
                    continue

                gia_tri_do = float(dong.FuelLevel)

                khoang_thoi_gian = gap + khoang_thoi_gian_tich_luy
                khoang_thoi_gian_tich_luy = 0.0
                ty_le_dt = khoang_thoi_gian / thoi_gian_chuan_phut

                trang_thai_chuyen_dong_thay_the = str(getattr(dong, "MovementState", "Moving")).strip().upper()
                trang_thai_chuyen_dong_int = 0 if trang_thai_chuyen_dong_thay_the == "STOPPED" else 1

                gia_toc_hien_tai = float(getattr(dong, "Acceleration", 0.0))
                if pd.isna(gia_toc_hien_tai):
                    gia_toc_hien_tai = 0.0

                rolling_std_hien_tai = float(getattr(dong, "RollingStd", 0.0))
                if pd.isna(rolling_std_hien_tai):
                    rolling_std_hien_tai = 0.0

                van_toc_hien_tai = float(getattr(dong, "Speed", 0.0) or 0.0)
                if pd.isna(van_toc_hien_tai):
                    van_toc_hien_tai = 0.0

                if kf is None:
                    kf = BoLocKalmanThichNghi1D(
                        trang_thai_ban_dau=gia_tri_do,
                        capacity=200.0,
                        sai_so_uoc_luong_ban_dau=4.0,
                        nhieu_qua_trinh=0.2,
                        r_co_ban=9.0,
                        nhip_cho_xac_nhan=3,
                    )
                    gia_tri_da_loc = gia_tri_do
                else:
                    gia_tri_da_loc = kf.cap_nhat(
                        gia_tri_do,
                        ty_le_dt=ty_le_dt,
                        trang_thai_chuyen_dong=trang_thai_chuyen_dong_int,
                        gia_toc=gia_toc_hien_tai,
                        rolling_std=rolling_std_hien_tai,
                        van_toc=van_toc_hien_tai,
                    )

                df.loc[vi_tri, "Kalman_Adaptive"] = gia_tri_da_loc

        df = df.sort_values("_OriginalOrder", kind="stable").drop(columns="_OriginalOrder")
        duong_dan_xuat = duong_dan_file.replace(".csv", "_Kalman_Adaptive.csv")
        df.to_csv(duong_dan_xuat, index=False, encoding="utf-8-sig")
        print(f"Hoan tat {ma_xe}! Da xuat: {duong_dan_xuat}")

    print("=== DA CHAY XONG ADAPTIVE KALMAN (GATING) ===")


if __name__ == "__main__":
    chay_kalman_thich_nghi_cho_tat_ca_xe()
