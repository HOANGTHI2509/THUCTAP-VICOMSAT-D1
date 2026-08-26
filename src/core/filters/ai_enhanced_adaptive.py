from __future__ import annotations

import numpy as np
import pandas as pd


def _numeric_array(df: pd.DataFrame, column: str, default: float = 0.0) -> np.ndarray:
    if column in df.columns:
        return pd.to_numeric(df[column], errors="coerce").fillna(default).to_numpy(dtype=float)
    return np.full(len(df), float(default), dtype=float)


def _series_or_default(df: pd.DataFrame, column: str, default: str) -> np.ndarray:
    if column in df.columns:
        return df[column].fillna(default).astype(str).to_numpy()
    return np.full(len(df), default, dtype=object)


def _future_median(values: np.ndarray, pos: int, width: int = 3) -> tuple[float, int]:
    future = values[pos + 1 : pos + 1 + width]
    future = future[~np.isnan(future)]
    if len(future) == 0:
        return float(values[pos]), 0
    return float(np.median(future)), int(len(future))


def _confirmed_refuel(
    raw: np.ndarray,
    output: float,
    pos: int,
    event: float,
    jitter: float,
    noise: float,
    speed: float,
) -> tuple[bool, float]:
    z = float(raw[pos])
    delta = z - output
    min_refuel_jump = max(event * 0.6, jitter * 4.0, noise * 3.0, 3.0)
    if delta < min_refuel_jump:
        return False, output

    future, count = _future_median(raw, pos, width=4)
    if count == 0:
        return speed <= 3.0 and delta >= min_refuel_jump, z

    close_to_new_level = abs(future - z) <= max(delta * 0.5, event * 0.70, jitter * 5.0)
    stays_above_old_level = future >= output + min_refuel_jump * 0.70
    
    if stays_above_old_level and close_to_new_level:
        return True, z
    return False, output


def _looks_like_up_spike(raw: np.ndarray, output: float, pos: int, jitter: float, noise: float) -> bool:
    z = float(raw[pos])
    if z <= output + max(jitter * 3.0, noise * 2.0, 1.0):
        return False
    future, count = _future_median(raw, pos, width=3)
    if count == 0:
        return False
    return future <= output + max(jitter * 2.0, noise * 1.5, 1.0)


def filter_ai_enhanced_adaptive(group: pd.DataFrame, config: dict = None) -> list[float]:
    """Use a true Kalman Filter where Measurement Noise is gated by AI labels."""
    if group.empty:
        return []

    if "ShapeCleanFuel" in group.columns:
        source_col = "ShapeCleanFuel"
    elif "CleanedFuel" in group.columns:
        source_col = "CleanedFuel"
    elif "ProfileCleanFuel" in group.columns:
        source_col = "ProfileCleanFuel"
    else:
        source_col = "FuelLevel"

    raw = _numeric_array(group, source_col, np.nan)
    states = _series_or_default(group, "AI_State", "UNKNOWN")
    jitter_values = _numeric_array(group, "flat_jitter_threshold", 0.8)
    noise_values = _numeric_array(group, "noise_sigma_liters", 0.8)
    event_values = _numeric_array(group, "event_threshold", 5.0)
    rolling_std = _numeric_array(group, "RollingStd", 0.0)
    speed = _numeric_array(group, "Speed", 0.0)
    has_distance = "DistanceMeters" in group.columns
    distance_m = _numeric_array(group, "DistanceMeters", 0.0)
    quality_flags = _series_or_default(group, "QualityFlag", "0")

    enhanced = np.full(len(group), np.nan, dtype=float)
    
    if config is None:
        config = {}
    
    cfg_unk_r, cfg_unk_q = config.get("UNKNOWN", (25.0, 0.15))
    cfg_ref_r, cfg_ref_q = config.get("REFUEL", (1.0, 5.0))
    cfg_slosh_r, cfg_slosh_q = config.get("SLOSHING", (1000.0, 0.001))
    cfg_cons_r, cfg_cons_q = config.get("CONSUMPTION", (25.0, 0.15))
    cfg_drain_r, cfg_drain_q = config.get("DRAIN", (5.0, 2.0))
    cfg_stable_r, cfg_stable_q, cfg_stable_r_very = config.get("STABLE_JITTER", (35.0, 0.05, 15.0))
    cfg_spike_r, cfg_spike_q = config.get("SPIKE", (10000.0, 0.0001))

    # Khởi tạo trạng thái Kalman
    x = np.nan
    P = 4.0
    drain_count = 0
    drop_count = 0
    last_valid_x = np.nan
    recent_refuel_steps = 0

    for i in range(len(group)):
        z = raw[i]
        qflag = str(quality_flags[i]) if i < len(quality_flags) else "0"
        is_qflag_zero = (qflag in {"FUEL_ZERO", "SENSOR_DROPOUT", "DROPOUT"}) or (z <= 1.0 and qflag in {"1", "1.0", 1})
        
        # Nhận diện triệt để Lỗi cảm biến mất tín hiệu / ngắt điện / FUEL_ZERO
        future_f, count_f = _future_median(raw, i, width=3)
        is_zero_dropout = (
            pd.isna(z)
            or is_qflag_zero
            or (z <= 1.0 and not pd.isna(last_valid_x) and last_valid_x >= 5.0)
            or (not pd.isna(last_valid_x) and (last_valid_x - z) >= 20.0 and count_f > 0 and future_f <= max(10.0, 0.15 * last_valid_x))
            or (not pd.isna(last_valid_x) and z <= 0.20 * last_valid_x and last_valid_x >= 20.0 and speed[i] <= 1.0)
        )

        if is_zero_dropout:
            if not pd.isna(last_valid_x) and last_valid_x > 3.0:
                enhanced[i] = last_valid_x  # Giữ nguyên mức xăng hợp lệ trước đó
                continue
            elif pd.isna(z) or z <= 0:
                enhanced[i] = x if not pd.isna(x) else 0.0
                continue
            
        if pd.isna(x):
            x = float(z)
            last_valid_x = x
            enhanced[i] = x
            continue

        # Cập nhật last_valid_x khi dữ liệu thực sự hợp lệ
        if not is_zero_dropout and not pd.isna(x) and x > 3.0 and not pd.isna(z) and z > 3.0:
            last_valid_x = float(x)

        state = str(states[i])
        jitter = max(float(jitter_values[i]), 0.1)
        noise = max(float(noise_values[i]), 0.1)
        event = max(float(event_values[i]), jitter * 5.0, noise * 4.0, 2.0)
        current_speed = max(float(speed[i]), 0.0)
        high_noise = float(rolling_std[i]) >= max(jitter * 1.5, noise * 3.0)

        # Xác định trạng thái xe đỗ chuẩn xác kết hợp cả Speed và khoảng cách dịch chuyển GPS
        is_truly_parked = (
            current_speed <= 1.0
            or (has_distance and distance_m[i] <= 15.0 and current_speed <= 3.0)
        )

        # Cập nhật số nhịp sau khi vừa bơm xăng
        if state == "REFUEL":
            recent_refuel_steps = 4
        elif recent_refuel_steps > 0:
            recent_refuel_steps -= 1

        # Đếm số nhịp sụt giảm (để lọc nhầm nhãn AI 1-2 nhịp tạo hình chữ U)
        if z < x - max(jitter * 2.0, 1.5):
            drop_count += 1
        else:
            drop_count = 0

        # Khóa chống nhiễu cục bộ khi xe đỗ (dựa vào Rolling Std và trạng thái sóng sánh)
        # Giúp băng qua toàn bộ các đợt hố sụt nhiễu và quả đồi ảo khi xe đang đỗ, nhưng KHÔNG khóa khi bơm xăng thật
        is_parked_noisy = (
            is_truly_parked
            and state not in {"REFUEL", "DRAIN"}
            and recent_refuel_steps == 0
            and drain_count < 2
            and (z - x) < max(event * 0.6, 4.0)
            and (high_noise or float(rolling_std[i]) >= max(jitter * 1.1, 0.8) or state in {"SLOSHING_NOISE", "SPIKE"})
        )
        if is_parked_noisy:
            enhanced[i] = x
            continue

        # Chỉ đóng băng 1 nhịp nếu x không bị trệch cao quá jitter và nhãn là nhiễu rung nhỏ
        if drop_count == 1 and (x - z) <= jitter * 1.5 and state in {"STABLE_JITTER", "SLOSHING_NOISE"}:
            enhanced[i] = x
            continue

        # 1. Base tracking configuration
        if state == "DRAIN":
            drain_count += 1
        else:
            drain_count = 0

        # Mặc định: bám khá nhanh
        R = cfg_unk_r
        Q_current = cfg_unk_q
        jump_to_z = False

        # 2. Xác định Q và R
        is_spike = (state == "SPIKE") or _looks_like_up_spike(raw, x, i, jitter, noise)
        
        # Cơ chế Giải phóng Bẫy Treo (Trap Recovery): Chỉ kích hoạt khi XE ĐỖ DỪNG NGHỈ mà x bị treo lơ lửng trên ngọn đồi cũ
        future, count = _future_median(raw, i, width=3)
        trap_recovery = (
            is_truly_parked
            and (state == "STABLE_JITTER")
            and (x - z >= max(event * 0.5, jitter * 2.5, 3.0))
            and (count > 0)
            and (future <= x - max(jitter * 1.5, 2.0))
        )

        if is_spike:
            R = cfg_spike_r
            Q_current = cfg_spike_q
        elif trap_recovery:
            # Xóa bẫy lơ lửng do nhô lên nhầm cũ khi đỗ xe, kéo x về mức thực tế
            target_val = future if count > 0 else float(z)
            x = float(target_val)
            R = cfg_stable_r
            Q_current = 0.5
        elif state == "REFUEL" or (z - x) >= max(event * 0.7, 5.0) or (recent_refuel_steps > 0 and (z - x) >= 2.0):
            ok, refuel_target = _confirmed_refuel(raw, x, i, event, jitter, noise, current_speed)
            
            # Chặn ngọn đồi ảo trôi chậm (cả khi đỗ xe lẫn khi xe vừa lăn bánh ở đuôi đồi):
            if ok and state != "REFUEL" and recent_refuel_steps == 0:
                prev_z = float(raw[i - 1]) if i > 0 else float(z)
                prev2_z = float(raw[i - 2]) if i > 1 else prev_z
                fast_step = (z - prev_z) >= max(event * 0.6, 5.5) or (z - prev2_z) >= max(event * 0.8, 7.5)
                if not fast_step:
                    ok = False
            
            if ok:
                jump_to_z = True  # Nhảy thẳng và bám sát ngay lập tức khi CÓ XÁC NHẬN bơm xăng thật
                z = refuel_target
                recent_refuel_steps = 4
            else:
                # Chưa xác nhận -> Coi là sóng sánh (Sloshing/Spike), KHÔNG đẩy x lên đỉnh
                R = cfg_slosh_r
                Q_current = cfg_slosh_q
        elif drain_count >= 2:
            # Tụt liên tục (Rút trộm thật) -> Bám sát nhanh
            R = cfg_drain_r
            Q_current = cfg_drain_q
        elif state == "SLOSHING_NOISE" or high_noise:
            R = cfg_slosh_r
            Q_current = cfg_slosh_q
        elif state == "CONSUMPTION" or not is_truly_parked:
            R = cfg_cons_r
            Q_current = cfg_cons_q
        elif state == "STABLE_JITTER" and is_truly_parked:
            very_stable = float(rolling_std[i]) <= max(jitter * 0.8, 1.0)
            if very_stable:
                R = cfg_stable_r
                Q_current = cfg_stable_q
            else:
                R = cfg_stable_r
                Q_current = cfg_stable_q

        # Thích nghi R và Q tự động:
        # Bám mượt theo dốc tiêu hao khi xe chạy để không bị trễ lơ lửng trên cao
        if not is_spike and not jump_to_z:
            if z < x - jitter:
                if drain_count >= 2:
                    lag_ratio = max(1.0, (x - z) / jitter)
                    R = max(2.0, R / (lag_ratio ** 1.2))
                    Q_current = max(Q_current, 0.15 * lag_ratio)
                elif not is_truly_parked or state in {"CONSUMPTION", "UNKNOWN"}:
                    slope_lag = min(5.0, (x - z) / jitter)
                    R = max(8.0, R / slope_lag)
                    Q_current = max(Q_current, 0.20 * slope_lag)
            elif z > x + jitter and (state == "REFUEL" or recent_refuel_steps > 0):
                lag_ratio = max(1.0, (z - x) / jitter)
                R = max(1.0, R / (lag_ratio ** 1.5))

        # 3. Update Step
        if jump_to_z:
            x = float(z)
            P = 4.0
            recent_refuel_steps = 4
        else:
            P = P + Q_current
            K = P / (P + R)
            x_new = x + K * (z - x)
            P = (1 - K) * P

            # Khóa dâng ảo vật lý:
            # Nếu không phải REFUEL, chỉ khóa khi dữ liệu dềnh ảo (nhiệt độ ban ngày hoặc sóng sánh đơn lẻ).
            # Nếu tương lai duy trì ổn định ở mức cao (xe vận hành thực tế ở mức này), cho phép x thích nghi lên.
            if state != "REFUEL" and recent_refuel_steps == 0 and z > x + jitter:
                is_sustained_higher = (count > 0 and future >= x + max(jitter * 0.8, 0.8) and abs(future - z) <= max(jitter * 2.5, 3.5))
                if is_sustained_higher:
                    x = x_new
                else:
                    x = x
            else:
                x = x_new

        x = max(0.0, float(x))
        enhanced[i] = x

    return enhanced.tolist()


