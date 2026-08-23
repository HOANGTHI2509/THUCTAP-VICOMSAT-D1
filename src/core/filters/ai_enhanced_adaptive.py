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
        return True, max(z, future)
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

    if "CleanedFuel" in group.columns:
        source_col = "CleanedFuel"
    elif "ShapeCleanFuel" in group.columns:
        source_col = "ShapeCleanFuel"
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

    enhanced = np.full(len(group), np.nan, dtype=float)
    
    if config is None:
        config = {}
    
    cfg_unk_r, cfg_unk_q = config.get("UNKNOWN", (16.0, 0.25))
    cfg_ref_r, cfg_ref_q = config.get("REFUEL", (1.0, 5.0))
    cfg_slosh_r, cfg_slosh_q = config.get("SLOSHING", (1000.0, 0.001))
    cfg_cons_r, cfg_cons_q = config.get("CONSUMPTION", (150.0, 0.05))
    cfg_drain_r, cfg_drain_q = config.get("DRAIN", (5.0, 2.0))
    cfg_stable_r, cfg_stable_q, cfg_stable_r_very = config.get("STABLE_JITTER", (50.0, 0.1, 5.0))
    cfg_spike_r, cfg_spike_q = config.get("SPIKE", (10000.0, 0.0001))

    # Khởi tạo trạng thái Kalman
    x = np.nan
    P = 4.0
    drain_count = 0
    drop_count = 0
    last_valid_x = np.nan

    for i in range(len(group)):
        z = raw[i]
        
        # Lưu lại mức nhiên liệu hợp lệ trước đó (khi x > 15.0L)
        if not pd.isna(x) and x > 15.0:
            last_valid_x = float(x)
        
        # Xử lý triệt để Lỗi cảm biến mất tín hiệu/ngắt điện
        # Bắt gọn cả mốc quá độ (z sụt dốc >= 25L về phía 0L) để triệt tiêu vệt kim nhọn
        future_f, count_f = _future_median(raw, i, width=3)
        is_zero_dropout = (
            pd.isna(z)
            or z <= 20.0
            or (not pd.isna(last_valid_x) and (last_valid_x - z) >= 20.0 and count_f > 0 and future_f <= 20.0)
        )

        if is_zero_dropout:
            if not pd.isna(last_valid_x) and last_valid_x > 20.0:
                x = last_valid_x  # Giữ nguyên trạng thái x chuẩn
                enhanced[i] = last_valid_x  # Giữ nguyên mức xăng hợp lệ trước đó (~135L)
                continue
            elif pd.isna(z) or z <= 0:
                enhanced[i] = x if not pd.isna(x) else 0.0
                continue
            
        if pd.isna(x):
            x = float(z)
            enhanced[i] = x
            continue

        state = str(states[i])
        jitter = max(float(jitter_values[i]), 0.1)
        noise = max(float(noise_values[i]), 0.1)
        event = max(float(event_values[i]), jitter * 5.0, noise * 4.0, 2.0)
        current_speed = max(float(speed[i]), 0.0)
        high_noise = float(rolling_std[i]) >= max(jitter * 1.5, noise * 3.0)

        # Đếm số nhịp sụt giảm (để lọc nhầm nhãn AI 1-2 nhịp tạo hình chữ U)
        if z < x - max(jitter * 2.0, 1.5):
            drop_count += 1
        else:
            drop_count = 0

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
        
        # Cơ chế Giải phóng Bẫy Treo (Trap Recovery): Xả ngay khi x bị treo cao hơn z và tương lai đã hạ xuống
        future, count = _future_median(raw, i, width=3)
        trap_recovery = (
            (x - z >= max(event * 0.5, jitter * 2.5, 3.0))
            and (count > 0)
            and (future <= x - max(jitter * 1.5, 2.0))
        )

        if is_spike:
            R = cfg_spike_r
            Q_current = cfg_spike_q
        elif trap_recovery:
            # Xóa ngay bẫy lơ lửng do nhô lên nhầm cũ, kéo x về mức thực tế
            target_val = future if count > 0 else float(z)
            x = float(target_val)
            R = cfg_cons_r
            Q_current = 2.0
        elif state == "REFUEL" or (z - x) >= max(event * 0.6, 3.5):
            ok, refuel_target = _confirmed_refuel(raw, x, i, event, jitter, noise, current_speed)
            
            # Chặn ngọn đồi ảo trôi chậm khi đỗ xe:
            # Bơm xăng thật phải dâng NHANH (bước nhảy z - prev_z >= 3.0L hoặc 2 nhịp dâng >= 5.0L).
            # Trôi ảo khi đỗ dâng chậm (+0.5L - 1.5L/nhịp) sẽ bị chặn ok = False.
            if ok and current_speed <= 1.0 and state != "REFUEL":
                prev_z = float(raw[i - 1]) if i > 0 else float(z)
                prev2_z = float(raw[i - 2]) if i > 1 else prev_z
                fast_step = (z - prev_z) >= 3.0 or (z - prev2_z) >= 5.0
                if not fast_step:
                    ok = False
            
            if ok:
                jump_to_z = True  # Nhảy thẳng và bám sát ngay lập tức khi CÓ XÁC NHẬN bơm xăng thật
                z = refuel_target
            else:
                # Chưa xác nhận -> Coi là sóng sánh (Sloshing/Spike), KHÔNG đẩy x lên đỉnh
                R = cfg_slosh_r
                Q_current = cfg_slosh_q
        elif drain_count >= 2:
            # Tụt liên tục -> Bám sát
            R = cfg_drain_r
            Q_current = cfg_drain_q
        elif state == "SLOSHING_NOISE" or high_noise:
            R = cfg_slosh_r
            Q_current = cfg_slosh_q
        elif state == "CONSUMPTION":
            if x - z > max(jitter * 1.2, 1.5):
                # Dốc sụt tiêu hao rõ ràng -> Hạ R nhanh để bám sát dữ liệu sụt ngay tức thì như Adaptive Kalman
                R = max(1.0, cfg_cons_r / 5.0)
                Q_current = max(Q_current, 2.0)
            else:
                R = cfg_cons_r
                Q_current = cfg_cons_q
        elif state == "STABLE_JITTER" and current_speed <= 1.0:
            very_stable = float(rolling_std[i]) <= max(jitter * 0.8, 1.0)
            if very_stable:
                R = cfg_stable_r_very
                Q_current = cfg_stable_q
            else:
                R = cfg_stable_r
                Q_current = cfg_stable_q

        # Thích nghi R tự động 2 chiều (khi đường lọc trệch phía trên HOẶC phía dưới dữ liệu)
        if not is_spike and not jump_to_z:
            if z < x - jitter:
                lag_ratio = max(1.0, (x - z) / jitter)
                R = max(1.0, R / (lag_ratio ** 1.5))
            elif z > x + jitter and (state == "REFUEL" or (z - x >= 4.0 and current_speed > 1.0)):
                # CHỈ tăng tốc bám lên khi có xác nhận REFUEL hoặc vọt lớn z - x >= 4.0 khi di chuyển.
                # Chặn hoàn toàn trôi ngược khi xe chạy cao tốc tiêu thụ nhiên liệu.
                lag_ratio = max(1.0, (z - x) / jitter)
                R = max(1.0, R / (lag_ratio ** 1.5))

        # 3. Update Step
        if jump_to_z:
            x = float(z)
            P = 4.0
        else:
            P = P + Q_current
            K = P / (P + R)
            x = x + K * (z - x)
            P = (1 - K) * P

        x = max(0.0, float(x))
        enhanced[i] = x

    return enhanced.tolist()
