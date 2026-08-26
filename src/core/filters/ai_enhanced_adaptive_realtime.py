from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import asdict, dataclass


def _numeric_array(df: pd.DataFrame, column: str, default: float = 0.0) -> np.ndarray:
    if column in df.columns:
        return pd.to_numeric(df[column], errors="coerce").fillna(default).to_numpy(dtype=float)
    return np.full(len(df), float(default), dtype=float)


def _series_or_default(df: pd.DataFrame, column: str, default: str) -> np.ndarray:
    if column in df.columns:
        return df[column].fillna(default).astype(str).to_numpy()
    return np.full(len(df), default, dtype=object)


@dataclass
class RealtimeAdaptiveKalmanState:
    """Serializable state. Keep one instance per vehicle in the streaming service."""

    x: float | None = None
    P: float = 4.0
    last_valid_x: float | None = None
    drain_count: int = 0
    drop_count: int = 0
    rise_count: int = 0
    recent_refuel_steps: int = 0
    previous_raw: float | None = None
    previous_raw_2: float | None = None
    segment_id: str | None = None
    fuel_time: str | None = None
    # Pending-drain state. A deep fall starts here and is promoted only after
    # causal confirmation; values near zero are never promoted to DRAIN.
    dropout_anchor: float | None = None
    dropout_count: int = 0

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict | None) -> "RealtimeAdaptiveKalmanState":
        return cls(**(value or {}))


def _as_timestamp(value) -> pd.Timestamp | None:
    if value is None or pd.isna(value):
        return None
    timestamp = pd.to_datetime(value, errors="coerce")
    return None if pd.isna(timestamp) else timestamp


def _reset_state(state: RealtimeAdaptiveKalmanState) -> None:
    state.x = None
    state.P = 4.0
    state.last_valid_x = None
    state.drain_count = 0
    state.drop_count = 0
    state.rise_count = 0
    state.recent_refuel_steps = 0
    state.previous_raw = None
    state.previous_raw_2 = None
    state.dropout_anchor = None
    state.dropout_count = 0


def filter_ai_enhanced_adaptive_realtime(
    group: pd.DataFrame,
    config: dict | None = None,
    state: RealtimeAdaptiveKalmanState | None = None,
    return_state: bool = False,
) -> list[float] | tuple[list[float], RealtimeAdaptiveKalmanState]:
    """Causal (Real-time / Streaming) AI-Enhanced Adaptive Kalman Filter.

    STRICTLY CAUSAL: Does NOT look into future data.
    - Zero shelf-lag: Tightly and smoothly tracks consumption slopes (R=25, Q=0.15) with no square steps.
    - Physical anti-hill lock: Locks flat on non-refuel upward sloshing (z > x).
    - Refuel binding requires a causal confirmation, not only one RF label.
    - Perfect dropout hold: Stays flat across sensor dropouts.

    Pass the same ``state`` for subsequent points of a vehicle. ``return_state``
    is useful when the caller persists state in Redis or a database.
    """
    if group.empty:
        return ([], state or RealtimeAdaptiveKalmanState()) if return_state else []

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
    quality_reasons = _series_or_default(group, "QualityReason", "")
    confidence_values = _numeric_array(group, "AI_State_Confidence", 0.0)
    capacity_values = _numeric_array(group, "capacity_est", 200.0)
    observed_fuel = _numeric_array(group, "FuelLevel", np.nan)
    segment_values = _series_or_default(group, "SegmentID", "")
    time_values = group["FuelTime"].to_numpy() if "FuelTime" in group.columns else np.full(len(group), None)

    enhanced = np.full(len(group), np.nan, dtype=float)

    if config is None:
        config = {}
    stream_state = state or RealtimeAdaptiveKalmanState()
    reset_gap_minutes = float(config.get("reset_gap_minutes", 120.0))
    refuel_confidence = float(config.get("refuel_min_confidence", 0.55))
    dropout_recovery_ratio = float(config.get("dropout_recovery_ratio", 0.70))
    max_dropout_hold_points = int(config.get("max_dropout_hold_points", 120))
    drain_confirm_points = int(config.get("drain_confirm_points", 3))
    drain_hard_confirm_points = int(config.get("drain_hard_confirm_points", 5))
    drain_min_confidence = float(config.get("drain_min_confidence", 0.55))
    drain_max_gap_minutes = float(config.get("drain_max_gap_minutes", 30.0))

    cfg_unk_r, cfg_unk_q = config.get("UNKNOWN", (25.0, 0.15))
    cfg_ref_r, cfg_ref_q = config.get("REFUEL", (1.0, 5.0))
    cfg_slosh_r, cfg_slosh_q = config.get("SLOSHING", (1000.0, 0.001))
    cfg_cons_r, cfg_cons_q = config.get("CONSUMPTION", (25.0, 0.15))
    cfg_drain_r, cfg_drain_q = config.get("DRAIN", (5.0, 2.0))
    cfg_stable_r, cfg_stable_q, cfg_stable_r_very = config.get("STABLE_JITTER", (35.0, 0.05, 15.0))
    cfg_spike_r, cfg_spike_q = config.get("SPIKE", (10000.0, 0.0001))

    # Restore state from the previous telemetry point for this vehicle.
    x = np.nan if stream_state.x is None else float(stream_state.x)
    P = float(stream_state.P)
    last_valid_x = np.nan if stream_state.last_valid_x is None else float(stream_state.last_valid_x)
    drain_count = int(stream_state.drain_count)
    drop_count = int(stream_state.drop_count)
    rise_count = int(stream_state.rise_count)
    recent_refuel_steps = int(stream_state.recent_refuel_steps)
    previous_raw = np.nan if stream_state.previous_raw is None else float(stream_state.previous_raw)
    previous_raw_2 = np.nan if stream_state.previous_raw_2 is None else float(stream_state.previous_raw_2)
    dropout_anchor = np.nan if stream_state.dropout_anchor is None else float(stream_state.dropout_anchor)
    dropout_count = int(stream_state.dropout_count)
    last_seen_time = _as_timestamp(stream_state.fuel_time)

    for i in range(len(group)):
        z = raw[i]
        qflag = str(quality_flags[i]) if i < len(quality_flags) else "0"
        qreason = str(quality_reasons[i]).upper()
        current_segment = str(segment_values[i]) or None
        current_time = _as_timestamp(time_values[i])
        previous_time = last_seen_time
        gap_minutes = (current_time - previous_time).total_seconds() / 60.0 if current_time is not None and previous_time is not None else 0.0
        if current_time is not None:
            last_seen_time = current_time
        segment_changed = stream_state.segment_id is not None and current_segment is not None and current_segment != stream_state.segment_id
        if segment_changed or gap_minutes > reset_gap_minutes:
            _reset_state(stream_state)
            x, P, last_valid_x = np.nan, 4.0, np.nan
            drain_count = drop_count = rise_count = recent_refuel_steps = 0
            previous_raw = previous_raw_2 = np.nan
            dropout_anchor = np.nan
            dropout_count = 0

        is_qflag_zero = (
            any(reason in qreason for reason in {"FUEL_ZERO", "SENSOR_DROPOUT", "DROPOUT"})
            or (z <= 1.0 and qflag in {"1", "1.0"})
        )

        ai_state = str(states[i])
        capacity = max(float(capacity_values[i]), 50.0)
        noise = max(float(noise_values[i]), 0.1)
        jitter = max(float(jitter_values[i]), 0.1)
        event = max(float(event_values[i]), jitter * 5.0, noise * 4.0, 2.0)
        current_speed = max(float(speed[i]), 0.0)
        high_noise = float(rolling_std[i]) >= max(jitter * 1.5, noise * 3.0)
        observed = observed_fuel[i] if not pd.isna(observed_fuel[i]) else z

        # PENDING_DRAIN: a deep fall is held at its physical baseline first.
        # It becomes DRAIN only after consecutive low readings under normal
        # sampling; values near zero are treated as a sensor dropout forever.
        drain_confirmed_now = False
        if not pd.isna(dropout_anchor):
            if observed >= dropout_anchor * dropout_recovery_ratio:
                # Returned near the pre-drop level: this was a sensor fault.
                dropout_anchor = np.nan
                dropout_count = 0
            elif observed <= dropout_anchor * dropout_recovery_ratio:
                normal_gap = 0.0 <= gap_minutes <= drain_max_gap_minutes
                if normal_gap:
                    dropout_count += 1
                model_supports_drain = ai_state == "DRAIN" and float(confidence_values[i]) >= drain_min_confidence
                can_confirm_drain = (
                    observed > 5.0
                    and (
                        (model_supports_drain and dropout_count >= drain_confirm_points)
                        or dropout_count >= drain_hard_confirm_points
                    )
                )
                if can_confirm_drain:
                    drain_confirmed_now = True
                    dropout_anchor = np.nan
                    dropout_count = 0
                else:
                    enhanced[i] = dropout_anchor
                    continue
            else:
                dropout_anchor = np.nan
                dropout_count = 0

        # 1. Nhận diện Lỗi cảm biến mất tín hiệu / rơi về 0 / hố sụt
        is_zero_dropout = not drain_confirmed_now and (
            pd.isna(z)
            or is_qflag_zero
            or (observed <= 5.0 and not pd.isna(last_valid_x) and last_valid_x >= 15.0)
            or (not pd.isna(last_valid_x) and observed <= 0.40 * last_valid_x and last_valid_x >= 20.0)
            or (not pd.isna(last_valid_x) and (last_valid_x - observed) >= max(event * 1.5, 30.0) and observed <= 0.55 * last_valid_x)
        )

        if is_zero_dropout:
            if not pd.isna(last_valid_x) and last_valid_x > 3.0:
                dropout_anchor = last_valid_x
                dropout_count = 1
                enhanced[i] = last_valid_x  # Giữ phẳng tuyệt đối mức hợp lệ trước đó
                continue
            elif pd.isna(z) or z <= 0:
                # A slice can begin inside a dropout and therefore has no
                # historical state. Never manufacture a false 0L level: emit
                # a gap until the service restores the vehicle state or a
                # valid measurement arrives.
                enhanced[i] = x if not pd.isna(x) else np.nan
                continue

        # Khởi tạo điểm hợp lệ đầu tiên
        if pd.isna(x):
            x = float(z)
            last_valid_x = x
            enhanced[i] = x
            previous_raw_2 = previous_raw
            previous_raw = float(z)
            continue

        # Cập nhật last_valid_x khi dữ liệu thực sự hợp lệ
        if not is_zero_dropout and x > 3.0 and not pd.isna(z) and z > 3.0:
            last_valid_x = x

        # Xác định trạng thái xe đỗ chuẩn xác
        is_truly_parked = (
            current_speed <= 1.0
            or (has_distance and distance_m[i] <= 15.0 and current_speed <= 3.0)
        )

        # Only a *confirmed* refuel starts this grace period.
        if recent_refuel_steps > 0:
            recent_refuel_steps -= 1

        # Đếm số nhịp tăng / giảm
        if z >= x + max(jitter * 1.5, 1.2):
            rise_count += 1
        else:
            rise_count = 0

        if z <= x - max(jitter * 1.5, 1.2):
            drop_count += 1
        else:
            drop_count = 0

        if drain_confirmed_now:
            ai_state = "DRAIN"
            drain_count = max(drain_count, 2)
        elif ai_state == "DRAIN":
            drain_count += 1
        else:
            drain_count = 0

        # Khóa chống nhiễu rung nhỏ khi xe đỗ (chỉ khóa khi dao động nhỏ quanh x)
        is_parked_noisy = (
            is_truly_parked
            and ai_state not in {"REFUEL", "DRAIN"}
            and recent_refuel_steps == 0
            and abs(z - x) <= max(jitter * 2.0, 1.8)
            and (high_noise or ai_state in {"SLOSHING_NOISE", "SPIKE"})
        )
        if is_parked_noisy:
            enhanced[i] = x
            previous_raw_2 = previous_raw
            previous_raw = float(z)
            continue

        # Đóng băng 1 nhịp nếu nhiễu rung nhỏ khi xe ổn định
        if drop_count == 1 and (x - z) <= jitter * 1.5 and ai_state in {"STABLE_JITTER", "SLOSHING_NOISE"}:
            enhanced[i] = x
            previous_raw_2 = previous_raw
            previous_raw = float(z)
            continue

        # 1. Xác định Spike
        is_spike = (ai_state == "SPIKE")

        # 2. Causal Trap Recovery: Xe đỗ, tụt kéo dài >= 3 nhịp mà x vẫn treo trên cao
        trap_recovery = (
            is_truly_parked
            and (ai_state == "STABLE_JITTER")
            and (x - z >= max(event * 0.5, jitter * 2.5, 3.0))
            and (drop_count >= 3)
        )

        # 3. Xác nhận Bơm Xăng (REFUEL):
        # Bắt buộc: Nhãn AI là REFUEL, HOẶC dốc đứng cực mạnh (fast_step) vượt ngưỡng nạp và không phải sóng sánh
        prev_z = float(raw[i - 1]) if i > 0 else (previous_raw if not pd.isna(previous_raw) else float(z))
        prev2_z = float(raw[i - 2]) if i > 1 else (previous_raw_2 if not pd.isna(previous_raw_2) else prev_z)
        fast_step = (z - prev_z) >= max(event * 0.6, 5.5) or (z - prev2_z) >= max(event * 0.8, 7.5)
        min_refuel_jump = max(0.025 * capacity, event * 0.6, jitter * 4.0, 5.0)
        labeled_refuel = ai_state == "REFUEL" and float(confidence_values[i]) >= refuel_confidence

        is_refuel = (labeled_refuel and (fast_step or rise_count >= 2)) or (
            is_truly_parked
            and ai_state not in {"SLOSHING_NOISE", "SPIKE"}
            and (z - x >= min_refuel_jump)
            and fast_step
        ) or (recent_refuel_steps > 0 and (z - x) >= 1.5)

        if is_spike:
            R = cfg_spike_r
            Q_current = cfg_spike_q
            jump_to_z = False
        elif trap_recovery:
            x = x + 0.5 * (float(z) - x)
            R = cfg_stable_r
            Q_current = 0.5
            jump_to_z = False
        elif is_refuel:
            R = cfg_ref_r
            Q_current = cfg_ref_q
            recent_refuel_steps = 4
            jump_to_z = True
        elif ai_state == "SLOSHING_NOISE" or high_noise:
            R = cfg_slosh_r
            Q_current = cfg_slosh_q
            jump_to_z = False
        elif drain_count >= 2:
            R = cfg_drain_r
            Q_current = cfg_drain_q
            jump_to_z = False
        elif ai_state == "CONSUMPTION" or not is_truly_parked:
            R = cfg_cons_r
            Q_current = cfg_cons_q
            jump_to_z = False
        else:  # STABLE_JITTER
            very_stable = float(rolling_std[i]) <= max(jitter * 0.8, 1.0)
            R = cfg_stable_r_very if very_stable else cfg_stable_r
            Q_current = cfg_stable_q
            jump_to_z = False

        # Thích nghi R và Q tự động theo độ dốc tiêu hao (bám sát dốc mượt mà không trễ)
        if not is_spike and not jump_to_z:
            if z < x - max(jitter * 0.4, 0.3):
                if drain_count >= 2:
                    lag_ratio = max(1.0, (x - z) / jitter)
                    R = max(2.0, R / (lag_ratio ** 1.2))
                    Q_current = max(Q_current, 0.25 * lag_ratio)
                elif not is_truly_parked or ai_state in {"CONSUMPTION", "UNKNOWN"}:
                    slope_lag = min(5.0, (x - z) / max(jitter * 0.5, 0.3))
                    R = max(4.0, R / slope_lag)
                    Q_current = max(Q_current, 0.35 * slope_lag)
            elif z > x + jitter and (ai_state == "REFUEL" or recent_refuel_steps > 0):
                lag_ratio = max(1.0, (z - x) / jitter)
                R = max(1.0, R / (lag_ratio ** 1.5))

        # Cập nhật Kalman
        if jump_to_z:
            x = float(z)
            P = 4.0
            recent_refuel_steps = 4
        else:
            P = P + Q_current
            K = P / (P + R)
            x_new = x + K * (z - x)
            P = (1 - K) * P

            # KHÓA DÂNG ẢO VẬT LÝ KHI XE ĐANG ĐỖ / SÓNG SÁNH:
            # Nếu không phải REFUEL xác nhận, không cho x leo dốc theo các quả đồi sóng sánh
            if not is_refuel and recent_refuel_steps == 0 and z > x + jitter:
                x = x  # Khóa phẳng xuyên qua các đợt sóng sánh dâng ảo
            else:
                x = x_new  # Bám sát mượt mà dốc tiêu hao (không bị bậc thang vuông)

        x = max(0.0, float(x))
        enhanced[i] = x
        if not is_zero_dropout and not pd.isna(z):
            last_valid_x = x
            previous_raw_2 = previous_raw
            previous_raw = float(z)
        stream_state.segment_id = current_segment
        stream_state.fuel_time = current_time.isoformat() if current_time is not None else stream_state.fuel_time

    stream_state.x = None if pd.isna(x) else float(x)
    stream_state.P = float(P)
    stream_state.last_valid_x = None if pd.isna(last_valid_x) else float(last_valid_x)
    stream_state.drain_count = drain_count
    stream_state.drop_count = drop_count
    stream_state.rise_count = rise_count
    stream_state.recent_refuel_steps = recent_refuel_steps
    stream_state.previous_raw = None if pd.isna(previous_raw) else float(previous_raw)
    stream_state.previous_raw_2 = None if pd.isna(previous_raw_2) else float(previous_raw_2)
    stream_state.dropout_anchor = None if pd.isna(dropout_anchor) else float(dropout_anchor)
    stream_state.dropout_count = dropout_count
    if len(group):
        stream_state.segment_id = str(segment_values[-1]) or None
        last_time = _as_timestamp(time_values[-1])
        if last_time is not None:
            stream_state.fuel_time = last_time.isoformat()
    result = enhanced.tolist()
    return (result, stream_state) if return_state else result
