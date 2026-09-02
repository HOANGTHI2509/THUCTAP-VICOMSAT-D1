from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import asdict, dataclass
from src.core.signal_labels import to_signal_label


# Chuyển một cột DataFrame sang mảng số float; nếu cột thiếu hoặc giá trị lỗi thì dùng giá trị mặc định.
def _numeric_array(df: pd.DataFrame, column: str, default: float = 0.0) -> np.ndarray:
    if column in df.columns:
        return pd.to_numeric(df[column], errors="coerce").fillna(default).to_numpy(dtype=float)
    return np.full(len(df), float(default), dtype=float)


# Chuyển một cột DataFrame sang mảng chuỗi; nếu cột không tồn tại thì tạo mảng toàn giá trị mặc định.
def _series_or_default(df: pd.DataFrame, column: str, default: str) -> np.ndarray:
    if column in df.columns:
        return df[column].fillna(default).astype(str).to_numpy()
    return np.full(len(df), default, dtype=object)


@dataclass
class RealtimeAdaptiveKalmanState:
    """Trạng thái có thể tuần tự hóa; trong service streaming, mỗi xe phải giữ một instance riêng."""

    x: float | None = None          # Ước lượng nhiên liệu hiện tại của Kalman.
    P: float = 4.0                    # Phương sai/sai số ước lượng của Kalman.
    last_valid_x: float | None = None # Mức nhiên liệu hợp lệ gần nhất, dùng làm mốc khi sensor dropout.
    drain_count: int = 0            # Số nhịp đang hỗ trợ xu hướng giảm thật.
    drop_count: int = 0             # Số nhịp liên tiếp raw thấp hơn x đủ ngưỡng.
    rise_count: int = 0             # Số nhịp liên tiếp raw cao hơn x đủ ngưỡng.
    recent_refuel_steps: int = 0    # Số nhịp ân hạn sau một UPWARD_SHIFT đã xác nhận.
    previous_raw: float | None = None   # Measurement raw ngay trước điểm hiện tại.
    previous_raw_2: float | None = None # Measurement raw cách hiện tại hai nhịp.
    segment_id: str | None = None   # Segment gần nhất của xe.
    fuel_time: str | None = None    # Timestamp gần nhất đã xử lý.
    # Trạng thái chờ xác nhận giảm sâu: khi xuất hiện một cú tụt mạnh, hệ thống giữ mốc nền tại đây.
    # Chỉ khi các điểm đến sau xác nhận theo cách causal mới chuyển thành DOWNWARD_SHIFT; giá trị gần 0 luôn coi là dropout.
    dropout_anchor: float | None = None
    dropout_count: int = 0
    dropout_recovery_count: int = 0
    dropout_seen_near_zero: bool = False

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict | None) -> "RealtimeAdaptiveKalmanState":
        return cls(**(value or {}))


# Chuẩn hóa một giá trị thời gian về pd.Timestamp; dữ liệu lỗi hoặc rỗng sẽ trả về None.
def _as_timestamp(value) -> pd.Timestamp | None:
    if value is None or pd.isna(value):
        return None
    timestamp = pd.to_datetime(value, errors="coerce")
    return None if pd.isna(timestamp) else timestamp


# Xóa toàn bộ memory ngắn hạn của bộ lọc khi đổi segment hoặc khoảng cách thời gian quá lớn.
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
    state.dropout_recovery_count = 0
    state.dropout_seen_near_zero = False


def filter_ai_enhanced_adaptive_realtime(
    group: pd.DataFrame,
    config: dict | None = None,
    state: RealtimeAdaptiveKalmanState | None = None,
    return_state: bool = False,
) -> list[float] | tuple[list[float], RealtimeAdaptiveKalmanState]:
    """Bộ lọc Adaptive Kalman tăng cường AI, chạy causal cho realtime/streaming.

    NGUYÊN TẮC CAUSAL NGHIÊM NGẶT: không nhìn dữ liệu tương lai.
    - Bám dốc mượt: theo xu hướng giảm dần với R=25, Q=0.15 để hạn chế dạng bậc thang.
    - Khóa dâng ảo vật lý: khi không có UPWARD_SHIFT đã xác nhận, không cho x leo theo sóng sánh tăng giả (z > x).
    - Dịch mức tăng phải được xác nhận causal, không chỉ dựa vào một nhãn Random Forest.
    - Giữ phẳng khi dropout: nếu cảm biến mất tín hiệu, giữ mức hợp lệ trước đó thay vì tụt theo giá trị lỗi.

    Khi xử lý các bản tin kế tiếp của cùng một xe, phải truyền lại cùng ``state``.
    ``return_state`` hữu ích khi cần lưu trạng thái sang Redis hoặc cơ sở dữ liệu.
    """
    if group.empty:
        return ([], state or RealtimeAdaptiveKalmanState()) if return_state else []

    # Pipeline streaming nên truyền trực tiếp FuelLevel làm nguồn đầu vào.
    # Các nhánh fallback bên dưới chỉ để tương thích với những script so sánh offline cũ.
    source_col = str((config or {}).get("source_col", "")).strip()
    if source_col not in group.columns:
        if "ShapeCleanFuel" in group.columns:
            source_col = "ShapeCleanFuel"
        elif "CleanedFuel" in group.columns:
            source_col = "CleanedFuel"
        elif "ProfileCleanFuel" in group.columns:
            source_col = "ProfileCleanFuel"
        else:
            source_col = "FuelLevel"

    # Đọc các cột/feature cần thiết thành mảng để xử lý nhanh theo từng điểm.
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

    # Nếu không truyền cấu hình thì dùng dictionary rỗng để lấy các giá trị mặc định bên dưới.
    if config is None:
        config = {}
    stream_state = state or RealtimeAdaptiveKalmanState()
    reset_gap_minutes = float(config.get("reset_gap_minutes", 120.0))
    refuel_confidence = float(config.get("refuel_min_confidence", 0.55))
    dropout_recovery_ratio = float(config.get("dropout_recovery_ratio", 0.70))
    max_dropout_hold_points = int(config.get("max_dropout_hold_points", 120))
    drain_confirm_points = int(config.get("drain_confirm_points", 3))
    # Một mức giảm thật phải duy trì đủ lâu. Chỉ 5 mẫu là quá ngắn và có thể khiến
    # các dropout cảm biến lặp lại bị hiểu nhầm thành một sự kiện DOWNWARD_SHIFT.
    drain_hard_confirm_points = int(config.get("drain_hard_confirm_points", 35))
    drain_min_confidence = float(config.get("drain_min_confidence", 0.55))
    drain_max_gap_minutes = float(config.get("drain_max_gap_minutes", 30.0))
    dropout_recovery_confirm_points = int(config.get("dropout_recovery_confirm_points", 3))

    # Bộ tham số Q/R mặc định cho từng trạng thái tín hiệu. R lớn = ít tin measurement; Q lớn = cho phép trạng thái đổi nhanh.
    cfg_unk_r, cfg_unk_q = config.get("UNKNOWN", (25.0, 0.15))
    cfg_ref_r, cfg_ref_q = config.get("UPWARD_SHIFT", (1.0, 5.0))
    cfg_slosh_r, cfg_slosh_q = config.get("OSCILLATION_NOISE", (1000.0, 0.001))
    cfg_cons_r, cfg_cons_q = config.get("GRADUAL_CHANGE", (25.0, 0.15))
    cfg_drain_r, cfg_drain_q = config.get("DOWNWARD_SHIFT", (5.0, 2.0))
    cfg_stable_r, cfg_stable_q, cfg_stable_r_very = config.get("STABLE_JITTER", (35.0, 0.05, 15.0))
    cfg_spike_r, cfg_spike_q = config.get("IMPULSE_NOISE", (10000.0, 0.0001))

    # Khôi phục trạng thái đã lưu từ bản tin telemetry trước của đúng chiếc xe này.
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
    dropout_recovery_count = int(stream_state.dropout_recovery_count)
    dropout_seen_near_zero = bool(stream_state.dropout_seen_near_zero)
    last_seen_time = _as_timestamp(stream_state.fuel_time)

    # Duyệt tuần tự từng bản tin theo đúng thứ tự thời gian; mọi quyết định chỉ dùng hiện tại + quá khứ.
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
            dropout_recovery_count = 0
            dropout_seen_near_zero = False

        is_qflag_zero = (
            any(reason in qreason for reason in {"FUEL_ZERO", "SENSOR_DROPOUT", "DROPOUT"})
            or (z <= 1.0 and qflag in {"1", "1.0"})
        )

        # Chuẩn hóa nhãn cũ của mô hình sang bộ nhãn trạng thái tín hiệu mới.
        ai_state = to_signal_label(states[i])
        capacity = max(float(capacity_values[i]), 50.0)
        noise = max(float(noise_values[i]), 0.1)
        jitter = max(float(jitter_values[i]), 0.1)
        event = max(float(event_values[i]), jitter * 5.0, noise * 4.0, 2.0)
        current_speed = max(float(speed[i]), 0.0)
        high_noise = float(rolling_std[i]) >= max(jitter * 1.5, noise * 3.0)
        observed = observed_fuel[i] if not pd.isna(observed_fuel[i]) else z

        # PENDING_DRAIN: khi thấy một cú tụt sâu, trước tiên giữ output ở mức nền vật lý cũ.
        # Chỉ khi nhiều điểm thấp liên tiếp đến với chu kỳ lấy mẫu bình thường thì mới xác nhận giảm thật.
        # Nếu chuỗi đã chạm gần 0 L thì luôn ưu tiên coi đó là sensor dropout, không nâng thành DOWNWARD_SHIFT.
        drain_confirmed_now = False
        if not pd.isna(dropout_anchor):
            if observed >= dropout_anchor * dropout_recovery_ratio:
                # Một điểm tăng trở lại đơn lẻ cũng có thể chỉ là cảm biến bật nảy.
                # Vì vậy vẫn giữ baseline cũ cho đến khi mức phục hồi ổn định qua đủ số nhịp.
                dropout_recovery_count += 1
                if dropout_recovery_count < dropout_recovery_confirm_points:
                    enhanced[i] = dropout_anchor
                    continue
                dropout_anchor = np.nan
                dropout_count = 0
                dropout_recovery_count = 0
                dropout_seen_near_zero = False
            elif observed <= dropout_anchor * dropout_recovery_ratio:
                dropout_recovery_count = 0
                dropout_seen_near_zero = dropout_seen_near_zero or observed <= 5.0
                normal_gap = 0.0 <= gap_minutes <= drain_max_gap_minutes
                if normal_gap:
                    dropout_count += 1
                model_supports_drain = ai_state == "DOWNWARD_SHIFT" and float(confidence_values[i]) >= drain_min_confidence
                can_confirm_drain = (
                    not dropout_seen_near_zero
                    and observed > 5.0
                    and (
                        (model_supports_drain and dropout_count >= drain_confirm_points)
                        or dropout_count >= drain_hard_confirm_points
                    )
                )
                if can_confirm_drain:
                    drain_confirmed_now = True
                    dropout_anchor = np.nan
                    dropout_count = 0
                    dropout_recovery_count = 0
                    dropout_seen_near_zero = False
                else:
                    enhanced[i] = dropout_anchor
                    continue
            else:
                dropout_anchor = np.nan
                dropout_count = 0
                dropout_recovery_count = 0
                dropout_seen_near_zero = False

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
                dropout_recovery_count = 0
                dropout_seen_near_zero = observed <= 5.0
                enhanced[i] = last_valid_x  # Giữ phẳng tuyệt đối mức hợp lệ trước đó
                continue
            elif pd.isna(z) or z <= 0:
                # Một đoạn dữ liệu có thể bắt đầu đúng giữa lúc dropout nên chưa có trạng thái lịch sử.
                # Không tự tạo mức 0 L giả; trả NaN/khoảng trống cho tới khi service khôi phục được state của xe
                # hoặc nhận được một measurement hợp lệ mới.
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

        # Chỉ một UPWARD_SHIFT đã được xác nhận mới kích hoạt khoảng ân hạn vài nhịp sau đó.
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

        # Cập nhật bộ đếm DOWNWARD_SHIFT dựa trên xác nhận causal hoặc nhãn AI hiện tại.
        if drain_confirmed_now:
            ai_state = "DOWNWARD_SHIFT"
            drain_count = max(drain_count, 2)
        elif ai_state == "DOWNWARD_SHIFT":
            drain_count += 1
        else:
            drain_count = 0

        # Khóa chống nhiễu rung nhỏ khi xe đỗ (chỉ khóa khi dao động nhỏ quanh x)
        is_parked_noisy = (
            is_truly_parked
            and ai_state not in {"UPWARD_SHIFT", "DOWNWARD_SHIFT"}
            and recent_refuel_steps == 0
            and abs(z - x) <= max(jitter * 2.0, 1.8)
            and (high_noise or ai_state in {"OSCILLATION_NOISE", "IMPULSE_NOISE"})
        )
        if is_parked_noisy:
            enhanced[i] = x
            previous_raw_2 = previous_raw
            previous_raw = float(z)
            continue

        # Đóng băng 1 nhịp nếu nhiễu rung nhỏ khi xe ổn định
        if drop_count == 1 and (x - z) <= jitter * 1.5 and ai_state in {"STABLE_JITTER", "OSCILLATION_NOISE"}:
            enhanced[i] = x
            previous_raw_2 = previous_raw
            previous_raw = float(z)
            continue

        # 1. Xác định nhiễu xung (IMPULSE_NOISE / spike)
        is_spike = (ai_state == "IMPULSE_NOISE")

        # 2. Phục hồi causal khi bị "treo" trạng thái: xe đỗ, raw đã tụt >= 3 nhịp nhưng x vẫn nằm cao.
        trap_recovery = (
            is_truly_parked
            and (ai_state == "STABLE_JITTER")
            and (x - z >= max(event * 0.5, jitter * 2.5, 3.0))
            and (drop_count >= 3)
        )

        # 3. Xác nhận dịch mức tăng (UPWARD_SHIFT; tên biến cũ còn dùng refuel).
        # RF có thể gắn nhầm cạnh tăng của một mức tăng thật thành OSCILLATION_NOISE.
        # Vì vậy hệ thống còn có rule causal độc lập: nếu xe đỗ và mức tăng duy trì qua nhiều nhịp,
        # vẫn có thể xác nhận UPWARD_SHIFT mà không phụ thuộc hoàn toàn vào nhãn RF.
        prev_z = float(raw[i - 1]) if i > 0 else (previous_raw if not pd.isna(previous_raw) else float(z))
        prev2_z = float(raw[i - 2]) if i > 1 else (previous_raw_2 if not pd.isna(previous_raw_2) else prev_z)
        fast_step = (z - prev_z) >= max(event * 0.6, 5.5) or (z - prev2_z) >= max(event * 0.8, 7.5)
        min_refuel_jump = max(0.025 * capacity, event * 0.6, jitter * 4.0, 5.0)
        labeled_refuel = ai_state == "UPWARD_SHIFT" and float(confidence_values[i]) >= refuel_confidence
        sustained_refuel = (
            is_truly_parked
            and observed > 5.0
            and (z - x) >= min_refuel_jump
            and rise_count >= 3
            # Trôi chậm khi xe đỗ thường giống nhiễu cảm biến/nhiệt hơn là một dịch mức tăng thật.
            # Vì vậy rule fallback causal còn yêu cầu phải có một bước nhảy raw đủ nhanh.
            and fast_step
        )

        is_refuel = (labeled_refuel and (fast_step or rise_count >= 2)) or (
            is_truly_parked
            and ai_state not in {"OSCILLATION_NOISE", "IMPULSE_NOISE"}
            and (z - x >= min_refuel_jump)
            and fast_step
        ) or sustained_refuel or (recent_refuel_steps > 0 and (z - x) >= 1.5)

        # Chọn Q/R theo trạng thái cuối cùng đã được AI + rule/memory xác nhận.
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
        elif ai_state == "OSCILLATION_NOISE" or high_noise:
            R = cfg_slosh_r
            Q_current = cfg_slosh_q
            jump_to_z = False
        elif drain_count >= 2:
            R = cfg_drain_r
            Q_current = cfg_drain_q
            jump_to_z = False
        elif ai_state == "GRADUAL_CHANGE" or not is_truly_parked:
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
                elif not is_truly_parked or ai_state in {"GRADUAL_CHANGE", "UNKNOWN"}:
                    slope_lag = min(5.0, (x - z) / max(jitter * 0.5, 0.3))
                    R = max(4.0, R / slope_lag)
                    Q_current = max(Q_current, 0.35 * slope_lag)
            elif z > x + jitter and (ai_state == "UPWARD_SHIFT" or recent_refuel_steps > 0):
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
            # Nếu không có UPWARD_SHIFT đã xác nhận, không cho x leo theo các "quả đồi" sóng sánh
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

    # Ghi toàn bộ trạng thái cuối về object để lần gọi sau của cùng xe có thể tiếp tục đúng chuỗi.
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
    stream_state.dropout_recovery_count = dropout_recovery_count
    stream_state.dropout_seen_near_zero = dropout_seen_near_zero
    if len(group):
        stream_state.segment_id = str(segment_values[-1]) or None
        last_time = _as_timestamp(time_values[-1])
        if last_time is not None:
            stream_state.fuel_time = last_time.isoformat()
    result = enhanced.tolist()
    return (result, stream_state) if return_state else result