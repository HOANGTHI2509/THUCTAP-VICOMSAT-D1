from __future__ import annotations

import math
import numpy as np
import pandas as pd
from dataclasses import asdict, dataclass, field
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
    # Ứng viên dịch mức hai chiều. Đây chỉ là trạng thái làm sạch tín hiệu,
    # không mang nghĩa "đổ" hay "rút" nhiên liệu.
    level_candidate_direction: str | None = None  # "UP" | "DOWN"
    level_candidate_anchor: float | None = None
    level_candidate_center: float | None = None
    level_candidate_min_shift: float | None = None
    level_candidate_required_points: int = 3
    level_candidate_samples: list[float] = field(default_factory=list)
    level_candidate_times: list[str] = field(default_factory=list)
    # Nhánh độc lập cho "đồi" tăng nhỏ; không dùng chung bộ đếm với
    # UPWARD_LEVEL_SHIFT lớn để tránh làm chậm một lần tăng thật.
    micro_up_anchor: float | None = None
    micro_up_center: float | None = None
    micro_up_samples: list[float] = field(default_factory=list)
    micro_up_times: list[str] = field(default_factory=list)
    micro_up_release_steps: int = 0
    dt_expected_minutes: float = 2.0

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
    state.level_candidate_direction = None
    state.level_candidate_anchor = None
    state.level_candidate_center = None
    state.level_candidate_min_shift = None
    state.level_candidate_required_points = 3
    state.level_candidate_samples.clear()
    state.level_candidate_times.clear()
    state.micro_up_anchor = None
    state.micro_up_center = None
    state.micro_up_samples.clear()
    state.micro_up_times.clear()
    state.micro_up_release_steps = 0
    state.dt_expected_minutes = 2.0
    # Không để segment cũ làm reset lặp lại ở điểm kế tiếp của segment mới.
    state.segment_id = None
    state.fuel_time = None


def _clear_level_candidate(state: RealtimeAdaptiveKalmanState) -> None:
    """Xóa ứng viên chưa chốt, giữ nguyên mức Kalman đã phát ra."""
    state.level_candidate_direction = None
    state.level_candidate_anchor = None
    state.level_candidate_center = None
    state.level_candidate_min_shift = None
    state.level_candidate_required_points = 3
    state.level_candidate_samples.clear()
    state.level_candidate_times.clear()


def _clear_micro_up_candidate(state: RealtimeAdaptiveKalmanState) -> None:
    state.micro_up_anchor = None
    state.micro_up_center = None
    state.micro_up_samples.clear()
    state.micro_up_times.clear()


def _candidate_duration_minutes(state: RealtimeAdaptiveKalmanState) -> float:
    if len(state.level_candidate_times) < 2:
        return 0.0
    first = _as_timestamp(state.level_candidate_times[0])
    last = _as_timestamp(state.level_candidate_times[-1])
    if first is None or last is None:
        return 0.0
    return max(0.0, (last - first).total_seconds() / 60.0)


def _directionality(samples: list[float]) -> float:
    """1 = đi một hướng rõ ràng; gần 0 = lên/xuống lẫn lộn (sloshing)."""
    if len(samples) < 2:
        return 0.0
    values = np.asarray(samples, dtype=float)
    path = float(np.abs(np.diff(values)).sum())
    return abs(float(values[-1] - values[0])) / path if path > 1e-9 else 1.0


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
    level_shift_confirm_points = int(config.get("level_shift_confirm_points", 3))
    level_shift_min_duration = float(config.get("level_shift_min_duration_minutes", 4.0))
    level_shift_max_gap = float(config.get("level_shift_max_gap_minutes", 7.5))

    # Bộ tham số Q/R cho từng trạng thái tín hiệu (hỗ trợ cả nhãn mới và cấu hình dashboard).
    cfg_unk_r, cfg_unk_q = config.get("UNKNOWN", (25.0, 0.15))
    cfg_ref_r, cfg_ref_q = config.get("UPWARD_LEVEL_SHIFT", config.get("UPWARD_SHIFT", config.get("REFUEL", (1.0, 5.0))))
    cfg_slosh_r, cfg_slosh_q = config.get("OSCILLATION_NOISE", config.get("SLOSHING", (1000.0, 0.001)))
    cfg_cons_r, cfg_cons_q = config.get("GRADUAL_CHANGE", config.get("CONSUMPTION", (25.0, 0.15)))
    cfg_drain_r, cfg_drain_q = config.get("DOWNWARD_LEVEL_SHIFT", config.get("DOWNWARD_SHIFT", config.get("DRAIN", (5.0, 2.0))))
    cfg_stable_r, cfg_stable_q, cfg_stable_r_very = config.get("STABLE_JITTER", (35.0, 0.05, 15.0))
    cfg_spike_r, cfg_spike_q = config.get("IMPULSE_NOISE", config.get("SPIKE", (10000.0, 0.0001)))

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
    dt_expected_minutes = max(0.5, float(stream_state.dt_expected_minutes or 2.0))

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
        if 0.25 <= gap_minutes <= 15.0:
            # Học chu kỳ gửi tin của đúng xe này (2 phút hoặc 5 phút), chỉ từ
            # khoảng bình thường để gap bất thường không làm méo mốc.
            dt_expected_minutes = 0.8 * dt_expected_minutes + 0.2 * gap_minutes
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
            # Ghi nhận segment mới ngay, kể cả khi điểm này là điểm khởi tạo
            # và nhánh bên dưới thoát sớm.
            stream_state.segment_id = current_segment
            stream_state.fuel_time = current_time.isoformat() if current_time is not None else None

        is_qflag_zero = (
            any(reason in qreason for reason in {"FUEL_ZERO", "SENSOR_DROPOUT", "DROPOUT"})
            or (z <= 1.0 and qflag in {"1", "1.0"})
        )

        # Chuẩn hóa nhãn trạng thái tín hiệu trực tiếp (ưu tiên nhãn mới, dự phòng nhãn cũ).
        raw_state = str(states[i]).strip().upper()
        legacy_map = {
            "REFUEL": "UPWARD_SHIFT",
            "DRAIN": "DOWNWARD_SHIFT",
            "CONSUMPTION": "GRADUAL_CHANGE",
            "SLOSHING": "OSCILLATION_NOISE",
            "SLOSHING_NOISE": "OSCILLATION_NOISE",
            "SPIKE": "IMPULSE_NOISE",
            "UPWARD_LEVEL_SHIFT": "UPWARD_SHIFT",
            "DOWNWARD_LEVEL_SHIFT": "DOWNWARD_SHIFT",
        }
        ai_state = legacy_map.get(raw_state, raw_state)
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
        # Chỉ coi 0/gần 0 hoặc cờ chất lượng là dropout. Một mức thấp sâu
        # nhưng hợp lệ có thể là chuyển mức thật; nó sẽ được xác nhận bằng
        # ứng viên DOWN sau 2--3 mẫu thay vì bị giữ 35 nhịp.
        is_zero_dropout = not drain_confirmed_now and (
            pd.isna(z)
            or is_qflag_zero
            or (observed <= 5.0 and not pd.isna(last_valid_x) and last_valid_x >= 15.0)
        )

        if is_zero_dropout:
            # Một measurement không hợp lệ không được nối hai cụm ổn định ở
            # hai phía của nó thành cùng một chuyển mức.
            _clear_level_candidate(stream_state)
            _clear_micro_up_candidate(stream_state)
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
        if stream_state.micro_up_release_steps > 0:
            stream_state.micro_up_release_steps -= 1

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
        elif (
            ai_state == "DOWNWARD_SHIFT"
            and float(confidence_values[i]) >= drain_min_confidence
            and z <= x - max(event, 0.03 * capacity, 5.0)
        ):
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
            and (x - z >= max(event * 1.5, 0.05 * capacity, 10.0))
            and (drop_count >= 4)
        )

        # 3. Xác nhận dịch mức hai chiều, hoàn toàn causal.
        # Một điểm cao/thấp đơn lẻ chỉ mở ứng viên. Muốn đổi mức phải có cụm
        # 3 điểm gần nhau, đi theo một hướng và không bị ngắt quãng. Vì vậy
        # các "quả đồi"/chữ U ngắn không kéo đường sạch theo, còn mức thật
        # được bám sau 2--3 nhịp (4 phút với chu kỳ 2 phút).
        # Chỉ chuyển mức trực tiếp với biến động đủ lớn. Dao động vài lít
        # không phải một sự kiện mức: để Kalman xử lý đối xứng và làm mượt.
        # Điều này ngăn các cụm nhiễu 3 điểm tạo ra đường bậc thang.
        # Ngưỡng hai tầng:
        # - dịch mức đứng riêng phải đủ lớn (xấp xỉ 3% bình);
        # - bậc tăng vừa chỉ được xét khi nó nối tiếp một UP đã xác nhận.
        # Nhờ đó cụm đồi 279->293 L không bị chốt, trong khi phần cuối của
        # quá trình nạp 815->828 L vẫn được bám sau 3 mẫu.
        major_shift_min = max(event * 0.8, 0.03 * capacity, jitter * 3.0, noise * 4.0, 10.0)
        continuation_shift_min = max(jitter * 1.5, noise * 2.0, 0.005 * capacity, 3.0)
        upward_open_threshold = continuation_shift_min if recent_refuel_steps > 0 else major_shift_min
        # Chiều giảm ưu tiên bảo toàn mức thấp bền vững cho tầng nghiệp vụ
        # phía sau; sau 3 mẫu vẫn bám, nhưng không tự kết luận nguyên nhân.
        downward_open_threshold = max(event * 0.35, 0.0125 * capacity, jitter * 3.0, noise * 4.0, 8.0)
        micro_up_min = max(jitter * 1.25, noise * 1.5, 2.0)
        micro_up_tolerance = max(jitter * 1.5, noise * 2.0, 2.5)
        plateau_tolerance = max(jitter * 2.0, noise * 2.5, 3.0)
        return_to_anchor_tolerance = max(jitter * 1.25, noise * 1.5, 1.5)
        candidate_confirmed_up = False
        candidate_confirmed_down = False
        sample_time = current_time.isoformat() if current_time is not None else ""

        # Nhánh đồi tăng nhỏ: chỉ hoạt động bên dưới ngưỡng mở UP lớn và
        # hoàn toàn không kế thừa mẫu/bộ đếm sang nhánh tăng lớn.
        micro_anchor = stream_state.micro_up_anchor
        micro_center = stream_state.micro_up_center
        if micro_anchor is not None:
            returned_to_anchor = abs(z - micro_anchor) <= return_to_anchor_tolerance
            promoted_to_large_shift = z >= x + upward_open_threshold
            interrupted = gap_minutes > level_shift_max_gap
            same_micro_plateau = (
                micro_center is not None
                and abs(z - micro_center) <= micro_up_tolerance
                and z >= micro_anchor + micro_up_min
            )
            if returned_to_anchor or interrupted or promoted_to_large_shift:
                _clear_micro_up_candidate(stream_state)
            elif same_micro_plateau:
                stream_state.micro_up_samples.append(float(z))
                stream_state.micro_up_times.append(sample_time)
                stream_state.micro_up_samples = stream_state.micro_up_samples[-5:]
                stream_state.micro_up_times = stream_state.micro_up_times[-5:]
                stream_state.micro_up_center = float(np.median(stream_state.micro_up_samples))
                micro_required = 4 if (not is_truly_parked or high_noise) else 3
                micro_duration = 0.0
                if len(stream_state.micro_up_times) >= 2:
                    micro_first = _as_timestamp(stream_state.micro_up_times[0])
                    micro_last = _as_timestamp(stream_state.micro_up_times[-1])
                    if micro_first is not None and micro_last is not None:
                        micro_duration = max(0.0, (micro_last - micro_first).total_seconds() / 60.0)
                if len(stream_state.micro_up_samples) >= micro_required and micro_duration >= 4.0:
                    _clear_micro_up_candidate(stream_state)
                    # Sau khi cụm nhỏ đã bền vững, trả quyền cho Kalman trong
                    # vài nhịp để tiến lên mượt thay vì nhảy thẳng.
                    stream_state.micro_up_release_steps = int(config.get("micro_up_release_steps", 8))
            else:
                _clear_micro_up_candidate(stream_state)

        if (
            stream_state.micro_up_anchor is None
            and stream_state.micro_up_release_steps == 0
            and stream_state.level_candidate_direction is None
            and recent_refuel_steps == 0
            and z >= x + micro_up_min
            and z < x + upward_open_threshold
        ):
            stream_state.micro_up_anchor = float(x)
            stream_state.micro_up_center = float(z)
            stream_state.micro_up_samples = [float(z)]
            stream_state.micro_up_times = [sample_time]

        candidate_direction = stream_state.level_candidate_direction
        candidate_anchor = stream_state.level_candidate_anchor
        candidate_center = stream_state.level_candidate_center
        level_shift_min = float(stream_state.level_candidate_min_shift or major_shift_min)

        # Quay về quanh nền ban đầu là bằng chứng mạnh cho nhiễu chữ U/hill;
        # hủy ứng viên thay vì biến nó thành chuyển mức.
        if candidate_direction is not None and candidate_anchor is not None:
            if abs(z - candidate_anchor) <= return_to_anchor_tolerance:
                _clear_level_candidate(stream_state)
                candidate_direction = None
            elif gap_minutes > level_shift_max_gap:
                # Không được cộng dồn bằng chứng qua khoảng trống telemetry.
                _clear_level_candidate(stream_state)
                candidate_direction = None

        if candidate_direction is None:
            opens_down_candidate = False
            if z >= x + upward_open_threshold:
                stream_state.level_candidate_direction = "UP"
                stream_state.level_candidate_anchor = float(x)
                stream_state.level_candidate_center = float(z)
                stream_state.level_candidate_min_shift = float(upward_open_threshold)
                stream_state.level_candidate_required_points = level_shift_confirm_points
                stream_state.level_candidate_samples = [float(z)]
                stream_state.level_candidate_times = [sample_time]
            else:
                raw_drop_step = (
                    previous_raw - z
                    if not pd.isna(previous_raw)
                    else 0.0
                )
                abrupt_down_step = max(event * 0.35, jitter * 2.0, noise * 3.0, 5.0)
                model_supports_abrupt_down = (
                    ai_state == "DOWNWARD_SHIFT"
                    and float(confidence_values[i]) >= drain_min_confidence
                    and raw_drop_step >= max(jitter * 1.5, noise * 2.0, 2.5)
                )
                opens_down_candidate = (
                    z <= x - downward_open_threshold
                    and (raw_drop_step >= abrupt_down_step or model_supports_abrupt_down)
                )
            if candidate_direction is None and opens_down_candidate:
                stream_state.level_candidate_direction = "DOWN"
                stream_state.level_candidate_anchor = float(x)
                stream_state.level_candidate_center = float(z)
                stream_state.level_candidate_min_shift = float(downward_open_threshold)
                # Bắt đầu giảm khi đang chạy hoặc cửa sổ đang nhiễu cần thêm
                # một mẫu: 4 điểm thay vì 3 điểm ở trạng thái đỗ ổn định.
                stream_state.level_candidate_required_points = (
                    max(4, level_shift_confirm_points)
                    if (not is_truly_parked or high_noise)
                    else level_shift_confirm_points
                )
                stream_state.level_candidate_samples = [float(z)]
                stream_state.level_candidate_times = [sample_time]
        else:
            candidate_center = float(candidate_center if candidate_center is not None else z)
            same_plateau = abs(z - candidate_center) <= plateau_tolerance
            same_direction = (
                (candidate_direction == "UP" and z >= x + level_shift_min)
                or (candidate_direction == "DOWN" and z <= x - level_shift_min)
            )
            if same_plateau and same_direction:
                stream_state.level_candidate_samples.append(float(z))
                stream_state.level_candidate_times.append(sample_time)
                # Bộ nhớ bị chặn; quyết định chỉ dùng 5 điểm gần nhất.
                stream_state.level_candidate_samples = stream_state.level_candidate_samples[-5:]
                stream_state.level_candidate_times = stream_state.level_candidate_times[-5:]
                stream_state.level_candidate_center = float(np.median(stream_state.level_candidate_samples))
            elif candidate_direction == "UP" and z > candidate_center + plateau_tolerance:
                # Tăng nhiều bậc cùng hướng: giữ chuỗi để nhận ra một quá
                # trình tăng thật, nhưng tâm so sánh chuyển sang bậc mới.
                stream_state.level_candidate_center = float(z)
                stream_state.level_candidate_samples.append(float(z))
                stream_state.level_candidate_times.append(sample_time)
                stream_state.level_candidate_samples = stream_state.level_candidate_samples[-5:]
                stream_state.level_candidate_times = stream_state.level_candidate_times[-5:]
            elif candidate_direction == "DOWN" and z < candidate_center - plateau_tolerance:
                # Giảm thật thường đi qua nhiều bậc trước khi phẳng. Không
                # reset bộ đếm ở từng bậc vì sẽ gây trễ hàng chục phút.
                stream_state.level_candidate_center = float(z)
                stream_state.level_candidate_samples.append(float(z))
                stream_state.level_candidate_times.append(sample_time)
                stream_state.level_candidate_samples = stream_state.level_candidate_samples[-5:]
                stream_state.level_candidate_times = stream_state.level_candidate_times[-5:]
            else:
                # Đảo hướng hoặc không còn nằm ở cụm mới: đây là sloshing.
                _clear_level_candidate(stream_state)

        samples = stream_state.level_candidate_samples
        duration = _candidate_duration_minutes(stream_state)
        stable_spread = (max(samples) - min(samples)) if samples else np.inf
        # Tính cả đoạn từ nền cũ tới ứng viên. Cụm mới có thể phẳng (173,
        # 173, 173), khi đó directionality nội bộ bằng 0 nhưng rõ ràng vẫn là
        # một dịch mức so với nền 170.
        directional = _directionality(
            [float(stream_state.level_candidate_anchor)] + samples
            if stream_state.level_candidate_anchor is not None else samples
        )
        duration_ok = duration >= level_shift_min_duration or not sample_time
        required_points = max(2, int(stream_state.level_candidate_required_points))
        plateau_ready = (
            len(samples) >= required_points
            and duration_ok
            and stable_spread <= plateau_tolerance
            and directional >= 0.50
            and ai_state != "IMPULSE_NOISE"
        )
        # Nhánh ramp chỉ được chốt nhanh khi mẫu hiện tại vẫn
        # tiếp tục mở rộng mức theo đúng hướng. Trường hợp
        # 824 -> 849.9 -> 847.5 đã quay đầu ở mẫu thứ ba; nếu chỉ
        # nhìn directionality tổng thể thì nó vẫn rất cao và bị chốt
        # nhầm thành UPWARD_LEVEL_SHIFT. Một lần tăng thật có quay
        # đầu nhỏ vẫn có thể được xác nhận bằng plateau_ready sau
        # khi mặt bằng cuối ổn định.
        ramp_is_still_advancing = False
        if len(samples) >= 2:
            ramp_epsilon = max(noise * 0.25, 0.3)
            if stream_state.level_candidate_direction == "UP":
                ramp_is_still_advancing = samples[-1] >= max(samples[:-1]) - ramp_epsilon
            elif stream_state.level_candidate_direction == "DOWN":
                ramp_is_still_advancing = samples[-1] <= min(samples[:-1]) + ramp_epsilon

        directional_ramp_ready = (
            len(samples) >= required_points
            and duration_ok
            and directional >= 0.85
            and ramp_is_still_advancing
            and stream_state.level_candidate_anchor is not None
            and abs(samples[-1] - stream_state.level_candidate_anchor) >= 1.5 * level_shift_min
            and ai_state != "IMPULSE_NOISE"
        )
        candidate_ready = plateau_ready or directional_ramp_ready
        if candidate_ready:
            # Với ramp mạnh, điểm mới nhất phản ánh bậc hiện tại tốt hơn
            # median; với mặt bằng phẳng dùng median để kháng nhiễu.
            candidate_target = float(samples[-1] if directional_ramp_ready else np.median(samples))
            confirmed_direction = stream_state.level_candidate_direction
            if confirmed_direction == "UP":
                candidate_confirmed_up = True
            elif confirmed_direction == "DOWN":
                candidate_confirmed_down = True
            _clear_level_candidate(stream_state)
        else:
            candidate_target = np.nan
        pending_level_shift = (
            stream_state.level_candidate_direction is not None
            or stream_state.micro_up_anchor is not None
        )

        # Giữ tên biến cũ để không phá API/config cũ. Ý nghĩa ở đây chỉ là
        # UPWARD_LEVEL_SHIFT đã xác nhận, không kết luận đó là đổ nhiên liệu.
        is_refuel = candidate_confirmed_up
        if candidate_confirmed_down:
            ai_state = "DOWNWARD_SHIFT"
            drain_count = max(drain_count, 2)

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
            recent_refuel_steps = int(config.get("upward_continuation_steps", 10))
            jump_to_z = True
        elif candidate_confirmed_down:
            # Bằng chứng chuỗi thời gian đã xác nhận phải có ưu tiên cao hơn
            # RollingStd/high_noise; một chuyển mức thật tự nó cũng làm độ
            # lệch chuẩn cửa sổ tăng mạnh.
            R = cfg_drain_r
            Q_current = cfg_drain_q
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

        # Nền làm mượt theo nhiễu đo thực tế và trạng thái AI; không ép R
        # tăng theo dung tích bình.
        if not is_spike and not jump_to_z and drain_count < 2 and not trap_recovery:
            # R phan anh do tin cay cua phep do, khong phai kich thuoc binh.
            # Dung tich chi con tham gia cac nguong dich muc o phia tren.
            local_std = max(0.25, float(rolling_std[i]))
            if ai_state == "OSCILLATION_NOISE" or high_noise:
                sloshing_noise_r = float(config.get(
                    "sloshing_noise_r",
                    (6.0 * local_std) ** 2,
                ))
                R = max(float(R), sloshing_noise_r)
            elif ai_state == "GRADUAL_CHANGE":
                # RollingStd cua ramp gom ca do doc, khong binh phuong truc
                # tiep nhu sloshing. Chi lay phan nhieu cuc bo vua phai.
                gradual_noise_r = float(config.get(
                    "gradual_noise_r",
                    (2.0 * min(local_std, max(noise, 1.0))) ** 2,
                ))
                R = max(float(R), gradual_noise_r)
                Q_current = max(float(Q_current), 0.5)
            else:
                stable_noise_r = float(config.get(
                    "stable_noise_r",
                    (3.0 * local_std) ** 2,
                ))
                R = max(float(R), stable_noise_r)
                Q_current = max(float(Q_current), 0.15)


        # Thích nghi R và Q tự động theo độ dốc tiêu hao (bám sát dốc mượt mà không trễ)
        if not is_spike and not jump_to_z:
            if z < x - max(jitter * 0.4, 0.3):
                if drain_count >= 2:
                    lag_ratio = max(1.0, (x - z) / jitter)
                    R = max(2.0, R / (lag_ratio ** 1.2))
                    Q_current = max(Q_current, 0.25 * lag_ratio)
                elif (
                    not pd.isna(previous_raw)
                    and not pd.isna(previous_raw_2)
                    and z <= previous_raw <= previous_raw_2
                    and (not is_truly_parked or ai_state in {"GRADUAL_CHANGE", "UNKNOWN"})
                ):
                    slope_lag = min(5.0, (x - z) / max(jitter * 0.5, 0.3))
                    R = max(4.0, R / slope_lag)
                    Q_current = max(Q_current, 0.35 * slope_lag)
            # Chiều tăng không được nới R chỉ vì một gai raw cao. Nó phải đi
            # qua candidate ở trên; như vậy ân hạn không thể kéo x lên đỉnh U.

        # Cập nhật Kalman
        if jump_to_z:
            x = float(candidate_target) if not pd.isna(candidate_target) else float(z)
            P = 4.0
            if candidate_confirmed_down:
                recent_refuel_steps = 0
        else:
            # Q là process noise theo một nhịp danh định. Khoảng 5 phút phải
            # tạo uncertainty lớn hơn khoảng 2 phút, nhưng vẫn bị kẹp để gap
            # bất thường không sinh một bước nhảy vô lý.
            dt_scale = min(3.0, max(0.5, gap_minutes / dt_expected_minutes)) if gap_minutes > 0 else 1.0
            P = P + Q_current * dt_scale
            K = P / (P + R)
            diff = float(z) - x
            shrink_threshold = max(1.5, jitter * 1.5)
            diff_compressed = shrink_threshold * math.tanh(diff / shrink_threshold)
            x_new = x + K * diff_compressed
            P = (1 - K) * P

            # Trong vùng dao động thông thường phải lọc đối xứng: khóa riêng
            # chiều tăng sẽ tạo thiên lệch xuống và khiến các đoạn phục hồi
            # bị treo. Gai lớn vẫn bị chặn bởi IMPULSE_NOISE/candidate.
            x = x if pending_level_shift else x_new

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
    stream_state.dt_expected_minutes = float(dt_expected_minutes)
    if len(group):
        stream_state.segment_id = str(segment_values[-1]) or None
        last_time = _as_timestamp(time_values[-1])
        if last_time is not None:
            stream_state.fuel_time = last_time.isoformat()
    result = enhanced.tolist()
    return (result, stream_state) if return_state else result


# =============================================================================
# BỘ LỌC KALMAN THÍCH NGHI 1D (INNOVATION GATING & SPEED BUCKETS)
# =============================================================================
class BoLocKalmanThichNghi1D:
    def __init__(
        self,
        trang_thai_ban_dau: float,
        capacity: float = 200.0,
        sai_so_uoc_luong_ban_dau: float = 4.0,
        nhieu_qua_trinh: float = 0.1, 
        r_co_ban: float = 9.0,
        nhip_cho_xac_nhan: int = 3,           
    ) -> None:
        self.x = float(trang_thai_ban_dau)
        self.P = float(sai_so_uoc_luong_ban_dau)
        self.Q = float(nhieu_qua_trinh)
        self.capacity = float(capacity)
        self.r_base = float(r_co_ban)
        self.threshold = max(5.0, 0.02 * self.capacity)
        self.persistence_required = int(nhip_cho_xac_nhan)
        self.so_nhip_nhieu_lon = 0
        self.dau_nhieu_truoc_do = 0
        
    def cap_nhat(
        self, 
        gia_tri_do: float, 
        ty_le_dt: float = 1.0, 
        trang_thai_chuyen_dong: int = 1, 
        gia_toc: float = 0.0,
        rolling_std: float = 0.0,
        van_toc: float = 0.0,
        xac_nhan_nap_nhanh: bool = False,
    ) -> float:
        z = float(gia_tri_do)

        phan_du = z - self.x
        phan_du_tuyet_doi = abs(phan_du)
        dau_hien_tai = int(np.sign(phan_du))

        nguong_hien_tai = max(self.threshold, 2.0 * rolling_std)

        if xac_nhan_nap_nhanh and phan_du >= 1.5:
            # Nhảy thẳng 100% lên mức xăng thô ngay lập tức (không trễ) để bám khít đường nạp nhiên liệu
            self.x = z
            self.P = 4.0
            self.so_nhip_nhieu_lon = 0
            self.dau_nhieu_truoc_do = 0
            return self.x

        noise_context = (
            van_toc > 10.0
            or rolling_std > max(3.0, self.threshold * 0.75)
            or abs(gia_toc) > 0.12
        )

        if phan_du_tuyet_doi > nguong_hien_tai:
            if dau_hien_tai == self.dau_nhieu_truoc_do or self.dau_nhieu_truoc_do == 0:
                self.so_nhip_nhieu_lon += 1
            else:
                self.so_nhip_nhieu_lon = 1
        else:
            self.so_nhip_nhieu_lon = 0

        self.dau_nhieu_truoc_do = dau_hien_tai

        trend_drop_candidate = (
            phan_du < -nguong_hien_tai
            and noise_context
            and self.so_nhip_nhieu_lon >= 2
        )

        if self.so_nhip_nhieu_lon >= self.persistence_required:
            self.x = z
            self.P = 4.0
            self.so_nhip_nhieu_lon = 0
            self.dau_nhieu_truoc_do = 0
            return self.x
            
        elif self.so_nhip_nhieu_lon > 0 and not trend_drop_candidate:
            return self.x
            
        else:
            R_thich_nghi = self.r_base
            Q_thich_nghi = self.Q * max(ty_le_dt, 0.1)

            stable_sensor_context = (
                van_toc <= 3.0
                and rolling_std <= max(3.0, self.threshold * 0.75)
                and abs(gia_toc) <= 0.12
            )
            if stable_sensor_context:
                R_thich_nghi = max(3.0, R_thich_nghi * 0.35)
                Q_thich_nghi *= 4.0
            elif trend_drop_candidate:
                R_thich_nghi = max(4.0, R_thich_nghi * 0.55)
                Q_thich_nghi *= 8.0
            
            # Phân dải Vận Tốc (Speed Buckets) thay vì tuyến tính
            if van_toc > 0:
                if van_toc <= 5.0:
                    pass # Đã được ưu tiên giảm R ở stable_sensor_context nếu dao động thấp
                elif van_toc <= 40.0:
                    R_thich_nghi *= 2.0  # Đi phố, stop-and-go -> Sóng sánh nhiều
                elif van_toc <= 70.0:
                    R_thich_nghi *= 1.5  # Tốc độ vừa, đường thoáng -> Ít sóng sánh hơn đi phố
                else:
                    R_thich_nghi *= 2.5  # Tốc độ cao (>70) -> Rung xóc và sức cản gió lớn

            if abs(gia_toc) > 0.1:
                R_thich_nghi *= 3.0

            P_du_doan = self.P + Q_thich_nghi
            K = P_du_doan / (P_du_doan + R_thich_nghi)
            
            phan_du_z = z - self.x
            self.x = self.x + K * phan_du_z
            self.x = max(0.0, self.x)
            self.P = (1 - K) * P_du_doan

            return self.x


def is_valid_measurement(fuel, feature_status="") -> bool:
    fuel_val = pd.to_numeric(fuel, errors="coerce")
    if pd.isna(fuel_val) or fuel_val <= 0:
        return False
        
    status = str(feature_status).strip().upper()
    invalid_statuses = {"FUEL_ZERO", "PREVIOUS_INVALID", "INVALID"}
    return status not in invalid_statuses


def chay_kalman_thich_nghi_1d(
    df: pd.DataFrame,
    capacity: float = 200.0,
    nhieu_qua_trinh: float = 0.2,
    r_co_ban: float = 9.0,
    nhip_cho_xac_nhan: int = 3,
) -> pd.Series:
    """
    Chạy bộ lọc BoLocKalmanThichNghi1D trên DataFrame theo từng SegmentID.
    Trả về pd.Series cùng index với df.
    """
    if df.empty:
        return pd.Series(dtype=float, index=df.index)

    df_work = df.copy()
    original_index = df.index

    if "TimeGapMinutes" in df_work.columns:
        time_gaps = pd.to_numeric(df_work["TimeGapMinutes"], errors="coerce").dropna().to_numpy()
        valid_gaps = time_gaps[time_gaps > 0]
        thoi_gian_chuan_phut = float(np.median(valid_gaps)) if len(valid_gaps) > 0 else 5.0
    elif "FuelTime" in df_work.columns:
        f_times = pd.to_datetime(df_work["FuelTime"], errors="coerce")
        diff_s = df_work.groupby("SegmentID", dropna=False)["FuelTime"].diff().dt.total_seconds() / 60.0
        valid_diff = diff_s[diff_s > 0].dropna()
        thoi_gian_chuan_phut = float(valid_diff.median()) if len(valid_diff) > 0 else 5.0
    else:
        thoi_gian_chuan_phut = 5.0

    # Gia tốc
    if "Acceleration" in df_work.columns:
        acceleration = pd.to_numeric(df_work["Acceleration"], errors="coerce").fillna(0.0)
    elif "Speed" in df_work.columns:
        if "TimeGapMinutes" in df_work.columns:
            time_gap_sec = pd.to_numeric(df_work["TimeGapMinutes"], errors="coerce").fillna(thoi_gian_chuan_phut).replace(0, 1.0).clip(lower=1.0) * 60.0
        else:
            time_gap_sec = pd.Series(thoi_gian_chuan_phut * 60.0, index=df_work.index)
        safe_time_gap = time_gap_sec.replace(0, 1.0).clip(lower=1.0)
        acceleration = (df_work.groupby("SegmentID", dropna=False)["Speed"].diff().fillna(0.0) / 3.6) / safe_time_gap
    else:
        acceleration = pd.Series(0.0, index=df_work.index)

    # RollingStd
    if "RollingStd" in df_work.columns:
        rolling_std_col = pd.to_numeric(df_work["RollingStd"], errors="coerce").fillna(0.0)
    else:
        rolling_std_col = (
            df_work.groupby("SegmentID", dropna=False)["FuelLevel"]
            .transform(lambda values: values.rolling(12, min_periods=2).std(ddof=0))
            .fillna(0.0)
        )

    out_series = pd.Series(np.nan, index=original_index, dtype=float)

    for ma_doan, nhom in df_work.groupby("SegmentID", sort=False, dropna=False):
        kf = None
        khoang_thoi_gian_tich_luy = 0.0
        for dong in nhom.itertuples():
            vi_tri = dong.Index
            gap = getattr(dong, "TimeGapMinutes", thoi_gian_chuan_phut)
            if pd.isna(gap) or gap <= 0:
                gap = thoi_gian_chuan_phut

            fuel = getattr(dong, "FuelLevel", None)
            feat_status = getattr(dong, "FeatureStatus", "")
            if not is_valid_measurement(fuel, feat_status):
                khoang_thoi_gian_tich_luy += gap
                out_series.loc[vi_tri] = np.nan
                continue

            gia_tri_do = float(fuel)
            khoang_thoi_gian = gap + khoang_thoi_gian_tich_luy
            khoang_thoi_gian_tich_luy = 0.0
            ty_le_dt = khoang_thoi_gian / thoi_gian_chuan_phut

            trang_thai_chuyen_dong_thay_the = str(getattr(dong, "MovementState", "Moving")).strip().upper()
            trang_thai_chuyen_dong_int = 0 if trang_thai_chuyen_dong_thay_the == "STOPPED" else 1

            gia_toc_hien_tai = float(acceleration.loc[vi_tri]) if vi_tri in acceleration.index else 0.0
            rolling_std_hien_tai = float(rolling_std_col.loc[vi_tri]) if vi_tri in rolling_std_col.index else 0.0
            van_toc_hien_tai = float(getattr(dong, "Speed", 0.0))
            if pd.isna(van_toc_hien_tai):
                van_toc_hien_tai = 0.0

            if kf is None:
                kf = BoLocKalmanThichNghi1D(
                    trang_thai_ban_dau=gia_tri_do,
                    capacity=capacity,
                    sai_so_uoc_luong_ban_dau=4.0,
                    nhieu_qua_trinh=nhieu_qua_trinh,
                    r_co_ban=r_co_ban,
                    nhip_cho_xac_nhan=nhip_cho_xac_nhan,
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

            out_series.loc[vi_tri] = gia_tri_da_loc

    return out_series


def chay_kalman_thich_nghi_cho_tat_ca_xe(
    mau_ten_file: str = "CarFuelHistory_Processed_*.csv",
) -> None:
    import glob
    import os
    import matplotlib.pyplot as plt

    print("=== BẮT ĐẦU CHẠY ADAPTIVE KALMAN FILTER (INNOVATION GATING) ===")
    danh_sach_file = sorted(glob.glob(mau_ten_file))
    if not danh_sach_file:
        return
    
    da_ve_bieu_do = False

    for duong_dan_file in danh_sach_file:
        ma_xe = os.path.basename(duong_dan_file).replace("CarFuelHistory_Processed_", "").replace(".csv", "")
        print(f"Đang chạy Adaptive Kalman cho xe {ma_xe}...")

        df = pd.read_csv(duong_dan_file)
        
        df["FuelTime"] = pd.to_datetime(df["FuelTime"], errors="coerce")
        df["FuelLevel"] = pd.to_numeric(df["FuelLevel"], errors="coerce")
        df["_OriginalOrder"] = np.arange(len(df))
        df = df.sort_values(["SegmentID", "FuelTime", "_OriginalOrder"], kind="stable")
        
        df["Kalman_Adaptive"] = np.nan
        
        if "TimeGapMinutes" in df.columns:
            time_gaps = pd.to_numeric(df["TimeGapMinutes"], errors="coerce").dropna().to_numpy()
            valid_gaps = time_gaps[time_gaps > 0]
            thoi_gian_chuan_phut = float(np.median(valid_gaps)) if len(valid_gaps) > 0 else 5.0
        else:
            thoi_gian_chuan_phut = 5.0

        if "Speed" in df.columns:
            time_gap_sec = df["TimeGapMinutes"].fillna(thoi_gian_chuan_phut) * 60.0
            safe_time_gap = time_gap_sec.replace(0, 1.0)
            df["Acceleration"] = (df.groupby("SegmentID", dropna=False)["Speed"].diff().fillna(0.0) / 3.6) / safe_time_gap
        else:
            df["Acceleration"] = 0.0

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

                van_toc_hien_tai = float(getattr(dong, "Speed", 0.0))
                if pd.isna(van_toc_hien_tai):
                    van_toc_hien_tai = 0.0

                if kf is None:
                    kf = BoLocKalmanThichNghi1D(
                        trang_thai_ban_dau=gia_tri_do,
                        capacity=200.0,
                        sai_so_uoc_luong_ban_dau=4.0,
                        nhieu_qua_trinh=0.2,
                        r_co_ban=9.0,
                        nhip_cho_xac_nhan=3
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
        print(f"Hoàn tất {ma_xe}! Đã xuất: {duong_dan_xuat}")

        if not da_ve_bieu_do:
            plt.figure(figsize=(12, 6))
            plt.plot(df['FuelTime'], df['FuelLevel'], label='Xăng Gốc (Thô)', color='red', alpha=0.5)
            plt.plot(df['FuelTime'], df['Kalman_Adaptive'], label='Adaptive Kalman (Khử nhiễu)', color='blue', linewidth=2)
            plt.title(f"So sánh Xăng Gốc và Adaptive Kalman - Xe {ma_xe}")
            plt.xlabel("Thời gian")
            plt.ylabel("Lít")
            plt.legend()
            plt.grid(True)
            plt.show()
            da_ve_bieu_do = True

    print("=== ĐÃ CHẠY XONG ADAPTIVE KALMAN (GATING) ===")
