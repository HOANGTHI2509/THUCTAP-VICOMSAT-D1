from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import asdict, dataclass, field



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

    # Máy trạng thái ứng viên mức tăng (Candidate Level Tracking)
    candidate_anchor: float | None = None               # Mức nền x trước khi bắt đầu sự kiện tăng
    candidate_event_start_time: str | None = None        # Timestamp bắt đầu sự kiện tăng đầu tiên
    candidate_plateau_start_time: str | None = None      # Timestamp bắt đầu mặt bằng hiện tại
    candidate_plateau_center: float | None = None        # Mức tâm tham chiếu của mặt bằng (để chống trôi dốc)
    candidate_level: float | None = None                # Mức đại diện hiện tại (median của samples)
    candidate_count: int = 0                            # Số mẫu hợp lệ trong mặt bằng hiện tại
    candidate_samples: list[float] = field(default_factory=list) # Danh sách mẫu hợp lệ của mặt bằng
    candidate_sample_times: list[str | None] = field(default_factory=list) # Danh sách timestamp của từng mẫu trong mặt bằng
    candidate_sample_ai: list[bool] = field(default_factory=list) # Nhãn AI của từng mẫu trong mặt bằng
    candidate_ai_confirm_count: int = 0                 # Số mẫu có ai_state == UPWARD_SHIFT trong mặt bằng
    candidate_step_count: int = 0                       # Số bậc tăng trong sự kiện
    candidate_recovery_count: int = 0                   # Số mẫu hồi phục về gần nền cũ (để hủy sau 2 mẫu)
    candidate_confirmed_level: float | None = None      # Mức đã xác nhận gần nhất (khống chế ân hạn)
    dt_expected_minutes: float = 2.0                    # Chu kỳ kỳ vọng giữa các bản tin hợp lệ

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


def _clear_pending_candidate(state: RealtimeAdaptiveKalmanState) -> None:
    """Xóa sạch trạng thái ứng viên đang chờ xác nhận, giữ nguyên candidate_confirmed_level."""
    state.candidate_anchor = None
    state.candidate_event_start_time = None
    state.candidate_plateau_start_time = None
    state.candidate_plateau_center = None
    state.candidate_level = None
    state.candidate_count = 0
    state.candidate_samples.clear()
    state.candidate_sample_times.clear()
    state.candidate_sample_ai.clear()
    state.candidate_ai_confirm_count = 0
    state.candidate_step_count = 0
    state.candidate_recovery_count = 0


def _reset_plateau(
    state: RealtimeAdaptiveKalmanState,
    z: float,
    current_time: pd.Timestamp | None,
    ai_state: str,
) -> None:
    """Khởi động lại một mặt bằng đo lường mới (khi bắt đầu hoặc khi dời mặt bằng)."""
    time_str = current_time.isoformat() if current_time is not None else None
    state.candidate_plateau_start_time = time_str
    state.candidate_plateau_center = float(z)
    state.candidate_level = float(z)
    state.candidate_samples = [float(z)]
    state.candidate_sample_times = [time_str]
    is_ai = (ai_state == "UPWARD_SHIFT")
    state.candidate_sample_ai = [is_ai]
    state.candidate_count = 1
    state.candidate_ai_confirm_count = 1 if is_ai else 0
    state.candidate_recovery_count = 0


def _add_sample_to_plateau(
    state: RealtimeAdaptiveKalmanState,
    z: float,
    current_time: pd.Timestamp | None,
    ai_state: str,
    max_samples: int = 15,
) -> None:
    """Bổ sung một mẫu vào mặt bằng hiện tại; cắt đuôi FIFO để khống chế bộ nhớ và cập nhật lại metrics."""
    time_str = current_time.isoformat() if current_time is not None else None
    is_ai = (ai_state == "UPWARD_SHIFT")
    state.candidate_samples.append(float(z))
    state.candidate_sample_times.append(time_str)
    state.candidate_sample_ai.append(is_ai)

    if len(state.candidate_samples) > max_samples:
        state.candidate_samples.pop(0)
        state.candidate_sample_times.pop(0)
        state.candidate_sample_ai.pop(0)
        if state.candidate_sample_times and state.candidate_sample_times[0] is not None:
            state.candidate_plateau_start_time = state.candidate_sample_times[0]

    state.candidate_count = len(state.candidate_samples)
    state.candidate_ai_confirm_count = sum(state.candidate_sample_ai)
    state.candidate_level = float(np.median(state.candidate_samples))



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
    _clear_pending_candidate(state)
    state.candidate_confirmed_level = None


def _capture_trace(
    collector: list | None,
    idx: int,
    fuel_time: pd.Timestamp | None,
    gap_minutes: float,
    raw_z: float,
    speed: float,
    ai_state: str,
    confidence: float,
    x_before: float | None,
    x_after_rule: float | None,
    x_after: float | None,
    P_before: float,
    P_after: float,
    branch_selected: str,
    update_mode: str,
    R_selected: float | None = None,
    Q_selected: float | None = None,
    R_effective: float | None = None,
    Q_effective: float | None = None,
    K: float | None = None,
    adapt_action: str = "none",
    fast_step: bool = False,
    fast_step_th1: float = 0.0,
    fast_step_th2: float = 0.0,
    sustained_refuel: bool = False,
    labeled_refuel: bool = False,
    rise_count: int = 0,
    drop_count: int = 0,
    drain_count: int = 0,
    recent_refuel_steps: int = 0,
    dropout_count: int = 0,
    capacity_est: float = 200.0,
    flat_jitter: float = 0.8,
    event_threshold: float = 5.0,
) -> None:
    if collector is None:
        return
    collector.append({
        "idx": int(idx),
        "fuel_time": fuel_time.isoformat() if fuel_time is not None else None,
        "gap_minutes": round(float(gap_minutes), 3),
        "raw_z": round(float(raw_z), 2) if not pd.isna(raw_z) else None,
        "speed": round(float(speed), 1),
        "ai_state": str(ai_state),
        "confidence": round(float(confidence), 4),
        "x_before": round(float(x_before), 2) if x_before is not None and not pd.isna(x_before) else None,
        "x_after_rule": round(float(x_after_rule), 2) if x_after_rule is not None and not pd.isna(x_after_rule) else None,
        "x_after": round(float(x_after), 2) if x_after is not None and not pd.isna(x_after) else None,
        "P_before": round(float(P_before), 4),
        "P_after": round(float(P_after), 4),
        "branch_selected": str(branch_selected),
        "update_mode": str(update_mode),  # "direct", "kalman", "frozen"
        "R_selected": round(float(R_selected), 2) if R_selected is not None else None,
        "Q_selected": round(float(Q_selected), 4) if Q_selected is not None else None,
        "R_effective": round(float(R_effective), 2) if R_effective is not None else None,
        "Q_effective": round(float(Q_effective), 4) if Q_effective is not None else None,
        "K": round(float(K), 4) if K is not None else None,
        "adapt_action": str(adapt_action),
        "fast_step": bool(fast_step),
        "fast_step_th1": round(float(fast_step_th1), 2),
        "fast_step_th2": round(float(fast_step_th2), 2),
        "sustained_refuel": bool(sustained_refuel),
        "labeled_refuel": bool(labeled_refuel),
        "rise_count": int(rise_count),
        "drop_count": int(drop_count),
        "drain_count": int(drain_count),
        "recent_refuel_steps": int(recent_refuel_steps),
        "dropout_count": int(dropout_count),
        "capacity_est": round(float(capacity_est), 1),
        "flat_jitter": round(float(flat_jitter), 2),
        "event_threshold": round(float(event_threshold), 2),
    })


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
    trace_collector = config.get("trace_collector", None)
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

    # Bộ tham số Q/R cho từng trạng thái tín hiệu (hỗ trợ cả nhãn mới và cấu hình dashboard).
    cfg_unk_r, cfg_unk_q = config.get("UNKNOWN", (25.0, 0.15))
    cfg_ref_r, cfg_ref_q = config.get("UPWARD_SHIFT", config.get("REFUEL", (1.0, 5.0)))
    cfg_slosh_r, cfg_slosh_q = config.get("OSCILLATION_NOISE", config.get("SLOSHING", (1000.0, 0.001)))
    cfg_cons_r, cfg_cons_q = config.get("GRADUAL_CHANGE", config.get("CONSUMPTION", (25.0, 0.15)))
    cfg_drain_r, cfg_drain_q = config.get("DOWNWARD_SHIFT", config.get("DRAIN", (5.0, 2.0)))
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

    # Duyệt tuần tự từng bản tin theo đúng thứ tự thời gian; mọi quyết định chỉ dùng hiện tại + quá khứ.
    for i in range(len(group)):
        x_before_step = x
        P_before_step = P
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
            x_before_step = np.nan
            P_before_step = 4.0

        # Cập nhật chu kỳ lấy mẫu kỳ vọng dt_expected (chỉ cập nhật khi gap nằm trong khoảng bình thường 0.2 - 10.0 phút)
        # Khoảng trống lớn đơn lẻ (> 10 phút) KHÔNG được phép nâng dt_expected.
        if 0.2 <= gap_minutes <= 10.0:
            stream_state.dt_expected_minutes = 0.85 * stream_state.dt_expected_minutes + 0.15 * gap_minutes
        cfg_dt = config.get("dt_expected_minutes", config.get("expected_dt_minutes"))
        dt_expected = float(cfg_dt) if cfg_dt is not None else stream_state.dt_expected_minutes
        max_plateau_gap = max(2.5 * dt_expected, 7.5)

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
                    _capture_trace(trace_collector, i, current_time, gap_minutes, z, current_speed, ai_state, float(confidence_values[i]), x_before_step, None, dropout_anchor, P_before_step, P, "dropout_recovery_hold", "frozen", capacity_est=capacity, flat_jitter=jitter, event_threshold=event, rise_count=rise_count, drop_count=drop_count, drain_count=drain_count, recent_refuel_steps=recent_refuel_steps, dropout_count=dropout_count)
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
                    _capture_trace(trace_collector, i, current_time, gap_minutes, z, current_speed, ai_state, float(confidence_values[i]), x_before_step, None, dropout_anchor, P_before_step, P, "pending_drain_hold", "frozen", capacity_est=capacity, flat_jitter=jitter, event_threshold=event, rise_count=rise_count, drop_count=drop_count, drain_count=drain_count, recent_refuel_steps=recent_refuel_steps, dropout_count=dropout_count)
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
            _clear_pending_candidate(stream_state)
            if not pd.isna(last_valid_x) and last_valid_x > 3.0:
                dropout_anchor = last_valid_x
                dropout_count = 1
                dropout_recovery_count = 0
                dropout_seen_near_zero = observed <= 5.0
                enhanced[i] = last_valid_x  # Giữ phẳng tuyệt đối mức hợp lệ trước đó
                _capture_trace(trace_collector, i, current_time, gap_minutes, z, current_speed, ai_state, float(confidence_values[i]), x_before_step, None, last_valid_x, P_before_step, P, "is_zero_dropout_hold", "frozen", capacity_est=capacity, flat_jitter=jitter, event_threshold=event, rise_count=rise_count, drop_count=drop_count, drain_count=drain_count, recent_refuel_steps=recent_refuel_steps, dropout_count=dropout_count)
                continue
            elif pd.isna(z) or z <= 0:
                # Một đoạn dữ liệu có thể bắt đầu đúng giữa lúc dropout nên chưa có trạng thái lịch sử.
                # Không tự tạo mức 0 L giả; trả NaN/khoảng trống cho tới khi service khôi phục được state của xe
                # hoặc nhận được một measurement hợp lệ mới.
                enhanced[i] = x if not pd.isna(x) else np.nan
                _capture_trace(trace_collector, i, current_time, gap_minutes, z, current_speed, ai_state, float(confidence_values[i]), x_before_step, None, enhanced[i], P_before_step, P, "zero_dropout_no_history", "frozen", capacity_est=capacity, flat_jitter=jitter, event_threshold=event, rise_count=rise_count, drop_count=drop_count, drain_count=drain_count, recent_refuel_steps=recent_refuel_steps, dropout_count=dropout_count)
                continue

        # Khởi tạo điểm hợp lệ đầu tiên
        if pd.isna(x):
            x = float(z)
            last_valid_x = x
            enhanced[i] = x
            previous_raw_2 = previous_raw
            previous_raw = float(z)
            _clear_pending_candidate(stream_state)
            _capture_trace(trace_collector, i, current_time, gap_minutes, z, current_speed, ai_state, float(confidence_values[i]), x_before_step, None, x, P_before_step, P, "init_first_valid", "direct", capacity_est=capacity, flat_jitter=jitter, event_threshold=event, rise_count=rise_count, drop_count=drop_count, drain_count=drain_count, recent_refuel_steps=recent_refuel_steps, dropout_count=dropout_count)
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
            _capture_trace(trace_collector, i, current_time, gap_minutes, z, current_speed, ai_state, float(confidence_values[i]), x_before_step, None, x, P_before_step, P, "is_parked_noisy_freeze", "frozen", capacity_est=capacity, flat_jitter=jitter, event_threshold=event, rise_count=rise_count, drop_count=drop_count, drain_count=drain_count, recent_refuel_steps=recent_refuel_steps, dropout_count=dropout_count)
            continue

        # Đóng băng 1 nhịp nếu nhiễu rung nhỏ khi xe ổn định
        if drop_count == 1 and (x - z) <= jitter * 1.5 and ai_state in {"STABLE_JITTER", "OSCILLATION_NOISE"}:
            enhanced[i] = x
            previous_raw_2 = previous_raw
            previous_raw = float(z)
            _capture_trace(trace_collector, i, current_time, gap_minutes, z, current_speed, ai_state, float(confidence_values[i]), x_before_step, None, x, P_before_step, P, "drop_count_freeze", "frozen", capacity_est=capacity, flat_jitter=jitter, event_threshold=event, rise_count=rise_count, drop_count=drop_count, drain_count=drain_count, recent_refuel_steps=recent_refuel_steps, dropout_count=dropout_count)
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

        # 3. Máy trạng thái ứng viên mức tăng (Candidate Level Tracking)
        # Thay thế hoàn toàn các luật cũ (sustained_refuel, upward_trap_escape, labeled_refuel trực tiếp)
        # Ngưỡng bắt đầu ứng viên
        min_refuel_jump = max(0.025 * capacity, event * 0.6, jitter * 4.0, 5.0)
        tol_plateau = max(jitter * 1.5, 2.5)
        tol_anchor = max(jitter * 2.0, 3.0)

        # 3.1. Kiểm tra gián đoạn thời gian lớn
        if stream_state.candidate_anchor is not None and gap_minutes > max_plateau_gap:
            if z >= stream_state.candidate_anchor + min_refuel_jump and not is_spike:
                _reset_plateau(stream_state, z, current_time, ai_state)
            else:
                _clear_pending_candidate(stream_state)

        # 3.2. Kiểm tra phục hồi về nền hoặc tụt dưới nền
        skip_confirmation_this_step = False
        if stream_state.candidate_anchor is not None:
            if z < stream_state.candidate_anchor - tol_anchor:
                # Tụt sâu dưới nền: hủy ứng viên ngay, nhường đường cho nhánh giảm
                _clear_pending_candidate(stream_state)
                skip_confirmation_this_step = True
            elif abs(z - stream_state.candidate_anchor) <= tol_anchor:
                # Mẫu gần nền đầu tiên: ngắt bằng chứng ổn định của mặt bằng cao
                stream_state.candidate_recovery_count += 1
                stream_state.candidate_samples.clear()
                stream_state.candidate_sample_times.clear()
                stream_state.candidate_sample_ai.clear()
                stream_state.candidate_count = 0
                stream_state.candidate_ai_confirm_count = 0
                stream_state.candidate_plateau_start_time = None
                if stream_state.candidate_recovery_count >= 2:
                    _clear_pending_candidate(stream_state)
                # QUAN TRỌNG: Mẫu đang hồi phục về nền không được đi tiếp đến xác nhận
                skip_confirmation_this_step = True
            else:
                stream_state.candidate_recovery_count = 0

        # 3.3. Quản lý mặt bằng ứng viên (bổ sung mẫu hoặc mở mặt bằng mới)
        if not skip_confirmation_this_step:
            if stream_state.candidate_anchor is None:
                if z >= x + min_refuel_jump and not is_spike:
                    stream_state.candidate_anchor = float(x)
                    stream_state.candidate_event_start_time = current_time.isoformat() if current_time is not None else None
                    stream_state.candidate_step_count = 1
                    _reset_plateau(stream_state, z, current_time, ai_state)
            else:
                center = stream_state.candidate_plateau_center
                if center is None:
                    _reset_plateau(stream_state, z, current_time, ai_state)
                elif abs(z - center) <= tol_plateau:
                    _add_sample_to_plateau(stream_state, z, current_time, ai_state, max_samples=15)
                elif z > center + tol_plateau:
                    stream_state.candidate_step_count += 1
                    _reset_plateau(stream_state, z, current_time, ai_state)
                elif stream_state.candidate_anchor + tol_anchor < z < center - tol_plateau:
                    _reset_plateau(stream_state, z, current_time, ai_state)

        # 3.4. Đánh giá xác nhận dịch mức tăng
        is_refuel = False
        confirmed_target = None
        if (
            not skip_confirmation_this_step
            and stream_state.candidate_anchor is not None
            and stream_state.candidate_count >= 1
            and not is_spike
            and not is_zero_dropout
        ):
            # Cửa sổ xác nhận ổn định gần nhất: xét khoảng gần nhất có giới hạn (tối đa 5 mẫu)
            win_k = min(len(stream_state.candidate_samples), 5)
            win_samples = stream_state.candidate_samples[-win_k:]
            win_times = stream_state.candidate_sample_times[-win_k:]
            win_ai = stream_state.candidate_sample_ai[-win_k:]

            win_duration_min = 0.0
            if current_time is not None and win_times and win_times[0] is not None:
                w_start = _as_timestamp(win_times[0])
                if w_start is not None:
                    win_duration_min = max(0.0, (current_time - w_start).total_seconds() / 60.0)

            win_spread = max(win_samples) - min(win_samples) if win_samples else 0.0
            mid = len(win_samples) // 2
            half1 = win_samples[:mid]
            half2 = win_samples[mid:]
            level_diff_liters = abs(float(np.median(half2)) - float(np.median(half1))) if half1 and half2 else 0.0
            win_level = float(np.median(win_samples))

            # Ngoại lệ thử nghiệm: bước nhảy cực lớn khi xe chạy
            is_large_jump = (
                (win_level - stream_state.candidate_anchor) >= max(0.15 * capacity, 20.0)
            )
            context_valid = is_truly_parked or is_large_jump

            # Ngưỡng chênh lệch và độ dốc (lỏng hơn khi xe nạp lớn rồi chạy trên đường có sóng sánh/tiêu hao)
            max_level_diff_liters = max(jitter * 1.5, 2.5) if is_large_jump else max(jitter * 1.2, 2.0)
            win_slope = 0.0
            if win_duration_min >= 2.0 and len(win_samples) >= 2:
                # Độ dốc giữa hai nửa cửa sổ theo thời gian (L/phút)
                win_slope = level_diff_liters / max(win_duration_min * 0.5, 1.0)
            max_slope = max(jitter * 0.25, 0.45) if is_large_jump else max(jitter * 0.15, 0.30)
            is_flat_slope = (win_slope <= max_slope) and (level_diff_liters <= max_level_diff_liters)

            # Nhánh AI: đồng thuận với mô hình trong cửa sổ gần nhất
            ai_consensus = (ai_state == "UPWARD_SHIFT" and sum(win_ai) >= 2)
            ai_spread_ok = win_spread <= max(jitter * 2.0, 3.5)
            can_confirm_ai = (
                context_valid
                and ai_consensus
                and len(win_samples) >= 3
                and win_duration_min >= 4.0
                and ai_spread_ok
                and is_flat_slope
            )

            # Nhánh rule: mặt bằng độc lập không có AI hỗ trợ trong cửa sổ gần nhất
            rule_spread_ok = win_spread <= (max(jitter * 2.2, 4.0) if is_large_jump else max(jitter * 1.5, 2.5))
            can_confirm_rule = (
                context_valid
                and len(win_samples) >= 4
                and win_duration_min >= 6.0
                and rule_spread_ok
                and is_flat_slope
            )

            if can_confirm_ai or can_confirm_rule:
                is_refuel = True
                confirmed_target = win_level
                stream_state.candidate_confirmed_level = confirmed_target
                recent_refuel_steps = 2  # Tham số thử nghiệm ân hạn 2 mẫu
                _clear_pending_candidate(stream_state)

        # Chọn Q/R theo trạng thái cuối cùng đã được AI + rule/memory xác nhận.
        branch_selected = "unknown"
        x_after_rule = x
        if is_spike:
            R = cfg_spike_r
            Q_current = cfg_spike_q
            jump_to_z = False
            branch_selected = "is_spike"
        elif trap_recovery:
            x_after_rule = x + 0.5 * (float(z) - x)
            x = x_after_rule
            R = cfg_stable_r
            Q_current = 0.5
            jump_to_z = False
            branch_selected = "trap_recovery"
        elif is_refuel:
            R = cfg_ref_r
            Q_current = cfg_ref_q
            recent_refuel_steps = 2
            jump_to_z = True
            branch_selected = "is_refuel"
        elif ai_state == "OSCILLATION_NOISE" or high_noise:
            R = cfg_slosh_r
            Q_current = cfg_slosh_q
            jump_to_z = False
            branch_selected = "oscillation_noise"
        elif drain_count >= 2:
            R = cfg_drain_r
            Q_current = cfg_drain_q
            jump_to_z = False
            branch_selected = "drain_count>=2"
        elif ai_state == "GRADUAL_CHANGE" or not is_truly_parked:
            R = cfg_cons_r
            Q_current = cfg_cons_q
            jump_to_z = False
            branch_selected = "gradual_or_moving"
        else:  # STABLE_JITTER
            very_stable = float(rolling_std[i]) <= max(jitter * 0.8, 1.0)
            R = cfg_stable_r_very if very_stable else cfg_stable_r
            Q_current = cfg_stable_q
            jump_to_z = False
            branch_selected = "stable_jitter"

        R_selected = R
        Q_selected = Q_current
        adapt_action = "none"

        # Thích nghi R và Q tự động theo độ dốc tiêu hao (bám sát dốc mượt mà không trễ)
        if not is_spike and not jump_to_z:
            if z < x - max(jitter * 0.4, 0.3):
                if drain_count >= 2:
                    lag_ratio = max(1.0, (x - z) / jitter)
                    R = max(2.0, R / (lag_ratio ** 1.2))
                    Q_current = max(Q_current, 0.25 * lag_ratio)
                    adapt_action = "slope_lag_drain"
                elif not is_truly_parked or ai_state in {"GRADUAL_CHANGE", "UNKNOWN"}:
                    slope_lag = min(5.0, (x - z) / max(jitter * 0.5, 0.3))
                    R = max(4.0, R / slope_lag)
                    Q_current = max(Q_current, 0.35 * slope_lag)
                    adapt_action = "slope_lag_moving"
            elif z > x + jitter and (ai_state == "UPWARD_SHIFT" or recent_refuel_steps > 0):
                lag_ratio = max(1.0, (z - x) / jitter)
                R = max(1.0, R / (lag_ratio ** 1.5))
                adapt_action = "upward_adapt"

        R_effective = R
        Q_effective = Q_current

        # Cập nhật Kalman
        if jump_to_z:
            if confirmed_target is not None:
                x = float(confirmed_target)
            else:
                x = float(z)
            P = 4.0
            recent_refuel_steps = 2
            update_mode = "direct"
            K_val = None
        else:
            P = P + Q_current
            K = P / (P + R)
            x_new = x + K * (z - x)
            P = (1 - K) * P
            K_val = float(K)

            # KHÓA DÂNG ẢO VẬT LÝ KHI XE ĐANG ĐỖ / SÓNG SÁNH:
            if not is_refuel and recent_refuel_steps == 0 and z > x + jitter:
                x = x  # Khóa phẳng xuyên qua các đợt sóng sánh dâng ảo
                update_mode = "frozen"
                adapt_action = adapt_action + "|hold_flat" if adapt_action != "none" else "hold_flat"
            elif recent_refuel_steps > 0 and stream_state.candidate_confirmed_level is not None and z > stream_state.candidate_confirmed_level + max(jitter * 1.8, 3.0):
                # Gai trong thời gian ân hạn: không cho x leo theo gai
                x = x
                update_mode = "frozen"
                adapt_action = adapt_action + "|grace_spike_hold" if adapt_action != "none" else "grace_spike_hold"
            else:
                x = x_new  # Bám sát mượt mà dốc tiêu hao (không bị bậc thang vuông)
                update_mode = "kalman"

        x = max(0.0, float(x))
        enhanced[i] = x
        _capture_trace(
            trace_collector, i, current_time, gap_minutes, z, current_speed, ai_state, float(confidence_values[i]),
            x_before_step, x_after_rule, x, P_before_step, P,
            branch_selected, update_mode, R_selected, Q_selected, R_effective, Q_effective, K_val, adapt_action,
            False, max(event * 0.6, 5.5), max(event * 0.8, 7.5),
            is_refuel, (ai_state == "UPWARD_SHIFT"),
            rise_count, drop_count, drain_count, recent_refuel_steps, dropout_count,
            capacity, jitter, event
        )
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