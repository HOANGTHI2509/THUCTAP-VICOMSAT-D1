import numpy as np
import pandas as pd
import pytest

from src.core.filters.ai_enhanced_adaptive_realtime import (
    filter_ai_enhanced_adaptive_realtime,
    RealtimeAdaptiveKalmanState,
)


def _make_df(timestamps, raw_levels, speeds=None, ai_states=None, capacity=300.0, jitter=0.8, event=6.0):
    n = len(raw_levels)
    if speeds is None:
        speeds = [0.0] * n
    if ai_states is None:
        ai_states = ["STABLE_JITTER"] * n

    return pd.DataFrame({
        "FuelTime": [t.isoformat() for t in timestamps],
        "FuelLevel": [float(v) for v in raw_levels],
        "Speed": [float(s) for s in speeds],
        "AI_State": [str(s) for s in ai_states],
        "RollingStd": [jitter * 0.5] * n,
        "capacity_est": [capacity] * n,
        "flat_jitter_threshold": [jitter] * n,
        "event_threshold": [event] * n,
    })


def test_scenario_1_full_recovery():
    """Tình huống 1: Hồi phục hoàn toàn về nền cũ."""
    base_time = pd.Timestamp("2026-03-01 10:00:00")
    # Giai đoạn 1: 5 mẫu ổn định ở 220 L
    # Giai đoạn 2: 3 mẫu tụt sâu xuống đáy U 170 L
    # Giai đoạn 3: 8 mẫu hồi phục về cụm 219.5 - 220.5 L
    raws = [220.0] * 5 + [170.0, 169.5, 170.2] + [219.8, 220.2, 220.0, 219.9, 220.1, 220.0, 220.0, 220.0]
    ai = ["STABLE_JITTER"] * 5 + ["OSCILLATION_NOISE"] * 3 + ["OSCILLATION_NOISE"] * 8
    speeds = [0.0] * 5 + [0.0] * 3 + [35.0] * 8  # xe bắt đầu chạy
    times = [base_time + pd.Timedelta(minutes=2 * i) for i in range(len(raws))]

    df = _make_df(times, raws, speeds=speeds, ai_states=ai)
    trace = []
    clean = filter_ai_enhanced_adaptive_realtime(df, config={"trace_collector": trace})
    df_t = pd.DataFrame(trace)

    # 1. Tại đáy U (mẫu 5-7): output không bị lôi tụt xuống 170 L
    for idx in [5, 6, 7]:
        assert clean[idx] > 215.0, f"Output at dip idx {idx} fell too low: {clean[idx]}"

    # 2. Nhánh noise_recovery được kích hoạt sau khi rời đáy và hình thành cụm
    rec_rows = df_t[df_t["branch_selected"] == "noise_recovery"]
    assert len(rec_rows) >= 1, "noise_recovery branch was never activated"

    # 3. Output cuối cùng bám sát mức phục hồi 220 L
    assert abs(clean[-1] - 220.0) <= 1.0, f"Final output {clean[-1]} not close to 220.0"


def test_scenario_2_partial_recovery():
    """Tình huống 2: Hồi phục một phần (mức hồi phục thấp hơn nền cũ 15 L)."""
    base_time = pd.Timestamp("2026-03-01 10:00:00")
    # Giai đoạn 1: Nền 220 L (5 mẫu)
    # Giai đoạn 2: Tụt xuống đáy 160 L (3 mẫu)
    # Giai đoạn 3: Hồi phục về cụm 205 L (15 mẫu)
    raws = [220.0] * 5 + [160.0, 159.0, 160.5] + [204.8, 205.2, 205.0] + [205.0] * 12
    ai = ["STABLE_JITTER"] * 5 + ["OSCILLATION_NOISE"] * 3 + ["OSCILLATION_NOISE"] * 15
    speeds = [0.0] * 5 + [0.0] * 3 + [40.0] * 15
    times = [base_time + pd.Timedelta(minutes=2 * i) for i in range(len(raws))]

    df = _make_df(times, raws, speeds=speeds, ai_states=ai)
    trace = []
    clean = filter_ai_enhanced_adaptive_realtime(df, config={"trace_collector": trace})
    df_t = pd.DataFrame(trace)

    # 1. Đáy U được bảo vệ không bị tụt theo raw
    for idx in [5, 6, 7]:
        assert clean[idx] > 215.0, f"Dip idx {idx} fell too low: {clean[idx]}"

    # 2. Nhánh noise_recovery được kích hoạt
    rec_rows = df_t[df_t["branch_selected"] == "noise_recovery"]
    assert len(rec_rows) >= 1, "noise_recovery branch was never activated"

    # 3. Output phải thoát khỏi mức nền 220 L và tiến về sát cụm 205 L
    assert clean[-1] < 208.0, f"Output failed to adapt to partial recovery level: {clean[-1]}"
    assert abs(clean[-1] - 205.0) <= 2.0, f"Final output {clean[-1]} not close to 205.0"


def test_scenario_3_prolonged_bottom_no_recovery():
    """Tình huống 3: Đáy U kéo dài (raw nằm lì tại đáy, không kích hoạt phục hồi)."""
    base_time = pd.Timestamp("2026-03-01 10:00:00")
    # Nền 220 L, sau đó tụt xuống 170 L và nằm lì suốt 10 mẫu (20 phút)
    raws = [220.0] * 5 + [170.0, 170.2, 169.8, 170.1, 170.0, 169.9, 170.1, 170.0, 170.2, 170.0]
    ai = ["STABLE_JITTER"] * 5 + ["OSCILLATION_NOISE"] * 10
    times = [base_time + pd.Timedelta(minutes=2 * i) for i in range(len(raws))]

    df = _make_df(times, raws, ai_states=ai)
    trace = []
    clean = filter_ai_enhanced_adaptive_realtime(df, config={"trace_collector": trace})
    df_t = pd.DataFrame(trace)

    # Không được kích hoạt noise_recovery khi raw chưa rời đáy
    rec_rows = df_t[df_t["branch_selected"] == "noise_recovery"]
    assert len(rec_rows) == 0, "noise_recovery should NOT activate while remaining at bottom"


def test_scenario_4_real_drop_amidst_noise():
    """Tình huống 4: Giảm thật trong nhiễu (nhánh drain hoạt động bình thường, không bị recovery can thiệp)."""
    base_time = pd.Timestamp("2026-03-01 10:00:00")
    # Nền 200 L -> giảm liên tục với nhãn DOWNWARD_SHIFT
    raws = [200.0] * 4 + [190.0, 185.0, 180.0, 175.0, 170.0, 165.0]
    ai = ["STABLE_JITTER"] * 4 + ["DOWNWARD_SHIFT"] * 6
    times = [base_time + pd.Timedelta(minutes=2 * i) for i in range(len(raws))]

    df = _make_df(times, raws, ai_states=ai)
    trace = []
    clean = filter_ai_enhanced_adaptive_realtime(df, config={"trace_collector": trace})
    df_t = pd.DataFrame(trace)

    # Nhánh drain phải được kích hoạt và output bám đà giảm
    drain_rows = df_t[df_t["branch_selected"].str.contains("drain", case=False)]
    assert len(drain_rows) >= 1, "drain branch should be selected for real drop"
    assert clean[-1] < 180.0, f"Real drop was impeded: final clean={clean[-1]}"


def test_scenario_5_noise_returns_during_recovery():
    """Tình huống 5: Nhiễu quay lại giữa chừng (ngắt ngay phục hồi khi raw tụt lại đáy)."""
    base_time = pd.Timestamp("2026-03-01 10:00:00")
    # Nền 220 L -> Tụt đáy 165 L -> Bắt đầu hồi lên 210 L (mẫu 8-11) -> Tụt lại 160 L (mẫu 12-14)
    raws = [220.0] * 5 + [165.0, 164.8, 165.2] + [209.8, 210.2, 210.0, 210.1] + [160.0, 159.5, 160.2]
    ai = ["STABLE_JITTER"] * 5 + ["OSCILLATION_NOISE"] * 3 + ["OSCILLATION_NOISE"] * 4 + ["OSCILLATION_NOISE"] * 3
    times = [base_time + pd.Timedelta(minutes=2 * i) for i in range(len(raws))]

    state = RealtimeAdaptiveKalmanState()
    trace = []
    df = _make_df(times, raws, ai_states=ai)
    clean = filter_ai_enhanced_adaptive_realtime(df, config={"trace_collector": trace}, state=state)
    df_t = pd.DataFrame(trace)

    # Khi raw tụt lại về 160 L ở các mẫu cuối:
    # 1. recovery_active phải bị ngắt (False)
    assert not state.recovery_active, "recovery_active should be False after raw falls back to bottom"
    # 2. Output cuối cùng không bị kéo theo sóng mà được giữ bảo vệ bởi oscillation_noise
    assert df_t.iloc[-1]["branch_selected"] == "oscillation_noise"
