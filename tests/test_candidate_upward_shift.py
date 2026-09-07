import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from src.core.filters.ai_enhanced_adaptive_realtime import (
    RealtimeAdaptiveKalmanState,
    filter_ai_enhanced_adaptive_realtime,
)


def create_test_df(
    levels: list[float],
    timestamps: list[datetime],
    speeds: list[float] | None = None,
    ai_states: list[str] | None = None,
    capacity: float = 600.0,
    jitter: float = 2.0,
    event: float = 15.0,
) -> pd.DataFrame:
    n = len(levels)
    if speeds is None:
        speeds = [0.0] * n
    if ai_states is None:
        ai_states = ["STABLE_JITTER"] * n

    return pd.DataFrame({
        "FuelLevel": levels,
        "FuelTime": timestamps,
        "Speed": speeds,
        "DistanceMeters": [0.0] * n,
        "SegmentID": ["seg_1"] * n,
        "AI_State": ai_states,
        "AI_State_Confidence": [0.95] * n,
        "QualityFlag": [0] * n,
        "QualityReason": ["VALID"] * n,
        "capacity_est": [capacity] * n,
        "noise_sigma_liters": [jitter / 2.0] * n,
        "flat_jitter_threshold": [jitter] * n,
        "spike_threshold": [jitter * 2.0] * n,
        "event_threshold": [event] * n,
        "RollingStd": [0.5] * n,
    })


def test_1_partial_step_down():
    """Ca 1: Partial step-down.
    Mặt bằng cao 550 L nhảy xuống 530 L (vẫn cao hơn nền 500 L):
    Reset mặt bằng tại 530 L, giữ anchor = 500 L; xác nhận tại 530 L sau khi ổn định.
    """
    t0 = datetime(2026, 8, 15, 8, 0, 0)
    # Init 500 L (3 điểm), nhảy lên 550 L (2 điểm = 4 min, chưa đủ 6 min để rule confirm),
    # rồi step-down xuống 530 L (4 điểm = 6 min -> đủ điều kiện rule confirm tại 530 L)
    levels = [500.0, 500.0, 500.0, 550.0, 550.0, 530.0, 530.2, 529.8, 530.1]
    times = [t0 + timedelta(minutes=2 * i) for i in range(len(levels))]

    df = create_test_df(levels, times)
    clean_fuels, final_state = filter_ai_enhanced_adaptive_realtime(df, return_state=True)

    # Tại 550 L (chỉ 2 điểm, AI STABLE_JITTER), chưa được xác nhận
    assert clean_fuels[3] < 510.0
    assert clean_fuels[4] < 510.0

    # Tại điểm cuối (sau 4 điểm ở mức ~530 L, thời gian 6 phút từ khi step-down),
    # phải được xác nhận quanh median 530 L
    assert abs(clean_fuels[-1] - 530.0) < 1.0
    assert final_state.candidate_confirmed_level is not None
    assert abs(final_state.candidate_confirmed_level - 530.0) < 1.0


def test_2_large_gap():
    """Ca 2: Large gap (> max_plateau_gap).
    Khoảng trống thời gian lớn không được cho phép xác nhận mặt bằng xuyên qua khoảng trống.
    """
    t0 = datetime(2026, 8, 15, 8, 0, 0)
    # Init 500 L, sau đó nhảy lên 550 L tại t=4 min (mở candidate)
    # Mẫu kế tiếp đến sau 25 phút (> max_plateau_gap ~ 7.5 min)
    times = [
        t0,
        t0 + timedelta(minutes=2),
        t0 + timedelta(minutes=4),  # 550 L (mẫu 1)
        t0 + timedelta(minutes=29), # 550 L (sau 25 min) -> reset plateau start time
        t0 + timedelta(minutes=31), # 550 L (mẫu 2)
    ]
    levels = [500.0, 500.0, 550.0, 550.0, 550.0]

    df = create_test_df(levels, times)
    clean_fuels, final_state = filter_ai_enhanced_adaptive_realtime(df, return_state=True)

    # Không được xác nhận vội ở điểm thứ 4 (t=29) dù raw cao, vì gap 25 min đã reset mặt bằng
    # Tại điểm thứ 5 (t=31), thời lượng mới chỉ là 2 phút (chưa đủ 6 phút cho rule confirm)
    assert clean_fuels[-1] < 510.0
    assert final_state.candidate_confirmed_level is None


def test_3_spike_during_grace_period():
    """Ca 3: Spike during grace period (recent_refuel_steps).
    Sau khi vừa xác nhận 550 L, mẫu trong thời gian ân hạn vọt lên 580 L:
    Không cho x leo theo gai.
    """
    t0 = datetime(2026, 8, 15, 8, 0, 0)
    # 4 mẫu tại 550 L với nhãn UPWARD_SHIFT -> xác nhận tại mẫu thứ 4
    # Mẫu thứ 5 (trong grace period) vọt lên 580 L
    levels = [500.0, 500.0, 550.0, 550.0, 550.0, 580.0]
    times = [t0 + timedelta(minutes=2 * i) for i in range(len(levels))]
    ai_states = ["STABLE_JITTER", "STABLE_JITTER", "UPWARD_SHIFT", "UPWARD_SHIFT", "UPWARD_SHIFT", "OSCILLATION_NOISE"]

    df = create_test_df(levels, times, ai_states=ai_states)
    clean_fuels, final_state = filter_ai_enhanced_adaptive_realtime(df, return_state=True)

    # Điểm 4 xác nhận lên 550 L
    assert abs(clean_fuels[4] - 550.0) < 1.0
    # Điểm 5 (580 L) bị giữ phẳng tại 550 L, không leo lên 580
    assert abs(clean_fuels[5] - 550.0) < 1.0


def test_4_jump_below_baseline():
    """Ca 4: Jump below baseline.
    Đang theo dõi ứng viên tăng 550 L (anchor = 500 L), raw tụt xuống 470 L (< anchor - tol_anchor):
    Hủy ứng viên tăng ngay lập tức.
    """
    t0 = datetime(2026, 8, 15, 8, 0, 0)
    levels = [500.0, 500.0, 550.0, 550.0, 470.0]
    times = [t0 + timedelta(minutes=2 * i) for i in range(len(levels))]

    df = create_test_df(levels, times)
    clean_fuels, final_state = filter_ai_enhanced_adaptive_realtime(df, return_state=True)

    # Ứng viên phải bị xóa sạch khi raw < anchor - tol_anchor
    assert final_state.candidate_anchor is None
    assert final_state.candidate_count == 0


def test_5_creeping_ramp():
    """Ca 5: Creeping ramp (chống trôi dốc tăng chậm).
    Nền 500 L. Bước nhảy ban đầu vượt min_refuel_jump (515 L) để mở ứng viên,
    nhưng sau đó bò dốc liên tục mỗi mẫu tăng 4 L (515 -> 519 -> 523 -> 527 -> 531).
    Hệ thống không được xác nhận mặt bằng tăng.
    """
    t0 = datetime(2026, 8, 15, 8, 0, 0)
    levels = [500.0, 500.0, 515.0, 519.0, 523.0, 527.0, 531.0, 535.0]
    times = [t0 + timedelta(minutes=2 * i) for i in range(len(levels))]

    df = create_test_df(levels, times)
    clean_fuels, final_state = filter_ai_enhanced_adaptive_realtime(df, return_state=True)

    # Do dốc liên tục vượt tol_plateau và không bao giờ ổn định dốc phẳng, không được confirm
    assert final_state.candidate_confirmed_level is None
    # x không bị nhảy trực tiếp lên đỉnh dốc
    assert clean_fuels[-1] < 510.0


def test_6_dropout_interruption():
    """Ca 6: Dropout interruption.
    Đang theo dõi ứng viên tại 550 L, gặp zero dropout (raw = 0.0):
    Ứng viên bị xóa ngay lập tức, x giữ phẳng tại last_valid_x.
    """
    t0 = datetime(2026, 8, 15, 8, 0, 0)
    levels = [500.0, 500.0, 550.0, 550.0, 0.0, 500.0]
    times = [t0 + timedelta(minutes=2 * i) for i in range(len(levels))]

    df = create_test_df(levels, times)
    clean_fuels, final_state = filter_ai_enhanced_adaptive_realtime(df, return_state=True)

    # Tại điểm dropout (index 4), x giữ ở 500 L
    assert abs(clean_fuels[4] - 500.0) < 1.0
    # Ứng viên bị xóa sạch
    assert final_state.candidate_anchor is None
    assert final_state.candidate_count == 0


def test_7_jittery_300s_cycle():
    """Ca 7: Jittery 300s-cycle handling.
    Chu kỳ lấy mẫu 300s (5 phút) có biến thiên nhẹ:
    dt_expected thích nghi, gap 5-6 phút không bị ngắt quãng, xác nhận diễn ra bình thường.
    """
    t0 = datetime(2026, 8, 15, 8, 0, 0)
    # Khoảng cách 5 phút mỗi mẫu
    times = [
        t0,
        t0 + timedelta(seconds=300),
        t0 + timedelta(seconds=610),   # +310s (mở candidate tại 550 L)
        t0 + timedelta(seconds=920),   # +310s (mẫu 2)
        t0 + timedelta(seconds=1210),  # +290s (mẫu 3)
        t0 + timedelta(seconds=1520),  # +310s (mẫu 4: duration = 15.1 min > 6.0 min)
    ]
    levels = [500.0, 500.0, 550.0, 550.2, 549.8, 550.1]

    df = create_test_df(levels, times)
    clean_fuels, final_state = filter_ai_enhanced_adaptive_realtime(df, return_state=True)

    # Xác nhận thành công tại mẫu thứ 6 (4 mẫu tại 550 L, duration ~15 phút)
    assert abs(clean_fuels[-1] - 550.0) < 1.0
    assert final_state.candidate_confirmed_level is not None


def test_8_case1_hill_elimination():
    """Ca 8: Xe 21H-03052 loại bỏ hoàn toàn đồi giả ~72 L tại 10:54:00."""
    from src.service.state_manager import StreamingStateManager
    df_raw = pd.read_csv("TienXuLy/21H-03052_processed.csv")
    df_raw["FuelTime"] = pd.to_datetime(df_raw["FuelTime"])
    df_baseline = pd.read_csv("artifacts/baseline_results/case1_hill_21H-03052_trace.csv")

    mask = (df_raw["FuelTime"] >= "2026-08-14 10:08:00") & (df_raw["FuelTime"] <= "2026-08-14 11:56:00")
    matched = df_raw[mask]

    manager = StreamingStateManager()
    trace = []
    for _, row in matched.iterrows():
        manager.process_point(
            vehicle_id="21H-03052",
            fuel_time=row["FuelTime"],
            fuel_level=row["FuelLevel"],
            speed=row["Speed"],
            capacity_est=df_baseline["capacity_est"].iloc[0],
            noise_sigma_liters=df_baseline["flat_jitter"].iloc[0] / 2.5,
            trace_collector=trace,
        )
    df_new = pd.DataFrame(trace)
    old_x = df_baseline["x_after"].to_numpy()
    new_x = df_new["x_after"].to_numpy()

    # Tại bước nhảy 10:54:00 (index 23 trong trace), bản cũ nhảy lên 847.5 L (+71.99 L đồi giả)
    # Bản mới phải giữ phẳng ở mức nền quanh 775.5 L (chênh lệch > 65 L so với đồi giả cũ)
    assert abs(new_x[23] - 775.51) < 1.0
    assert old_x[23] - new_x[23] > 70.0

    # Đối chiếu cụm đo cuối, không ép nhánh phục hồi mới khớp đường lọc cũ.
    tail_level = float(matched["FuelLevel"].tail(5).median())
    assert abs(new_x[-1] - tail_level) <= max(float(df_baseline["flat_jitter"].iloc[0]), 1.0)


def test_9_baseline_cases_exact_parity():
    """Ca 9: Giữ parity tiêu hao/300s; kiểm tra mức ở ca sóng sánh đã sửa."""
    from src.service.state_manager import StreamingStateManager

    cases = [
        ("21H-03221", "TienXuLy/21H-03221_processed.csv", "artifacts/baseline_results/case2_sloshing_21H-03221_trace.csv", "2026-08-10 06:06:00", "2026-08-10 08:25:00"),
        ("29E-45520", "TienXuLy/29E-45520_processed.csv", "artifacts/baseline_results/case3_consumption_29E-45520_trace.csv", "2026-08-12 08:20:00", "2026-08-12 13:15:00"),
        ("20B-27762", "TienXuLy/20B-27762_processed.csv", "artifacts/baseline_results/case4_dt300s_20B-27762_trace.csv", "2026-08-10 16:30:00", "2026-08-10 20:35:00"),
    ]

    for v_id, raw_path, base_path, start_t, end_t in cases:
        df_raw = pd.read_csv(raw_path)
        df_raw["FuelTime"] = pd.to_datetime(df_raw["FuelTime"])
        df_baseline = pd.read_csv(base_path)
        mask = (df_raw["FuelTime"] >= start_t) & (df_raw["FuelTime"] <= end_t)
        matched = df_raw[mask]

        manager = StreamingStateManager()
        trace = []
        for _, row in matched.iterrows():
            manager.process_point(
                vehicle_id=v_id,
                fuel_time=row["FuelTime"],
                fuel_level=row["FuelLevel"],
                speed=row["Speed"],
                capacity_est=df_baseline["capacity_est"].iloc[0],
                noise_sigma_liters=df_baseline["flat_jitter"].iloc[0] / 2.5,
                trace_collector=trace,
            )
        df_new = pd.DataFrame(trace)
        old_x = df_baseline["x_after"].to_numpy()
        new_x = df_new["x_after"].to_numpy()
        max_diff = np.max(np.abs(new_x - old_x))
        if v_id == "21H-03221":
            # Chính nhánh sóng sánh đang được sửa: kiểm tra giới hạn mức và
            # bám cụm cuối, thay vì bắt buộc tái tạo lỗi bám đáy của baseline.
            values = matched["FuelLevel"].to_numpy()
            assert np.isfinite(new_x).all()
            assert new_x.min() >= values.min()
            assert new_x.max() <= values.max()
            tail_level = float(np.median(values[-5:]))
            assert abs(new_x[-1] - tail_level) <= max(float(df_baseline["flat_jitter"].iloc[0]), 2.0)
        else:
            assert max_diff < 1e-4, f"Parity regression on {v_id}: max_diff = {max_diff}"


def test_10_sustained_refuel_confirmed():
    """Ca 10: Tăng bền vững (nạp nhiên liệu thật).
    Từ nền 400 L nạp lên 550 L (+150 L). Sau khi duy trì đủ 4 mẫu và 6 phút,
    bộ lọc phải xác nhận thành công và nhảy lên mức ~550 L, không bị kẹt thấp.
    """
    t0 = datetime(2026, 8, 15, 8, 0, 0)
    levels = [400.0, 400.0, 400.0, 550.0, 550.1, 549.9, 550.0, 550.2]
    times = [t0 + timedelta(minutes=2 * i) for i in range(len(levels))]

    df = create_test_df(levels, times)
    clean_fuels, final_state = filter_ai_enhanced_adaptive_realtime(df, return_state=True)

    # Trước khi xác nhận (index 3), x chưa nhảy
    assert clean_fuels[3] < 410.0
    # Sau khi đủ 4 mẫu ổn định (index 6, 7), x phải được xác nhận quanh 550 L
    assert abs(clean_fuels[-1] - 550.0) < 1.0
    assert final_state.candidate_confirmed_level is not None
    assert abs(final_state.candidate_confirmed_level - 550.0) < 1.0


def test_11_multi_step_upward_shift():
    """Ca 11: Tăng nhiều bậc (nạp 2 giai đoạn / ngắt quãng).
    Bậc 1: Nền 400 L -> 480 L, xác nhận tại 480 L.
    Sau khi hết thời gian ân hạn, bậc 2: 480 L -> 560 L, tiếp tục mở ứng viên và xác nhận tại 560 L.
    """
    t0 = datetime(2026, 8, 15, 8, 0, 0)
    # Bậc 1: 400 L (3 điểm) -> 480 L (4 điểm: index 3, 4, 5, 6 xác nhận tại 6)
    # Ân hạn: 480 L (2 điểm: index 7, 8)
    # Bậc 2: 560 L (4 điểm: index 9, 10, 11, 12 xác nhận tại 12)
    levels = [
        400.0, 400.0, 400.0,
        480.0, 480.1, 479.9, 480.0,
        480.0, 480.0,
        560.0, 560.2, 559.8, 560.1
    ]
    times = [t0 + timedelta(minutes=2 * i) for i in range(len(levels))]

    df = create_test_df(levels, times)
    clean_fuels, final_state = filter_ai_enhanced_adaptive_realtime(df, return_state=True)

    # Tại cuối bậc 1 (index 6, 7, 8), x đã lên 480 L
    assert abs(clean_fuels[6] - 480.0) < 1.0
    assert abs(clean_fuels[8] - 480.0) < 1.0

    # Tại cuối bậc 2 (index 12), x tiếp tục xác nhận thành công lên 560 L
    assert abs(clean_fuels[-1] - 560.0) < 1.0
    assert final_state.candidate_confirmed_level is not None
    assert abs(final_state.candidate_confirmed_level - 560.0) < 1.0


def test_12_spike_below_baseline_no_candidate():
    """Ca 12: Bật mạnh nhưng vẫn dưới mức nền x (không mở ứng viên tăng sai).
    Nền x = 500 L. Gặp tụt sâu xuống 400 L rồi bật mạnh lên 460 L (bật +60 L nhưng vẫn < 500 L).
    Candidate Level Tracking không được coi đây là bước nhảy tăng để mở ứng viên tăng.
    """
    t0 = datetime(2026, 8, 15, 8, 0, 0)
    levels = [500.0, 500.0, 500.0, 400.0, 460.0, 460.0]
    times = [t0 + timedelta(minutes=2 * i) for i in range(len(levels))]

    df = create_test_df(levels, times)
    clean_fuels, final_state = filter_ai_enhanced_adaptive_realtime(df, return_state=True)

    # Ứng viên tăng không được mở vì 460 L < x (500 L)
    assert final_state.candidate_anchor is None
    assert final_state.candidate_count == 0
    assert final_state.candidate_confirmed_level is None
