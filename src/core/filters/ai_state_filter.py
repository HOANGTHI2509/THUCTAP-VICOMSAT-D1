from __future__ import annotations

import json
import os
import pickle
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.core.filters.fuel_state_filter import build_fuel_state_profile
from src.core.filters.tcn_state_classifier import predict_tcn_fuel_state


TRANSIENT_LABELS = {"TRANSIENT_NOISE", "TRANSIENT_UP_NOISE", "TRANSIENT_DOWN_NOISE", "TRANSIENT_CLUSTER_NOISE"}
TRANSIENT_STATE = "TRANSIENT_NOISE"


@dataclass
class AIStateFilterConfig:
    stable_alpha: float = 0.12
    stable_deadband_ratio: float = 0.25
    consumption_alpha: float = 0.58
    sloshing_alpha: float = 0.08
    refuel_alpha: float = 1.0
    drain_candidate_alpha: float = 0.45
    drain_alpha: float = 0.92
    unknown_alpha: float = 0.25
    stable_recenter_count: int = 4
    trend_follow_count: int = 3
    trend_follow_alpha: float = 0.82
    noisy_consumption_count: int = 2
    noisy_consumption_alpha: float = 0.42
    up_shift_follow_count: int = 4
    up_shift_follow_alpha: float = 0.55
    small_up_shift_alpha: float = 0.88


def load_fuel_state_classifier(model_dir: str = "models/fuel_state_classifier"):
    model_path = os.path.join(model_dir, "fuel_state_classifier.pkl")
    metadata_path = os.path.join(model_dir, "metadata.json")
    if not os.path.exists(model_path) or not os.path.exists(metadata_path):
        return None, None
    try:
        with open(metadata_path, encoding="utf-8") as handle:
            metadata = json.load(handle)
        with open(model_path, "rb") as handle:
            model = pickle.load(handle)
        return model, metadata
    except Exception:
        return None, None


def _numeric_series(df: pd.DataFrame, column: str, default: float = 0.0) -> pd.Series:
    if column in df.columns:
        return pd.to_numeric(df[column], errors="coerce").fillna(default)
    return pd.Series(default, index=df.index, dtype=float)


def _dt_alpha(alpha: float, dt_minutes: float, reference_minutes: float) -> float:
    alpha = float(np.clip(alpha, 0.0, 1.0))
    if alpha <= 0.0 or alpha >= 1.0:
        return alpha
    if reference_minutes <= 0 or pd.isna(reference_minutes):
        reference_minutes = 5.0
    ratio = max(float(dt_minutes) / reference_minutes, 0.1)
    return float(1.0 - (1.0 - alpha) ** ratio)


def _ensure_ai_features(df: pd.DataFrame, profile=None, mode: str = "offline") -> pd.DataFrame:
    result = df.sort_values("FuelTime", kind="stable").copy()
    state_profile = build_fuel_state_profile(result, profile)

    fuel = _numeric_series(result, "FuelLevel", 0.0)
    speed = _numeric_series(result, "Speed", 0.0)
    time_gap = _numeric_series(result, "TimeGapMinutes", 0.0)
    if (time_gap <= 0).all():
        times = pd.to_datetime(result["FuelTime"], errors="coerce")
        time_gap = times.diff().dt.total_seconds().div(60.0).fillna(state_profile.sample_gap_minutes)
        time_gap = time_gap.mask(time_gap <= 0, state_profile.sample_gap_minutes)

    if "DeltaFuel" in result.columns:
        delta = _numeric_series(result, "DeltaFuel", 0.0)
    else:
        delta = fuel.diff().fillna(0.0)

    if "RollingStd" in result.columns:
        rolling_std = _numeric_series(result, "RollingStd", 0.0)
    elif "RollingStd12" in result.columns:
        rolling_std = _numeric_series(result, "RollingStd12", 0.0)
    else:
        rolling_std = fuel.rolling(window=12, min_periods=1).std().fillna(0.0)

    capacity = max(float(state_profile.capacity_est), 1.0)
    noise = max(float(state_profile.noise_sigma_liters), 1e-6)

    result["FuelPct"] = fuel / capacity
    result["TimeGapMinutes"] = time_gap
    result["DeltaFuel"] = delta
    result["DeltaPct"] = delta / capacity
    result["AbsDeltaFuel"] = delta.abs()
    result["DeltaOverNoise"] = delta / noise
    result["RollingStd12"] = rolling_std
    result["RollingStdPct"] = rolling_std / capacity
    result["DistanceMeters"] = _numeric_series(result, "DistanceMeters", 0.0)
    result["GpsSpeedKmh"] = _numeric_series(result, "GpsSpeedKmh", 0.0)
    result["MotionSpeedKmh"] = np.maximum(speed.to_numpy(dtype=float), result["GpsSpeedKmh"].to_numpy(dtype=float))
    result["HasGPS"] = (
        result[["Lat", "Lng"]].notna().all(axis=1).astype(int)
        if {"Lat", "Lng"}.issubset(result.columns)
        else 0
    )
    result["capacity_est"] = capacity
    result["noise_sigma_liters"] = noise
    result["flat_jitter_threshold"] = state_profile.flat_jitter_threshold
    result["spike_threshold"] = state_profile.spike_threshold
    result["event_threshold"] = state_profile.event_threshold
    prev_median3 = fuel.rolling(window=3, min_periods=1).median().shift(1).fillna(fuel)
    if mode == "offline":
        future_median3 = fuel.iloc[::-1].rolling(window=3, min_periods=1).median().iloc[::-1].shift(-1).fillna(fuel)
        future_median5 = fuel.iloc[::-1].rolling(window=5, min_periods=1).median().iloc[::-1].shift(-1).fillna(fuel)
        local_range5 = fuel.rolling(window=5, center=True, min_periods=1).max() - fuel.rolling(window=5, center=True, min_periods=1).min()
        local_range7 = fuel.rolling(window=7, center=True, min_periods=1).max() - fuel.rolling(window=7, center=True, min_periods=1).min()
        next_fuel = fuel.shift(-1).fillna(fuel)
    else:
        future_median3 = prev_median3
        future_median5 = prev_median3
        local_range5 = fuel.rolling(window=5, min_periods=1).max() - fuel.rolling(window=5, min_periods=1).min()
        local_range7 = fuel.rolling(window=7, min_periods=1).max() - fuel.rolling(window=7, min_periods=1).min()
        next_fuel = fuel
    result["PrevMedian3"] = prev_median3
    result["FutureMedian3"] = future_median3
    result["FutureMedian5"] = future_median5
    result["ReturnToPrevLevel"] = (future_median5 - prev_median3).abs()
    result["LocalRange5"] = local_range5.fillna(0.0)
    result["LocalRange7"] = local_range7.fillna(0.0)
    result["PeakReversalFlag"] = ((delta > state_profile.flat_jitter_threshold * 1.5) & ((next_fuel - fuel) < -state_profile.flat_jitter_threshold * 1.5)).astype(int)
    result["ValleyReversalFlag"] = ((delta < -state_profile.flat_jitter_threshold * 1.5) & ((next_fuel - fuel) > state_profile.flat_jitter_threshold * 1.5)).astype(int)
    result["TransientScore"] = (1.0 - result["ReturnToPrevLevel"] / result["LocalRange5"].replace(0, np.nan)).clip(lower=0.0, upper=1.0).fillna(0.0)
    return result


def predict_ai_fuel_state(df: pd.DataFrame, model, metadata: dict | None, profile=None, mode: str = "offline") -> pd.DataFrame:
    result = _ensure_ai_features(df, profile, mode=mode)
    result["AI_State_Raw"] = ""
    result["AI_State"] = ""
    result["AI_State_Confidence"] = np.nan
    if model is None or not metadata:
        result["AI_State"] = "UNKNOWN"
        return result

    feature_columns = metadata.get("feature_columns", [])
    for column in feature_columns:
        if column not in result.columns:
            result[column] = 0.0
        result[column] = pd.to_numeric(result[column], errors="coerce").fillna(0.0)

    x_data = result[feature_columns].to_numpy(dtype=float)
    raw_label = model.predict(x_data)
    result["AI_State_Raw"] = raw_label
    result["AI_State"] = raw_label
    if hasattr(model, "predict_proba"):
        result["AI_State_Confidence"] = model.predict_proba(x_data).max(axis=1)

    return _postprocess_ai_state(result, mode=mode)


def _postprocess_ai_state(df: pd.DataFrame, mode: str = "offline") -> pd.DataFrame:
    if mode == "realtime":
        return df.sort_values("FuelTime", kind="stable").copy() if "FuelTime" in df.columns else df.copy()

    result = df.sort_values("FuelTime", kind="stable").copy()
    if "SegmentID" in result.columns:
        groups = result.groupby("SegmentID", sort=False)
    else:
        groups = [(0, result)]
    for _, group in groups:
        indices = list(group.index)
        for pos, idx in enumerate(indices):
            if pos == 0:
                continue
            row = result.loc[idx]
            prev = result.loc[indices[pos - 1]]
            delta = float(row.get("DeltaFuel", 0.0) or 0.0)
            spike = float(row.get("spike_threshold", 0.0) or 0.0)
            event = float(row.get("event_threshold", 0.0) or 0.0)
            flat = float(row.get("flat_jitter_threshold", 0.0) or 0.0)
            fuel = float(row.get("FuelLevel", 0.0) or 0.0)
            prev_fuel = float(prev.get("FuelLevel", 0.0) or 0.0)
            next_indices = indices[pos + 1 : pos + 4]
            if not next_indices:
                continue
            next_fuels = [float(result.loc[next_idx, "FuelLevel"]) for next_idx in next_indices]
            future_median = float(np.median(next_fuels))
            drop_is_large = delta <= -max(spike * 0.8, event * 0.5, 3.5)
            jump_is_large = delta >= max(spike * 0.8, event * 0.5, 3.5)
            recovers_quickly = abs(future_median - prev_fuel) <= max(flat * 2.5, abs(delta) * 0.35, 3.0)
            single_point_low = min(next_fuels) > fuel + max(flat, abs(delta) * 0.25)
            single_point_high = max(next_fuels) < fuel - max(flat, abs(delta) * 0.25)

            # Sụt 1 điểm rồi phục hồi (Chữ U / Spike down)
            if drop_is_large and (recovers_quickly or single_point_low):
                result.loc[idx, "AI_State"] = "SPIKE"
                continue

            # Nhô vọt 1 điểm rồi rơi lại (Quả đồi 1 điểm / Spike up)
            if jump_is_large and (recovers_quickly or single_point_high):
                result.loc[idx, "AI_State"] = "SPIKE"
                continue

            if row.get("AI_State") == "DRAIN":
                near_low_count = sum(
                    1 for next_fuel in next_fuels if abs(next_fuel - fuel) <= max(flat * 2.0, abs(delta) * 0.25)
                )
                if near_low_count < 2:
                    result.loc[idx, "AI_State"] = "SPIKE"

            # Khắc phục nhãn SLOSHING_NOISE nhầm khi xe đang chạy cao tốc tiêu thụ nhiên liệu
            speed_val = float(row.get("Speed", 0.0) or 0.0)
            if speed_val > 10.0 and row.get("AI_State") == "SLOSHING_NOISE":
                if fuel <= prev_fuel + 0.8 and future_median <= fuel + 1.5:
                    result.loc[idx, "AI_State"] = "CONSUMPTION"

    return result


def _apply_rf_tcn_ensemble(df: pd.DataFrame, tcn_model=None, tcn_metadata: dict | None = None) -> pd.DataFrame:
    result = df.copy()
    result["AI_State_RF"] = result["AI_State"].astype(str)
    result["AI_State_Ensemble"] = result["AI_State_RF"]
    if tcn_model is None or not tcn_metadata:
        result["AI_State"] = result["AI_State_Ensemble"]
        return result

    result = predict_tcn_fuel_state(result, tcn_model, tcn_metadata)
    groups = result.groupby("SegmentID", sort=False, dropna=False) if "SegmentID" in result.columns else [(0, result)]

    for _, group in groups:
        ordered = group.sort_values("FuelTime", kind="stable") if "FuelTime" in group.columns else group
        tcn_consumption_streak = 0
        tcn_spike_streak = 0
        prev_raw = np.nan

        for idx, row in ordered.iterrows():
            rf_state = str(row.get("AI_State_RF", "UNKNOWN"))
            tcn_state = str(row.get("TCN_State", "UNKNOWN"))
            tcn_conf = float(row.get("TCN_Confidence", 0.0) or 0.0)
            flat = float(row.get("flat_jitter_threshold", 0.8) or 0.8)
            fuel = float(row.get("FuelLevel", np.nan))
            raw_step = 0.0 if pd.isna(prev_raw) or pd.isna(fuel) else fuel - prev_raw

            if tcn_state == "CONSUMPTION" and raw_step < -flat * 0.25:
                tcn_consumption_streak += 1
            else:
                tcn_consumption_streak = 0

            if tcn_state == "SPIKE" or tcn_state in TRANSIENT_LABELS:
                tcn_spike_streak += 1
            else:
                tcn_spike_streak = 0

            ensemble_state = rf_state

            if rf_state in {"REFUEL", "DRAIN"}:
                ensemble_state = rf_state
            elif rf_state == "SPIKE":
                if tcn_state in TRANSIENT_LABELS and tcn_conf >= 0.70:
                    ensemble_state = tcn_state
                elif tcn_state == "SPIKE" or tcn_conf >= 0.88:
                    ensemble_state = "SPIKE"
                elif tcn_state in {"SLOSHING_NOISE", "CONSUMPTION"} and tcn_conf >= 0.75:
                    ensemble_state = tcn_state
            elif rf_state in TRANSIENT_LABELS:
                if tcn_state in TRANSIENT_LABELS or tcn_conf >= 0.70:
                    ensemble_state = rf_state
                else:
                    ensemble_state = "SPIKE"
            elif rf_state == "SLOSHING_NOISE":
                if tcn_state == "CONSUMPTION" and tcn_consumption_streak >= 3 and tcn_conf >= 0.62:
                    ensemble_state = "CONSUMPTION"
                elif tcn_state in TRANSIENT_LABELS and tcn_conf >= 0.82 and tcn_spike_streak <= 4:
                    ensemble_state = tcn_state
                elif tcn_state == "SPIKE" and tcn_conf >= 0.93 and tcn_spike_streak <= 2:
                    ensemble_state = "SPIKE"
                else:
                    ensemble_state = "SLOSHING_NOISE"
            elif rf_state == "CONSUMPTION":
                ensemble_state = "CONSUMPTION"
            elif rf_state == "STABLE_JITTER":
                if tcn_state == "CONSUMPTION" and tcn_consumption_streak >= 4 and tcn_conf >= 0.70:
                    ensemble_state = "CONSUMPTION"
                elif tcn_state in TRANSIENT_LABELS and tcn_conf >= 0.90:
                    ensemble_state = tcn_state
                elif tcn_state == "SPIKE" and tcn_conf >= 0.95:
                    ensemble_state = "SPIKE"
                else:
                    ensemble_state = "STABLE_JITTER"

            result.loc[idx, "AI_State_Ensemble"] = ensemble_state
            prev_raw = fuel

    result["AI_State"] = result["AI_State_Ensemble"]
    return result


def filter_with_ai_state(
    df: pd.DataFrame,
    model=None,
    metadata: dict | None = None,
    tcn_model=None,
    tcn_metadata: dict | None = None,
    profile=None,
    config: AIStateFilterConfig | None = None,
    mode: str = "offline",
) -> pd.DataFrame:
    config = config or AIStateFilterConfig()
    result = predict_ai_fuel_state(df, model, metadata, profile, mode=mode)
    result = _apply_rf_tcn_ensemble(result, tcn_model=tcn_model, tcn_metadata=tcn_metadata)

    filtered = np.full(len(result), np.nan, dtype=float)
    clean_input = np.full(len(result), np.nan, dtype=float)
    output = np.nan
    pending_drain = None
    pending_shift = None
    stable_anchor = None
    stable_clean_count = 0
    stable_recent_values = []
    down_trend_count = 0
    up_trend_count = 0
    prev_raw = np.nan
    filter_input_col = "FuelLevel"
    fuels = pd.to_numeric(result[filter_input_col], errors="coerce").to_numpy(dtype=float)
    raw_fuels = pd.to_numeric(result["FuelLevel"], errors="coerce").to_numpy(dtype=float)
    states = result["AI_State"].astype(str).to_numpy()
    flat_values = pd.to_numeric(result["flat_jitter_threshold"], errors="coerce").fillna(0.8).to_numpy(dtype=float)
    noise_values = pd.to_numeric(result["noise_sigma_liters"], errors="coerce").fillna(0.8).to_numpy(dtype=float)
    rolling_std_values = pd.to_numeric(result["RollingStd12"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    time_gaps = pd.to_numeric(result["TimeGapMinutes"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    reference_gap = float(np.nanmedian(time_gaps[time_gaps > 0])) if np.any(time_gaps > 0) else 5.0
    if pd.isna(reference_gap) or reference_gap <= 0:
        reference_gap = 5.0

    segments = result["SegmentID"].to_numpy() if "SegmentID" in result.columns else np.zeros(len(result))
    prev_segment = None
    recent_up_event = None
    pending_event = None

    for i, raw in enumerate(fuels):
        current_segment = segments[i]
        state = states[i]
        flat = max(float(flat_values[i]), 0.1)
        noise = max(float(noise_values[i]), 0.1)
        rolling_std = max(float(rolling_std_values[i]), 0.0)
        dt_minutes = float(time_gaps[i]) if i < len(time_gaps) and time_gaps[i] > 0 else reference_gap
        
        if current_segment != prev_segment or dt_minutes > max(15.0, reference_gap * 3):
            output = float(raw)
            filtered[i] = output
            clean_input[i] = raw
            prev_raw = float(raw)
            pending_event = None
            pending_drain = None
            pending_shift = None
            stable_anchor = None
            recent_up_event = None
            stable_clean_count = 0
            down_trend_count = 0
            up_trend_count = 0
            stable_recent_values = []
            prev_segment = current_segment
            continue
            
        prev_segment = current_segment

        if pd.isna(raw) or raw <= 0:
            filtered[i] = output
            clean_input[i] = output
            continue

        if pd.isna(output):
            output = float(raw)
            filtered[i] = output
            clean_input[i] = raw
            prev_raw = raw
            continue

        delta = float(raw - output)
        raw_step = 0.0 if pd.isna(prev_raw) else float(raw - prev_raw)
        is_clean_stable_raw = (
            state == "STABLE_JITTER"
            and abs(raw_step) <= flat * 0.35
            and rolling_std <= max(flat * 0.85, noise * 1.25)
        )
        if is_clean_stable_raw:
            stable_clean_count += 1
            stable_recent_values.append(float(raw))
            if len(stable_recent_values) > 5:
                stable_recent_values.pop(0)
        else:
            stable_clean_count = 0
            stable_recent_values = []

        if raw_step < -flat * 0.35:
            down_trend_count += 1
            up_trend_count = 0
        elif raw_step > flat * 0.35:
            up_trend_count += 1
            down_trend_count = 0
        else:
            down_trend_count = max(0, down_trend_count - 1)
            up_trend_count = max(0, up_trend_count - 1)

        measurement = float(raw)
        event = float(result.iloc[i].get("event_threshold", max(flat * 5.0, 5.0)) or max(flat * 5.0, 5.0))
        spike = float(result.iloc[i].get("spike_threshold", max(flat * 2.0, 2.0)) or max(flat * 2.0, 2.0))
        shift_gate = max(event * 0.55, spike * 1.2, flat * 3.0)
        level_shift_confirmed = False
        previous_output = float(output)

        if mode == "offline":
            next_window = fuels[i + 1 : i + 4]
            next_window = next_window[~np.isnan(next_window)]
            future_median = float(np.median(next_window)) if len(next_window) else float(raw)
            moderate_up_shift = (
                delta > max(flat * 1.2, noise * 1.6, 0.8)
                and delta < shift_gate
                and len(next_window) >= 1
                and future_median >= previous_output + max(flat * 0.8, noise * 1.1, 0.5)
                and abs(future_median - float(raw)) <= max(flat * 5.0, noise * 5.0, abs(delta) * 1.20)
            )
            short_bump_returns = (
                delta > shift_gate
                and len(next_window) >= 1
                and abs(future_median - previous_output) <= max(flat * 2.0, abs(delta) * 0.30)
            )
            short_drop_returns = (
                delta < -shift_gate
                and len(next_window) >= 1
                and abs(future_median - previous_output) <= max(flat * 2.0, abs(delta) * 0.30)
            )
            next_step = float(next_window[0] - raw) if len(next_window) else 0.0
            peak_reversal = raw_step > flat * 0.45 and next_step < -flat * 0.45
            trough_reversal = raw_step < -flat * 0.45 and next_step > flat * 0.45
            local_window = fuels[max(0, i - 2) : min(len(fuels), i + 3)]
            local_window = local_window[~np.isnan(local_window)]
            local_span = float(local_window.max() - local_window.min()) if len(local_window) >= 3 else 0.0
            mountain_reversal = (
                len(next_window) >= 1
                and delta > max(flat * 3.0, shift_gate * 0.45)
                and next_step < -max(flat * 1.5, abs(delta) * 0.25)
                and future_median <= previous_output + max(flat * 2.0, abs(delta) * 0.35)
            )
            valley_reversal = (
                len(next_window) >= 1
                and delta < -max(flat * 3.0, shift_gate * 0.45)
                and next_step > max(flat * 1.5, abs(delta) * 0.25)
                and future_median >= previous_output - max(flat * 2.0, abs(delta) * 0.35)
            )
            local_burst_noise = (
                local_span >= max(shift_gate * 0.65, flat * 4.0)
                and len(next_window) >= 2
                and abs(future_median - previous_output) <= max(flat * 2.5, local_span * 0.35)
            )
            short_excursion_returns = (
                len(next_window) >= 1
                and abs(delta) >= max(shift_gate * 0.60, flat * 4.0)
                and abs(future_median - previous_output) <= max(flat * 3.0, abs(delta) * 0.40)
            )
            up_excursion_then_drop = (
                delta > max(flat * 2.0, noise * 2.0, 1.0)
                and len(next_window) >= 2
                and next_step < -max(flat * 0.8, noise * 0.8, 0.5)
                and future_median <= previous_output + max(flat * 2.0, abs(delta) * 0.45)
            )

            if (
                short_bump_returns
                or short_drop_returns
                or peak_reversal
                or trough_reversal
                or mountain_reversal
                or valley_reversal
                or local_burst_noise
                or short_excursion_returns
                or up_excursion_then_drop
            ):
                if mountain_reversal or valley_reversal or local_burst_noise or up_excursion_then_drop:
                    state = TRANSIENT_STATE
                else:
                    state = "SPIKE"
                pending_shift = None
            elif state == "SPIKE" or state in TRANSIENT_LABELS or abs(delta) < shift_gate:
                pending_shift = None
            else:
                direction = 1 if delta > 0 else -1
                if (
                    pending_shift is not None
                    and pending_shift["direction"] == direction
                    and abs(raw - pending_shift["target"]) <= max(flat * 2.0, abs(delta) * 0.25)
                ):
                    pending_shift["target"] = 0.5 * pending_shift["target"] + 0.5 * float(raw)
                    pending_shift["count"] += 1
                else:
                    pending_shift = {"direction": direction, "target": float(raw), "count": 1}

                if pending_shift["count"] >= 2 or state in {"REFUEL", "DRAIN"}:
                    level_shift_confirmed = True
                    measurement = float(raw)
        else:
            moderate_up_shift = False
            short_bump_returns = False
            short_drop_returns = False
            peak_reversal = False
            trough_reversal = False
            mountain_reversal = False
            valley_reversal = False
            local_burst_noise = False
            short_excursion_returns = False
            up_excursion_then_drop = False
            
            if pending_event is not None:
                pending_event["age"] += 1
                if abs(raw - pending_event["level_before"]) <= max(flat * 2.0, noise * 2.0):
                    state = TRANSIENT_STATE
                    measurement = output
                    alpha = 0.0
                    pending_event = None
                elif abs(raw - pending_event["target"]) <= max(flat * 3.0, abs(delta) * 0.25):
                    pending_event["count"] += 1
                    if pending_event["count"] >= config.realtime_confirm_required:
                        level_shift_confirmed = True
                        state = pending_event["state"]
                        measurement = raw
                        alpha = config.refuel_alpha if state == "REFUEL" else config.drain_alpha
                        pending_event = None
                elif pending_event["age"] >= config.realtime_confirm_count:
                    pending_event = None
            
            if pending_event is None and not level_shift_confirmed:
                if abs(delta) >= shift_gate:
                    pending_event = {
                        "state": "REFUEL" if delta > 0 else "DRAIN",
                        "level_before": previous_output,
                        "target": raw,
                        "count": 1,
                        "age": 0,
                    }
                    measurement = output
                    alpha = config.realtime_event_hold_alpha
                    state = "PENDING_EVENT"

        if state == "SPIKE" or state in TRANSIENT_LABELS:
            measurement = output
            alpha = 0.0
            pending_drain = None
        elif state == "STABLE_JITTER":
            if moderate_up_shift:
                measurement = max(float(raw), future_median)
                alpha = config.small_up_shift_alpha
                stable_anchor = {"value": measurement, "count": config.stable_recenter_count}
            elif stable_clean_count >= 3 and abs(delta) <= max(flat * 1.25, noise * 1.75):
                measurement = float(np.median(stable_recent_values))
                alpha = max(config.stable_alpha, 0.42)
            elif abs(delta) <= flat * config.stable_deadband_ratio:
                if stable_anchor is not None and abs(raw - stable_anchor["value"]) <= flat * config.stable_deadband_ratio:
                    stable_anchor["count"] += 1
                    stable_anchor["value"] = 0.7 * stable_anchor["value"] + 0.3 * float(raw)
                else:
                    stable_anchor = {"value": float(raw), "count": 1}
                measurement = stable_anchor["value"]
                alpha = config.stable_alpha if stable_anchor["count"] >= config.stable_recenter_count else 0.0
            else:
                stable_anchor = {"value": float(raw), "count": 1}
                alpha = config.stable_alpha
            pending_drain = None
        elif state == "SLOSHING_NOISE":
            stable_anchor = None
            alpha = config.sloshing_alpha
            pending_drain = None
        elif state == "REFUEL":
            stable_anchor = None
            # REFUEL lớn, rõ ràng thì bám nhanh.
            # REFUEL nhỏ hoặc chưa vượt shift_gate thì bám vừa phải để tránh sóng nhiên liệu.
            if delta >= shift_gate:
                alpha = config.refuel_alpha
            else:
                alpha = min(config.refuel_alpha, 0.45)
            pending_drain = None
        elif state == "DRAIN":
            stable_anchor = None
            if pending_drain is None:
                pending_drain = measurement
                alpha = config.drain_candidate_alpha
            else:
                alpha = config.drain_alpha
            measurement = min(measurement, pending_drain if pending_drain is not None else measurement)
        elif state == "CONSUMPTION":
            stable_anchor = None
            if delta > flat:
                # Đang tiêu hao mà raw nhô lên nhỏ -> coi là nhiễu sóng, không kéo output lên
                measurement = output
                alpha = 0.0
            else:
                # Raw giảm -> cho bám theo xu hướng tiêu hao
                measurement = float(raw)
                alpha = config.consumption_alpha
            pending_drain = None
        else:
            stable_anchor = None
            alpha = config.unknown_alpha
            pending_drain = None

        noisy_consumption_context = (
            down_trend_count >= config.noisy_consumption_count
            and raw < output
            and raw_step < -flat * 0.25
            and state in {"SLOSHING_NOISE", "STABLE_JITTER", "CONSUMPTION"}
        )
        if noisy_consumption_context:
            state = "NOISY_CONSUMPTION"
            measurement = float(raw)
            alpha = max(alpha, config.noisy_consumption_alpha)

        if (
            down_trend_count >= config.trend_follow_count
            and raw < output
            and state not in {"SPIKE", "REFUEL", "DRAIN", "NOISY_CONSUMPTION", *TRANSIENT_LABELS}
        ):
            state = "CONSUMPTION"
            measurement = float(raw)
            alpha = max(alpha, config.trend_follow_alpha)

        if (
            up_trend_count >= config.up_shift_follow_count
            and raw > output
            and state not in {"SPIKE", "DRAIN", TRANSIENT_STATE, *TRANSIENT_LABELS}
            and not short_bump_returns
            and not peak_reversal
            and not mountain_reversal
            and not local_burst_noise
            and not short_excursion_returns
            and not up_excursion_then_drop
        ):
            state = "REFUEL" if raw - output >= shift_gate else "STABLE_JITTER"
            measurement = float(raw)
            alpha = max(alpha, config.up_shift_follow_alpha)

        if level_shift_confirmed:
            if pending_shift is not None and pending_shift["direction"] > 0:
                alpha = max(alpha, 0.95)
            else:
                alpha = max(alpha, 0.90)

        if short_excursion_returns and state not in {"REFUEL", "DRAIN"}:
            state = TRANSIENT_STATE
            measurement = output
            alpha = 0.0
            pending_shift = None

        # Consumption guard: khi đang có xu hướng giảm,
        # không cho dao động tăng nhỏ kéo output đi lên
        if (
            down_trend_count >= 2
            and raw > output
            and raw - output < shift_gate
            and state not in {"REFUEL", "DRAIN", "SPIKE", TRANSIENT_STATE}
            and state not in TRANSIENT_LABELS
        ):
            state = "SLOSHING_NOISE"
            measurement = output
            alpha = 0.0

        if (
            recent_up_event is not None
            and recent_up_event["age"] <= 3
            and raw < recent_up_event["peak"] - max(flat * 2.0, noise * 2.0, 1.0)
            and raw <= recent_up_event["level_before"] + max(flat * 2.0, noise * 2.0, 1.0)
            and state not in {"DRAIN"}
        ):
            state = TRANSIENT_STATE
            measurement = float(raw)
            alpha = max(alpha, 0.65)
            pending_shift = None
            recent_up_event = None

        alpha = _dt_alpha(alpha, dt_minutes, reference_gap)

        if state == "REFUEL":
            output = output + alpha * (measurement - output)
        elif state == "DRAIN":
            output = output + alpha * (measurement - output)
        elif state == "SPIKE" or state in TRANSIENT_LABELS:
            output = output
        else:
            output = output + alpha * (measurement - output)

        if output < 0:
            output = 0.0
        if level_shift_confirmed and abs(float(raw) - output) <= max(flat, shift_gate * 0.10):
            pending_shift = None

        if state == "REFUEL" or (level_shift_confirmed and raw > previous_output):
            recent_up_event = {
                "level_before": previous_output,
                "peak": float(raw),
                "age": 0,
            }
        elif recent_up_event is not None:
            recent_up_event["age"] += 1
            if recent_up_event["age"] > 3:
                recent_up_event = None

        clean_input[i] = measurement
        filtered[i] = output
        prev_raw = raw

    result["AI_CleanInput"] = clean_input
    result["AI_FilterInputFuel"] = fuels
    result["AI_State_Filtered"] = filtered
    return result
