from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class FuelStateProfile:
    capacity_est: float
    noise_sigma_liters: float
    flat_jitter_threshold: float
    spike_threshold: float
    event_threshold: float
    sample_gap_minutes: float
    max_drop_rate_lpm: float


def _profile_value(profile, name: str, fallback: float) -> float:
    value = getattr(profile, name, fallback)
    value = pd.to_numeric(value, errors="coerce")
    if pd.isna(value):
        return float(fallback)
    return float(value)


def build_fuel_state_profile(df: pd.DataFrame, profile=None) -> FuelStateProfile:
    fuel = pd.to_numeric(df.get("FuelLevel"), errors="coerce")
    fuel = fuel[(fuel > 0) & fuel.notna()]
    if fuel.empty:
        capacity = 200.0
        noise_sigma = 1.0
    else:
        capacity = float(fuel.quantile(0.995))
        if pd.isna(capacity) or capacity < 50.0:
            capacity = max(float(fuel.quantile(0.99)), 50.0)
        diff_abs = fuel.diff().abs().dropna()
        small_diff = diff_abs[diff_abs <= diff_abs.quantile(0.6)] if not diff_abs.empty else diff_abs
        noise_sigma = float(1.4826 * small_diff.median()) if not small_diff.empty else max(0.5, 0.002 * capacity)
        if pd.isna(noise_sigma) or noise_sigma <= 0:
            noise_sigma = max(0.5, 0.002 * capacity)

    gaps = pd.to_numeric(df.get("TimeGapMinutes", pd.Series(dtype=float)), errors="coerce")
    gaps = gaps[(gaps > 0) & gaps.notna()]
    sample_gap = float(gaps.median()) if not gaps.empty else 5.0
    if pd.isna(sample_gap) or sample_gap <= 0:
        sample_gap = 5.0

    capacity = _profile_value(profile, "capacity_est", capacity) if profile is not None else capacity
    noise_sigma = _profile_value(profile, "noise_sigma_liters", noise_sigma) if profile is not None else noise_sigma
    flat_jitter = max(2.5 * noise_sigma, 0.003 * capacity, 0.8)
    spike_threshold = max(4.5 * noise_sigma, 0.012 * capacity, flat_jitter * 1.8)
    event_threshold = max(8.0 * noise_sigma, 0.035 * capacity, spike_threshold * 1.8)

    return FuelStateProfile(
        capacity_est=capacity,
        noise_sigma_liters=noise_sigma,
        flat_jitter_threshold=_profile_value(profile, "flat_jitter_threshold", flat_jitter) if profile is not None else flat_jitter,
        spike_threshold=_profile_value(profile, "spike_threshold", spike_threshold) if profile is not None else spike_threshold,
        event_threshold=_profile_value(profile, "event_threshold", event_threshold) if profile is not None else event_threshold,
        sample_gap_minutes=_profile_value(profile, "sample_gap_minutes", sample_gap) if profile is not None else sample_gap,
        max_drop_rate_lpm=_profile_value(profile, "max_drop_rate_lpm", max(0.2, 0.0015 * capacity / max(sample_gap, 1.0))) if profile is not None else max(0.2, 0.0015 * capacity / max(sample_gap, 1.0)),
    )


def _mode_params(mode: str) -> tuple[float, float]:
    """Return alpha for robust EMA and edge-preserving level filter."""
    if mode in {"SPIKE_UP", "SPIKE_DOWN", "DROPOUT", "INVALID"}:
        return 0.0, 0.0
    if mode == "JITTER":
        return 0.06, 0.04
    if mode == "STABLE":
        return 0.14, 0.08
    if mode == "MOVING_CONSUMPTION":
        return 0.45, 0.55
    if mode == "TREND_DOWN":
        return 0.38, 0.45
    if mode in {"REFUEL_CONFIRMED", "DROP_CONFIRMED"}:
        return 1.0, 1.0
    if mode in {"REFUEL_CANDIDATE", "DROP_CANDIDATE"}:
        return 0.25, 0.25
    return 0.22, 0.20


def _bounded_update(current: float, target: float, alpha: float) -> float:
    updated = current + alpha * (target - current)
    if target >= current:
        return float(np.clip(updated, current, target))
    return float(np.clip(updated, target, current))


def _numeric_array(result: pd.DataFrame, column: str, default: float = 0.0) -> np.ndarray:
    if column in result.columns:
        values = pd.to_numeric(result[column], errors="coerce").fillna(default)
    else:
        values = pd.Series(default, index=result.index, dtype=float)
    return values.to_numpy(dtype=float)


def filter_fuel_series(
    df: pd.DataFrame,
    profile=None,
    source_col: str = "FuelLevel",
    lookback: int = 7,
    lookahead: int = 5,
) -> pd.DataFrame:
    if source_col not in df.columns:
        source_col = "FuelLevel"

    result = df.sort_values("FuelTime").copy()
    state_profile = build_fuel_state_profile(result, profile)

    raw = pd.to_numeric(result[source_col], errors="coerce").to_numpy(dtype=float)
    speed = _numeric_array(result, "Speed", 0.0)
    rolling_std = _numeric_array(result, "RollingStd", 0.0)
    delta_feature = pd.to_numeric(result.get("DeltaFuel", np.nan), errors="coerce").to_numpy(dtype=float)
    quality_reason = result.get("QualityReason", pd.Series([""] * len(result), index=result.index)).astype(str).to_numpy()
    times = pd.to_datetime(result.get("FuelTime"), errors="coerce")

    clean = np.full(len(result), np.nan, dtype=float)
    ema = np.full(len(result), np.nan, dtype=float)
    edge = np.full(len(result), np.nan, dtype=float)
    states = np.array(["INVALID"] * len(result), dtype=object)
    confidence = np.zeros(len(result), dtype=float)
    event_type = np.array(["NONE"] * len(result), dtype=object)

    accepted: list[float] = []
    ema_level = np.nan
    edge_level = np.nan
    prev_time = None
    prev_raw = np.nan

    for i, z in enumerate(raw):
        current_time = times.iloc[i] if hasattr(times, "iloc") else None
        if prev_time is not None and pd.notna(current_time):
            dt_minutes = max((current_time - prev_time).total_seconds() / 60.0, state_profile.sample_gap_minutes)
        else:
            dt_minutes = state_profile.sample_gap_minutes

        if pd.isna(z) or z <= 0:
            mode = "INVALID"
            measurement = accepted[-1] if accepted else np.nan
            conf = 0.0
        else:
            previous_clean = accepted[-1] if accepted else float(z)
            baseline = float(np.median(accepted[-lookback:])) if accepted else float(z)
            future = raw[i + 1 : i + 1 + lookahead]
            future = future[~np.isnan(future)]
            future_median = float(np.median(future)) if len(future) else float(z)

            delta_base = float(z - baseline)
            delta_raw = float(delta_feature[i]) if i < len(delta_feature) and not pd.isna(delta_feature[i]) else (
                0.0 if pd.isna(prev_raw) else float(z - prev_raw)
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

            returns_to_baseline = delta_future_to_base <= state_profile.flat_jitter_threshold
            holds_new_level = delta_future_to_z <= max(state_profile.flat_jitter_threshold, 0.35 * abs(delta_base))
            large_change = abs(delta_base) >= state_profile.spike_threshold or abs(delta_raw) >= state_profile.spike_threshold
            event_change = abs(delta_base) >= state_profile.event_threshold or abs(delta_raw) >= state_profile.event_threshold
            low_value = z <= max(5.0, 0.03 * state_profile.capacity_est)
            high_noise = rolling_std[i] > max(state_profile.flat_jitter_threshold, state_profile.noise_sigma_liters * 3.0)
            large_delta_flag = "LARGE_DELTA" in quality_reason[i].upper()

            if low_value and baseline > max(30.0, 0.25 * state_profile.capacity_est):
                mode = "DROPOUT"
                measurement = previous_clean
                conf = 0.95 if returns_to_baseline else 0.65
                event_type[i] = "DROPOUT"
            elif delta_base > 0 and event_change and holds_new_level and not recent_moving:
                mode = "REFUEL_CONFIRMED"
                measurement = float(z)
                conf = 0.9
                event_type[i] = "REFUEL"
            elif delta_base < 0 and event_change and holds_new_level and not returns_to_baseline:
                mode = "DROP_CONFIRMED"
                measurement = float(z)
                conf = 0.85
                event_type[i] = "DROP"
            elif large_change and returns_to_baseline:
                mode = "SPIKE_UP" if delta_base > 0 else "SPIKE_DOWN"
                measurement = previous_clean
                conf = 0.9
            elif delta_base > state_profile.flat_jitter_threshold and moving_context and not holds_new_level:
                mode = "SPIKE_UP"
                measurement = previous_clean
                conf = 0.8
            elif large_delta_flag and high_noise and not holds_new_level:
                mode = "SPIKE_UP" if delta_base > 0 else "SPIKE_DOWN"
                measurement = previous_clean
                conf = 0.7
            elif abs(float(z) - previous_clean) <= state_profile.flat_jitter_threshold:
                mode = "JITTER" if high_noise else "STABLE"
                alpha = 0.12 if mode == "JITTER" else 0.30
                measurement = previous_clean + alpha * (float(z) - previous_clean)
                conf = 0.75
            elif delta_base < -state_profile.flat_jitter_threshold:
                mode = "MOVING_CONSUMPTION" if recent_moving else "TREND_DOWN"
                measurement = float(z)
                conf = 0.75
            else:
                mode = "NORMAL"
                measurement = float(z)
                conf = 0.6

        if pd.isna(measurement):
            clean[i] = np.nan
            ema[i] = np.nan if pd.isna(ema_level) else ema_level
            edge[i] = np.nan if pd.isna(edge_level) else edge_level
            states[i] = mode
            confidence[i] = conf
            prev_raw = z
            if pd.notna(current_time):
                prev_time = current_time
            continue

        if pd.isna(ema_level):
            ema_level = float(measurement)
        if pd.isna(edge_level):
            edge_level = float(measurement)

        alpha_ema, alpha_edge = _mode_params(mode)

        if mode in {"REFUEL_CONFIRMED", "DROP_CONFIRMED"}:
            ema_level = float(measurement)
            edge_level = float(measurement)
        elif mode in {"SPIKE_UP", "SPIKE_DOWN", "DROPOUT", "INVALID"}:
            pass
        else:
            target = float(measurement)
            if abs(target - edge_level) <= state_profile.flat_jitter_threshold:
                recent = [v for v in accepted[-lookback:] if not pd.isna(v)]
                if recent:
                    target = float(np.median(recent + [target]))
            ema_level = _bounded_update(ema_level, target, alpha_ema)
            edge_level = _bounded_update(edge_level, target, alpha_edge)

        clean[i] = measurement
        ema[i] = ema_level
        edge[i] = edge_level
        states[i] = mode
        confidence[i] = conf
        accepted.append(float(measurement))
        prev_raw = z
        if pd.notna(current_time):
            prev_time = current_time

    result["FuelLevel_CleanInput"] = clean
    result["FuelState"] = states
    result["FuelStateConfidence"] = confidence
    result["FuelEventType"] = event_type
    result["FuelLevel_Filtered_EMA"] = ema
    result["FuelLevel_Filtered_Edge"] = edge
    result["FuelLevel_Filtered_AlphaBeta"] = edge
    return result
