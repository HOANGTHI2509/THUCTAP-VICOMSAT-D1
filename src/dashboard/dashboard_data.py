"""Data preparation shared by the Topic 1 Streamlit dashboard and tests."""

from __future__ import annotations

from pathlib import Path
import math

import numpy as np
import pandas as pd

from src.core.filters.smooth_tracking import AISmoothTrackingFilter
from src.core.filters.smooth_tracking.dataframe import filter_smooth_tracking_dataframe
from src.core.filters.smooth_tracking.state import VehicleFilterContext
from src.core.filters.smooth_tracking.capacity import capacity_for_vehicle


REQUIRED_COLUMNS = ("FuelTime", "FuelLevel")


def available_vehicle_sources(data_directory: Path) -> dict[str, Path]:
    """Return telemetry files (CSV or Excel) keyed by their vehicle identifier."""
    if not data_directory.is_dir():
        return {}
    sources = {}
    files = sorted(list(data_directory.glob("*.csv")) + list(data_directory.glob("*.xlsx")))
    for path in files:
        key = path.stem.replace("_processed", "").replace("_da_gop", "")
        sources[key] = path
    return sources


def load_telemetry_csv(file_path: Path) -> pd.DataFrame:
    """Load one vehicle telemetry file (CSV or Excel) and normalize fields."""
    if str(file_path).lower().endswith((".xlsx", ".xls")):
        frame = pd.read_excel(file_path)
    else:
        frame = pd.read_csv(file_path)

    missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing required telemetry columns: {', '.join(missing)}")

    frame = frame.copy()
    frame["FuelTime"] = pd.to_datetime(frame["FuelTime"], errors="coerce")
    frame["FuelLevel"] = pd.to_numeric(frame["FuelLevel"], errors="coerce")
    speed_source = (
        frame["Speed"]
        if "Speed" in frame.columns
        else frame["MotionSpeed"]
        if "MotionSpeed" in frame.columns
        else pd.Series(0.0, index=frame.index)
    )
    frame["Speed"] = pd.to_numeric(speed_source, errors="coerce").fillna(0.0)

    frame = frame.dropna(subset=["FuelTime"]).sort_values("FuelTime", kind="stable")

    # Tự động chia segment nếu file thô chưa có SegmentID (cắt segment khi mất tín hiệu > 30 phút)
    if "SegmentID" in frame.columns:
        frame["SegmentID"] = pd.to_numeric(frame["SegmentID"], errors="coerce").fillna(0).astype(int)
    else:
        dt_gap_s = frame["FuelTime"].diff().dt.total_seconds().fillna(0.0)
        frame["SegmentID"] = (dt_gap_s > 1800.0).cumsum().astype(int) + 1

    for column in ("Lat", "Lng"):
        if column in frame.columns:
            if frame[column].dtype == object:
                frame[column] = frame[column].astype(str).str.replace(",", ".")
            frame[column] = pd.to_numeric(frame[column], errors="coerce")

    # Tu dong hoan doi Lat va Lng neu nguon du lieu bi dao cot (Lat ~ 105 do E, Lng ~ 21 do N)
    if "Lat" in frame.columns and "Lng" in frame.columns:
        lat_valid = frame["Lat"].dropna()
        lng_valid = frame["Lng"].dropna()
        if not lat_valid.empty and not lng_valid.empty:
            if lat_valid.median() > 50.0 and lng_valid.median() < 50.0:
                frame["Lat"], frame["Lng"] = frame["Lng"].copy(), frame["Lat"].copy()

    return frame.sort_values(["SegmentID", "FuelTime"], kind="stable")


def estimate_capacity_liters(frame: pd.DataFrame, vehicle_id: str = "") -> float | None:
    """Return only calibrated capacity; unknown vehicles stay explicitly unknown."""
    del frame
    return capacity_for_vehicle(vehicle_id)


def _predict_causal_states_batch(
    frame: pd.DataFrame,
    engine: AISmoothTrackingFilter,
    vehicle_id: str,
    capacity_est_liters: float | None,
) -> pd.Series | None:
    """Predict model states in one call while preserving causal features.

    Model features depend on raw history, speed and GPS, but not on earlier
    classifier outputs. Building them sequentially and predicting as one matrix
    avoids thousands of expensive ``RandomForest.predict`` calls.
    """
    if engine.model is None or not engine.feature_columns or frame.empty:
        return None

    config = engine.config
    extractor = engine.feature_extractor
    context = VehicleFilterContext(
        vehicle_id=vehicle_id,
        capacity_est=(
            capacity_est_liters
            if capacity_est_liters and capacity_est_liters > config.minimum_capacity
            else 0.0
        ),
    )
    states = np.full(len(frame), "STABLE_JITTER", dtype=object)
    probabilities = np.zeros(len(frame), dtype=float)
    feature_rows: list[list[float]] = []
    feature_positions: list[int] = []

    for position, row in enumerate(frame.itertuples(index=False)):
        timestamp = engine._normalize_timestamp(getattr(row, "FuelTime"))
        raw_fuel = engine._normalize_number(getattr(row, "FuelLevel"), np.nan)
        speed = engine._normalize_number(getattr(row, "Speed", 0.0), 0.0)
        latitude = getattr(row, "Lat", None)
        longitude = getattr(row, "Lng", None)
        coordinate = extractor._valid_coordinate(latitude, longitude)
        motion = extractor.motion_evidence(context, speed, latitude, longitude)

        if context.last_clean_fuel is None:
            initial = (
                config.initial_fuel_fallback
                if math.isnan(raw_fuel) or raw_fuel <= 0.0
                else raw_fuel
            )
            context.last_clean_fuel = initial
            if not capacity_est_liters:
                context.capacity_est = max(config.minimum_capacity, initial * 1.25)
            context.kalman_x = initial
            context.kalman_p = 1.0
            context.last_time = timestamp
            context.last_raw_fuel = raw_fuel
            context.history_fuel.append(initial)
            context.history_time.append(timestamp)
            context.history_speed.append(speed)
            context.history_coordinates.append(coordinate)
            states[position] = "INIT"
            continue

        dt_seconds = (
            (timestamp - context.last_time).total_seconds()
            if context.last_time is not None
            else config.nominal_period_minutes * 60.0
        )
        if dt_seconds < 0.0:
            dt_seconds = config.nominal_period_minutes * 60.0
        dt_minutes = max(dt_seconds / 60.0, 0.1)

        if (
            dt_minutes > config.reset_gap_minutes
            and not math.isnan(raw_fuel)
            and raw_fuel > 0.0
        ):
            context.kalman_x = raw_fuel
            context.kalman_p = 1.0
            context.last_clean_fuel = raw_fuel
            context.recent_upward_steps = 0
            context.pending_downward_count = 0
            context.history_fuel.clear()
            context.history_time.clear()
            context.history_speed.clear()
            context.history_coordinates.clear()

        if not math.isnan(raw_fuel) and raw_fuel > context.capacity_est:
            context.capacity_est = raw_fuel * config.capacity_headroom

        valid_measurement = not math.isnan(raw_fuel) and raw_fuel > 0.0
        if valid_measurement:
            features = extractor.model_features(
                context,
                raw_fuel,
                speed,
                dt_minutes,
                motion,
            )
            vector = [features.get(column, 0.0) for column in engine.feature_columns]
            if all(math.isfinite(float(value)) for value in vector):
                feature_rows.append(vector)
                feature_positions.append(position)

        context.last_time = timestamp
        context.history_time.append(timestamp)
        context.history_speed.append(speed)
        context.history_coordinates.append(coordinate)
        if valid_measurement:
            context.last_raw_fuel = raw_fuel
            context.history_fuel.append(raw_fuel)

    if feature_rows:
        try:
            predictions = engine.model.predict(np.asarray(feature_rows, dtype=float))
            states[feature_positions] = predictions
            if hasattr(engine.model, "predict_proba"):
                predicted_probabilities = np.max(
                    engine.model.predict_proba(np.asarray(feature_rows, dtype=float)),
                    axis=1,
                )
                probabilities[feature_positions] = predicted_probabilities
        except Exception:
            # The realtime engine falls back to STABLE_JITTER on model errors.
            pass

    result = pd.Series(states, index=frame.index, dtype="object")
    result.attrs["probabilities"] = pd.Series(probabilities, index=frame.index, dtype=float)
    return result


def run_topic1_filter(
    frame: pd.DataFrame,
    vehicle_id: str,
    capacity_est_liters: float | None,
    model_dir: str = "models/fuel_state_classifier",
) -> pd.DataFrame:
    """Run only the causal purple filter and expose the canonical diagnostics.

    Every SegmentID starts with a fresh vehicle context. Historical model labels
    in processed CSV files are deliberately ignored: the dashboard must reflect
    the current Topic 1 pipeline, not a prior event-classification result.
    """
    if frame.empty:
        return frame.copy()

    engine = AISmoothTrackingFilter(model_dir=model_dir)
    parts: list[pd.DataFrame] = []
    for segment_id, segment in frame.groupby("SegmentID", sort=False, dropna=False):
        segment = segment.sort_values("FuelTime", kind="stable").copy()
        segment = segment.drop(
            columns=[column for column in ("AI_State",) if column in segment],
        )
        predicted_states = _predict_causal_states_batch(
            segment,
            engine,
            vehicle_id=f"{vehicle_id}:{segment_id}:features",
            capacity_est_liters=capacity_est_liters,
        )
        if predicted_states is not None:
            segment["AI_State"] = predicted_states
            segment["AI_Probability"] = predicted_states.attrs.get("probabilities", 0.0)
        filtered = filter_smooth_tracking_dataframe(
            segment,
            vehicle_id=vehicle_id,
            capacity_est=capacity_est_liters,
            filter_engine=engine,
        )
        parts.append(filtered)

    result = pd.concat(parts).sort_index(kind="stable")
    result["CleanFuel"] = result["CleanFuel_SmoothTracking"]
    result["SignalState"] = result["AI_State_SmoothTracking"]
    result["QualityFlag"] = result["QualityFlag_SmoothTracking"]
    result["MotionState"] = result["MotionState_SmoothTracking"]
    result["MotionConfidence"] = result["MotionConfidence_SmoothTracking"]
    result["GpsDisplacementMeters"] = result["GpsDisplacementMeters_SmoothTracking"]
    result["RollingStd"] = (
        result.groupby("SegmentID", dropna=False)["FuelLevel"]
        .transform(lambda values: values.rolling(12, min_periods=2).std(ddof=0))
        .fillna(0.0)
    )
    from src.core.filters.ai_enhanced_adaptive_realtime import chay_kalman_thich_nghi_1d

    adaptive_capacity = capacity_est_liters or max(
        1.0, float(pd.to_numeric(result["FuelLevel"], errors="coerce").quantile(0.995)) * 1.25
    )
    result["Kalman_Adaptive"] = chay_kalman_thich_nghi_1d(result, capacity=adaptive_capacity)
    return result
