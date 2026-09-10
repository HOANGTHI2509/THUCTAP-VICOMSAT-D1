"""Data preparation shared by the Topic 1 Streamlit dashboard and tests."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.core.filters.smooth_tracking import AISmoothTrackingFilter
from src.core.filters.smooth_tracking.dataframe import filter_smooth_tracking_dataframe


REQUIRED_COLUMNS = ("FuelTime", "FuelLevel")


def available_vehicle_sources(data_directory: Path) -> dict[str, Path]:
    """Return processed telemetry CSV files keyed by their vehicle identifier."""
    if not data_directory.is_dir():
        return {}
    return {
        path.name.removesuffix("_processed.csv"): path
        for path in sorted(data_directory.glob("*_processed.csv"))
    }


def load_telemetry_csv(file_path: Path) -> pd.DataFrame:
    """Load one processed vehicle file and normalize fields required by Topic 1."""
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
    segment_source = (
        frame["SegmentID"]
        if "SegmentID" in frame.columns
        else pd.Series(0, index=frame.index)
    )
    frame["Speed"] = pd.to_numeric(speed_source, errors="coerce").fillna(0.0)
    frame["SegmentID"] = pd.to_numeric(segment_source, errors="coerce").fillna(0)
    for column in ("Lat", "Lng"):
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame.dropna(subset=["FuelTime"]).sort_values(
        ["SegmentID", "FuelTime"], kind="stable"
    )


def estimate_capacity_liters(frame: pd.DataFrame) -> float:
    """Estimate a safe threshold scale when calibration capacity is unavailable."""
    fuel = pd.to_numeric(frame["FuelLevel"], errors="coerce")
    fuel = fuel[(fuel > 0) & fuel.notna()]
    if fuel.empty:
        return 200.0
    return max(200.0, float(fuel.quantile(0.995)))


def run_topic1_filter(
    frame: pd.DataFrame,
    vehicle_id: str,
    capacity_est_liters: float,
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
        filtered = filter_smooth_tracking_dataframe(
            segment,
            vehicle_id=f"{vehicle_id}:{segment_id}",
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
    return result
