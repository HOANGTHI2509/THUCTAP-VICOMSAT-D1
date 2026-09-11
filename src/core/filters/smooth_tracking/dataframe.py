"""Adapter DataFrame va cache mac dinh cho Smooth-Tracking."""

from typing import Optional

import numpy as np
import pandas as pd

from .config import SmoothTrackingConfig
from .engine import AISmoothTrackingFilter

# Cache chi dung cho cau hinh mac dinh; caller realtime co the inject engine rieng.
_CACHED_FILTER_ENGINE: Optional[AISmoothTrackingFilter] = None

def get_shared_smooth_engine(model_dir: str = "models/fuel_state_classifier") -> AISmoothTrackingFilter:
    global _CACHED_FILTER_ENGINE
    if _CACHED_FILTER_ENGINE is None:
        _CACHED_FILTER_ENGINE = AISmoothTrackingFilter(model_dir=model_dir)
    return _CACHED_FILTER_ENGINE


def filter_smooth_tracking_dataframe(
    df: pd.DataFrame,
    vehicle_id: Optional[str] = None,
    capacity_est: Optional[float] = None,
    filter_engine: Optional[AISmoothTrackingFilter] = None,
    model_dir: str = "models/fuel_state_classifier",
    config: Optional[SmoothTrackingConfig] = None,
) -> pd.DataFrame:
    """
    Ham tien ich cho Dashboard hoac xu ly Batch theo DataFrame / File.
    Dau ra cua tang tien xu ly De tai 1:
      - CleanFuel_SmoothTracking: Muc nhien lieu da loc bam sat & lam muot.
      - FuelRate_SmoothTracking: Toc do bien thien cua tin hieu sach (L/min).
      - IsStopped: co tuong thich cu, chi dua tren van toc (Speed <= 0.5).
      - MotionState_SmoothTracking: MOVING / LOW_MOTION / UNCERTAIN, dung GPS + van toc.
      - QualityFlag_SmoothTracking: Co chat luong danh dau trang thai loc.
    """
    df_out = df.copy()
    if vehicle_id is None:
        if "VehicleID" in df_out.columns and len(df_out) > 0:
            vehicle_id = str(df_out["VehicleID"].iloc[0])
        else:
            vehicle_id = "Car_Default"

    if filter_engine is None:
        if "AI_State" in df_out.columns:
            filter_engine = AISmoothTrackingFilter(model_dir="", config=config)
        elif config is not None:
            filter_engine = AISmoothTrackingFilter(model_dir=model_dir, config=config)
        else:
            filter_engine = get_shared_smooth_engine(model_dir=model_dir)

    # Reset context cho phan doan / lan chay moi
    filter_engine.reset_context(vehicle_id)

    clean_fuels = []
    fuel_rates = []
    is_stopped_list = []
    ai_states = []
    quality_flags = []
    motion_states = []
    motion_confidences = []
    gps_displacements = []
    capacity_estimates = []
    capacity_modes = []
    capacity_sources = []
    operational_states = []

    time_col = "FuelTime" if "FuelTime" in df_out.columns else df_out.columns[0]
    fuel_col = "FuelLevel" if "FuelLevel" in df_out.columns else "Nhiên liệu"
    speed_col = "Speed" if "Speed" in df_out.columns else ("Vận tốc" if "Vận tốc" in df_out.columns else None)
    has_precomputed_ai = "AI_State" in df_out.columns
    lat_col = "Lat" if "Lat" in df_out.columns else ("Latitude" if "Latitude" in df_out.columns else None)
    lng_col = "Lng" if "Lng" in df_out.columns else ("Longitude" if "Longitude" in df_out.columns else None)

    for row in df_out.itertuples():
        t = getattr(row, time_col, None)
        f = getattr(row, fuel_col, np.nan)
        s = getattr(row, speed_col, 0.0) if speed_col else 0.0
        lat = getattr(row, lat_col, None) if lat_col else None
        lng = getattr(row, lng_col, None) if lng_col else None
        known_ai = getattr(row, "AI_State", None) if has_precomputed_ai else None

        res = filter_engine.process_point(
            vehicle_id=vehicle_id,
            timestamp=t,
            raw_fuel=f,
            speed=s,
            capacity_est=capacity_est,
            known_ai_state=known_ai,
            lat=lat,
            lng=lng,
        )
        clean_fuels.append(res["clean_fuel"])
        fuel_rates.append(res["fuel_rate"])
        is_stopped_list.append(res["is_stopped"])
        ai_states.append(res["ai_state"])
        quality_flags.append(res["quality_flag"])
        motion_states.append(res["motion_state"])
        motion_confidences.append(res["motion_confidence"])
        gps_displacements.append(res["gps_displacement_meters"])
        capacity_estimates.append(res["capacity_est"])
        capacity_modes.append(res["capacity_mode"])
        capacity_sources.append(res["capacity_source"])
        operational_states.append(res["operational_state"])

    df_out["CleanFuel_SmoothTracking"] = clean_fuels
    df_out["FuelRate_SmoothTracking"] = fuel_rates
    df_out["IsStopped"] = is_stopped_list
    df_out["AI_State_SmoothTracking"] = ai_states
    df_out["QualityFlag_SmoothTracking"] = quality_flags
    df_out["MotionState_SmoothTracking"] = motion_states
    df_out["MotionConfidence_SmoothTracking"] = motion_confidences
    df_out["GpsDisplacementMeters_SmoothTracking"] = gps_displacements
    df_out["CapacityEstimate"] = capacity_estimates
    df_out["CapacityMode"] = capacity_modes
    df_out["CapacitySource"] = capacity_sources
    df_out["OperationalState"] = operational_states

    return df_out
