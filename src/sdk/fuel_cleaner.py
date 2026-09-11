"""Python SDK for the Topic 1 realtime fuel-denoising service."""

from __future__ import annotations

import os
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

import pandas as pd

from src.service.state_manager import StreamingStateManager


SIGNAL_STATE_DESCRIPTIONS = {
    "INIT": "Khởi tạo bộ lọc",
    "UPWARD_SHIFT": "Dịch chuyển mức tín hiệu theo chiều tăng",
    "DOWNWARD_SHIFT": "Dịch chuyển mức tín hiệu theo chiều giảm",
    "GRADUAL_CHANGE": "Tín hiệu thay đổi có xu hướng",
    "STABLE_JITTER": "Dao động nhỏ quanh mặt bằng ổn định",
    "OSCILLATION_NOISE": "Nhiễu dao động/sóng sánh",
    "SLOSHING": "Nhiễu sóng sánh",
    "SPIKE": "Xung nhiễu cảm biến",
    "UNCERTAIN": "Chưa đủ bằng chứng để phân loại tín hiệu",
}

# Backward-compatible import name. Descriptions are now signal-only.
AI_STATE_DESCRIPTIONS = SIGNAL_STATE_DESCRIPTIONS


class FuelCleanerEngine:
    """Thread-safe causal Smooth-Tracking adapter for Python callers."""

    def __init__(
        self,
        model_dir: Optional[str] = None,
        state_manager: Optional[StreamingStateManager] = None,
    ):
        if state_manager is not None:
            self.state_manager = state_manager
            return

        if model_dir is None:
            project_root = os.path.abspath(
                os.path.join(os.path.dirname(__file__), "..", "..")
            )
            candidate = os.path.join(project_root, "models", "fuel_state_classifier")
            model_dir = candidate if os.path.exists(candidate) else "models/fuel_state_classifier"
        self.state_manager = StreamingStateManager(model_dir=model_dir)

    @staticmethod
    def _parse_timestamp(value: Union[str, datetime]) -> datetime:
        if isinstance(value, datetime):
            return value
        normalized = str(value).split("+")[0].split("Z")[0]
        for fmt in (
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%dT%H:%M:%S",
            "%Y-%m-%d %H:%M:%S.%f",
        ):
            try:
                return datetime.strptime(normalized, fmt)
            except ValueError:
                continue
        return datetime.utcnow()

    def clean_point(
        self,
        vehicle_id: str,
        timestamp: Union[str, datetime],
        raw_fuel: float,
        speed: float = 0.0,
        distance_m: float = 0.0,
        lat: Optional[float] = None,
        lng: Optional[float] = None,
        capacity_est: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Clean one litre-valued telemetry point using past/current data only."""
        started = time.perf_counter()
        dt = self._parse_timestamp(timestamp)
        result = self.state_manager.process_point(
            vehicle_id=vehicle_id,
            fuel_time=dt,
            fuel_level=float(raw_fuel),
            speed=float(speed) if speed is not None else 0.0,
            distance_meters=float(distance_m) if distance_m is not None else 0.0,
            lat=lat,
            lng=lng,
            capacity_est=capacity_est,
        )

        signal_state = str(result.get("signal_state", "UNCERTAIN"))
        clean_value = result.get("clean_fuel_liters")
        clean_fuel = float(clean_value) if clean_value is not None else None
        return {
            "vehicle_id": vehicle_id,
            "timestamp": dt.isoformat(),
            "raw_fuel": round(float(raw_fuel), 2),
            "clean_fuel": None if clean_fuel is None else round(clean_fuel, 2),
            "speed": round(float(speed or 0.0), 1),
            "signal_state": signal_state,
            "signal_state_description": SIGNAL_STATE_DESCRIPTIONS.get(
                signal_state,
                SIGNAL_STATE_DESCRIPTIONS["UNCERTAIN"],
            ),
            "quality_flag": str(result.get("quality_flag", "VALID")),
            "motion_state": result.get("motion_state", "UNCERTAIN"),
            "motion_confidence": float(result.get("motion_confidence", 0.0)),
            "gps_displacement_meters": float(
                result.get("gps_displacement_meters", 0.0)
            ),
            "confidence": float(result.get("confidence", 1.0)),
            "capacity_est": result.get("capacity_est"),
            "capacity_mode": result.get("capacity_mode", "UNKNOWN"),
            "capacity_source": result.get("capacity_source", "NONE"),
            "processing_time_ms": round((time.perf_counter() - started) * 1000.0, 3),
            # Deprecated SDK alias; use signal_state in new integrations.
            "ai_state": signal_state,
        }

    def clean_batch(
        self,
        df: pd.DataFrame,
        vehicle_id_col: str = "vehicle_id",
        time_col: str = "timestamp",
        fuel_col: str = "raw_fuel",
        speed_col: str = "speed",
        distance_col: Optional[str] = None,
        capacity_est: Optional[float] = None,
    ) -> pd.DataFrame:
        """Clean a historical frame in timestamp order using the realtime path."""
        output = df.copy()
        column_map = {column.lower(): column for column in output.columns}

        def choose(*candidates: Optional[str]) -> Optional[str]:
            return next(
                (
                    column_map[name.lower()]
                    for name in candidates
                    if name and name.lower() in column_map
                ),
                None,
            )

        vehicle_column = choose(vehicle_id_col, "vehicle_id", "car_id", "bien_so", "vehicleid")
        time_column = choose(time_col, "timestamp", "fueltime", "time", "thoi_gian")
        fuel_column = choose(fuel_col, "raw_fuel", "fuellevel", "fuel", "raw", "xang_tho")
        speed_column = choose(speed_col, "speed", "van_toc")
        distance_column = choose(distance_col, "distancemeters", "distance")
        latitude_column = choose("lat", "latitude")
        longitude_column = choose("lng", "longitude")
        if fuel_column is None:
            raise ValueError("No raw-fuel column found")

        results: List[Dict[str, Any]] = []
        for _, row in output.iterrows():
            results.append(
                self.clean_point(
                    vehicle_id=str(row[vehicle_column]) if vehicle_column else "DEFAULT_VEHICLE",
                    timestamp=row[time_column] if time_column else datetime.utcnow(),
                    raw_fuel=float(row[fuel_column]) if pd.notnull(row[fuel_column]) else 0.0,
                    speed=(float(row[speed_column]) if speed_column and pd.notnull(row[speed_column]) else 0.0),
                    distance_m=(float(row[distance_column]) if distance_column and pd.notnull(row[distance_column]) else 0.0),
                    lat=(float(row[latitude_column]) if latitude_column and pd.notnull(row[latitude_column]) else None),
                    lng=(float(row[longitude_column]) if longitude_column and pd.notnull(row[longitude_column]) else None),
                    capacity_est=capacity_est,
                )
            )

        for field in (
            "clean_fuel",
            "signal_state",
            "quality_flag",
            "motion_state",
            "motion_confidence",
            "gps_displacement_meters",
        ):
            output[field] = [result[field] for result in results]
        return output

    def reset_vehicle(self, vehicle_id: str) -> bool:
        return self.state_manager.reset_vehicle_state(vehicle_id)

    def get_active_vehicles(self) -> List[str]:
        with self.state_manager._lock:
            return list(self.state_manager._contexts)
