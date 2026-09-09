"""Thread-safe adapter that exposes the purple Smooth-Tracking engine to APIs."""

from __future__ import annotations

import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.core.filters.smooth_tracking import AISmoothTrackingFilter, VehicleFilterContext


class StreamingStateManager:
    """Quan ly state causal theo xe va chi su dung thuat toan duong tim."""

    def __init__(self, model_dir: Optional[str] = None):
        if model_dir is None:
            model_dir = str(Path("models/fuel_state_classifier"))
        self._registry_lock = threading.RLock()
        # Backward-compatible alias for callers that inspect the context registry.
        self._lock = self._registry_lock
        self._vehicle_locks: Dict[str, threading.RLock] = {}
        self.engine = AISmoothTrackingFilter(model_dir=model_dir)
        # Giu cac thuoc tinh cu de SDK/API hien tai tuong thich.
        self._contexts = self.engine.contexts
        self.model = self.engine.model
        self.metadata = self.engine.metadata
        self.feature_columns = self.engine.feature_columns
        self.labels = self.metadata.get("labels", []) if self.metadata else []
        self._stats: Dict[str, Dict[str, Any]] = {}

    def _get_vehicle_lock(self, vehicle_id: str) -> threading.RLock:
        """Create/read one stable lock per vehicle under the registry lock."""
        with self._registry_lock:
            vehicle_lock = self._vehicle_locks.get(vehicle_id)
            if vehicle_lock is None:
                vehicle_lock = threading.RLock()
                self._vehicle_locks[vehicle_id] = vehicle_lock
            return vehicle_lock

    def get_or_create_context(
        self,
        vehicle_id: str,
        capacity_est: Optional[float] = None,
        noise_sigma_liters: Optional[float] = None,
    ) -> VehicleFilterContext:
        # noise_sigma_liters duoc giu trong API de tuong thich; engine tu tinh
        # jitter theo capacity va cua so raw da quy doi sang lit.
        del noise_sigma_liters
        vehicle_lock = self._get_vehicle_lock(vehicle_id)
        with vehicle_lock:
            with self._registry_lock:
                return self.engine.get_or_create_context(vehicle_id, capacity_est)

    def reset_vehicle_state(self, vehicle_id: str) -> bool:
        vehicle_lock = self._get_vehicle_lock(vehicle_id)
        with vehicle_lock:
            with self._registry_lock:
                existed = vehicle_id in self._contexts
                self.engine.reset_context(vehicle_id)
                self._stats.pop(vehicle_id, None)
                return existed

    def list_active_vehicles(self) -> List[Dict[str, Any]]:
        with self._registry_lock:
            vehicle_ids = list(self._contexts)

        vehicles: List[Dict[str, Any]] = []
        for vehicle_id in vehicle_ids:
            vehicle_lock = self._get_vehicle_lock(vehicle_id)
            with vehicle_lock:
                with self._registry_lock:
                    context = self._contexts.get(vehicle_id)
                    stats = dict(self._stats.get(vehicle_id, {}))
                if context is None:
                    continue
                vehicles.append({
                    "vehicle_id": vehicle_id,
                    "capacity_est": context.capacity_est,
                    "noise_sigma": max(0.5, 0.002 * context.capacity_est),
                    "total_points": stats.get("total_points", 0),
                    "last_seen": (
                        context.last_time.isoformat()
                        if context.last_time is not None
                        else None
                    ),
                    "last_clean_fuel": context.last_clean_fuel,
                    "last_state": stats.get("last_state", "UNKNOWN"),
                    "last_motion_state": stats.get("last_motion_state", "UNCERTAIN"),
                })
        return vehicles

    @property
    def active_vehicle_count(self) -> int:
        with self._registry_lock:
            return len(self._contexts)

    def process_point(
        self,
        vehicle_id: str,
        fuel_time: datetime,
        fuel_level: float,
        speed: float = 0.0,
        lat: Optional[float] = None,
        lng: Optional[float] = None,
        distance_meters: float = 0.0,
        segment_id: Optional[str] = None,
        capacity_est: Optional[float] = None,
        noise_sigma_liters: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Loc mot diem da quy doi sang lit bang Smooth-Tracking causal."""
        del distance_meters, segment_id, noise_sigma_liters
        started = time.perf_counter()
        vehicle_lock = self._get_vehicle_lock(vehicle_id)
        with vehicle_lock:
            # Register the context safely before the engine reads it. From this
            # point only this vehicle lock guards mutations of that context.
            with self._registry_lock:
                context = self.engine.get_or_create_context(vehicle_id, capacity_est)
            previous_clean_fuel = (
                float(context.last_clean_fuel)
                if context.last_clean_fuel is not None
                else float(fuel_level)
            )
            result = self.engine.process_point(
                vehicle_id=vehicle_id,
                timestamp=fuel_time,
                raw_fuel=fuel_level,
                speed=speed,
                capacity_est=capacity_est,
                lat=lat,
                lng=lng,
            )
            with self._registry_lock:
                stats = self._stats.setdefault(vehicle_id, {"total_points": 0})
                stats["total_points"] += 1
                stats["last_state"] = result["ai_state"]
                stats["last_motion_state"] = result["motion_state"]

        return {
            "vehicle_id": vehicle_id,
            "fuel_time": fuel_time.isoformat(),
            "raw_fuel_liters": round(float(fuel_level), 2),
            "clean_fuel_liters": float(result["clean_fuel"]),
            "ai_signal_state": result["ai_state"],
            "confidence": 1.0,
            "quality_flag": result["quality_flag"],
            "latency_ms": round((time.perf_counter() - started) * 1000.0, 3),
            "previous_clean_fuel_liters": previous_clean_fuel,
            "motion_state": result["motion_state"],
            "motion_confidence": result["motion_confidence"],
            "gps_displacement_meters": result["gps_displacement_meters"],
        }
