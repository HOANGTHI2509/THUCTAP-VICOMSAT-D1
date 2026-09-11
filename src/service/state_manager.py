"""Thread-safe adapter that exposes the purple Smooth-Tracking engine to APIs."""

from __future__ import annotations

import threading
import time
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from src.core.filters.smooth_tracking import AISmoothTrackingFilter, VehicleFilterContext
from src.core.filters.smooth_tracking.contracts import normalize_signal_state
from src.service.state_store import (
    StateStore,
    create_state_store_from_env,
    deserialize_vehicle_context,
    serialize_vehicle_context,
    state_ttl_from_env,
)


logger = logging.getLogger(__name__)


class StreamingStateManager:
    """Quan ly state causal theo xe va chi su dung thuat toan duong tim."""

    def __init__(
        self,
        model_dir: Optional[str] = None,
        state_store: Optional[StateStore] = None,
        state_ttl_seconds: Optional[int] = None,
        capacity_resolver: Optional[Callable[[str], Optional[float]]] = None,
    ):
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
        self.state_store = state_store or create_state_store_from_env()
        self.state_ttl_seconds = state_ttl_seconds or state_ttl_from_env()
        self.capacity_resolver = capacity_resolver

    def _resolve_capacity(self, vehicle_id: str, requested: Optional[float]) -> tuple[Optional[float], str]:
        master = self.capacity_resolver(vehicle_id) if self.capacity_resolver is not None else None
        if self.engine._valid_capacity(master):
            return float(master), "MASTER_DATA"
        if self.engine._valid_capacity(requested):
            return float(requested), "REQUEST"
        return None, "NONE"

    def _restore_context_locked(self, vehicle_id: str) -> None:
        """Restore one missing context while its vehicle lock is held."""
        with self._registry_lock:
            if vehicle_id in self._contexts:
                return
        document = self.state_store.load(vehicle_id)
        if document is None:
            return
        try:
            context, stats = deserialize_vehicle_context(document)
        except (KeyError, TypeError, ValueError):
            logger.warning("Discarding invalid persisted state for vehicle %s", vehicle_id)
            self.state_store.delete(vehicle_id)
            return
        if context.vehicle_id != vehicle_id:
            logger.warning("Discarding mismatched persisted state for vehicle %s", vehicle_id)
            self.state_store.delete(vehicle_id)
            return
        with self._registry_lock:
            self._contexts[vehicle_id] = context
            self._stats[vehicle_id] = stats

    def _persist_context_locked(self, vehicle_id: str) -> None:
        """Persist one complete context while its vehicle lock is held."""
        with self._registry_lock:
            context = self._contexts.get(vehicle_id)
            stats = dict(self._stats.get(vehicle_id, {}))
        if context is None:
            return
        self.state_store.save(
            vehicle_id,
            serialize_vehicle_context(context, stats),
            self.state_ttl_seconds,
        )

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
            self._restore_context_locked(vehicle_id)
            with self._registry_lock:
                capacity, source = self._resolve_capacity(vehicle_id, capacity_est)
                return self.engine.get_or_create_context(vehicle_id, capacity, source)

    def reset_vehicle_state(self, vehicle_id: str) -> bool:
        vehicle_lock = self._get_vehicle_lock(vehicle_id)
        with vehicle_lock:
            with self._registry_lock:
                existed = vehicle_id in self._contexts
                self.engine.reset_context(vehicle_id)
                self._stats.pop(vehicle_id, None)
            persisted = self.state_store.delete(vehicle_id)
            return existed or persisted

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
                    "noise_sigma": max(0.5, 0.002 * context.capacity_est) if context.capacity_est is not None else 0.5,
                    "capacity_mode": context.capacity_mode,
                    "capacity_source": context.capacity_source,
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

    def state_store_health(self) -> Dict[str, Any]:
        return self.state_store.health_check()

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
            self._restore_context_locked(vehicle_id)
            # Register the context safely before the engine reads it. From this
            # point only this vehicle lock guards mutations of that context.
            with self._registry_lock:
                capacity, capacity_source = self._resolve_capacity(vehicle_id, capacity_est)
                context = self.engine.get_or_create_context(vehicle_id, capacity, capacity_source)
            previous_clean_fuel = (
                float(context.last_clean_fuel)
                if context.last_clean_fuel is not None
                else None
            )
            result = self.engine.process_point(
                vehicle_id=vehicle_id,
                timestamp=fuel_time,
                raw_fuel=fuel_level,
                speed=speed,
                capacity_est=capacity,
                capacity_source=capacity_source,
                lat=lat,
                lng=lng,
            )
            with self._registry_lock:
                stats = self._stats.setdefault(vehicle_id, {"total_points": 0})
                stats["total_points"] += 1
                stats["last_state"] = result["ai_state"]
                stats["last_motion_state"] = result["motion_state"]
            self._persist_context_locked(vehicle_id)

        signal_state = normalize_signal_state(result["ai_state"])

        return {
            "vehicle_id": vehicle_id,
            "fuel_time": fuel_time.isoformat(),
            "raw_fuel_liters": round(float(fuel_level), 2),
            "clean_fuel_liters": None if result["clean_fuel"] is None else float(result["clean_fuel"]),
            "signal_state": signal_state,
            # Compatibility key for existing Python callers. Public API uses SignalState.
            "ai_signal_state": signal_state,
            "confidence": 1.0,
            "quality_flag": result["quality_flag"],
            "latency_ms": round((time.perf_counter() - started) * 1000.0, 3),
            "previous_clean_fuel_liters": previous_clean_fuel,
            "motion_state": result["motion_state"],
            "motion_confidence": result["motion_confidence"],
            "gps_displacement_meters": result["gps_displacement_meters"],
            "capacity_est": result["capacity_est"],
            "capacity_mode": result["capacity_mode"],
            "capacity_source": result["capacity_source"],
            "operational_state": result["operational_state"],
        }
