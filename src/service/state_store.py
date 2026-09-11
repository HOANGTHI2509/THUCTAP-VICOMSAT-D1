"""Pluggable persistence for per-vehicle causal filter state.

Memory is the safe default for local/demo use. Redis is optional and selected
with ``STATE_BACKEND=redis``; stored values are versioned JSON, never pickle.
"""

from __future__ import annotations

import json
import os
import threading
import time
from abc import ABC, abstractmethod
from collections import deque
from datetime import datetime
from typing import Any, Callable, Dict, Optional

from src.core.filters.smooth_tracking import VehicleFilterContext


STATE_SCHEMA_VERSION = 1
DEFAULT_STATE_TTL_SECONDS = 72 * 60 * 60


class StateStore(ABC):
    """Small storage contract used by ``StreamingStateManager``."""

    backend_name = "abstract"

    @abstractmethod
    def load(self, vehicle_id: str) -> Optional[Dict[str, Any]]:
        """Load a state document or return ``None`` when absent/expired."""

    @abstractmethod
    def save(self, vehicle_id: str, state: Dict[str, Any], ttl_seconds: int) -> None:
        """Atomically replace one vehicle's state document."""

    @abstractmethod
    def delete(self, vehicle_id: str) -> bool:
        """Delete one vehicle state and report whether it existed."""

    @abstractmethod
    def health_check(self) -> Dict[str, Any]:
        """Return a non-secret backend health summary."""


class MemoryStateStore(StateStore):
    """Thread-safe in-process state store with TTL semantics."""

    backend_name = "memory"

    def __init__(self, clock: Callable[[], float] = time.time):
        self._clock = clock
        self._lock = threading.RLock()
        self._documents: Dict[str, tuple[float, Dict[str, Any]]] = {}

    def load(self, vehicle_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            entry = self._documents.get(vehicle_id)
            if entry is None:
                return None
            expires_at, document = entry
            if expires_at <= self._clock():
                self._documents.pop(vehicle_id, None)
                return None
            # JSON round-trip prevents callers mutating the stored copy.
            return json.loads(json.dumps(document))

    def save(self, vehicle_id: str, state: Dict[str, Any], ttl_seconds: int) -> None:
        if ttl_seconds <= 0:
            raise ValueError("ttl_seconds must be positive")
        document = json.loads(json.dumps(state))
        with self._lock:
            self._documents[vehicle_id] = (
                self._clock() + ttl_seconds,
                document,
            )

    def delete(self, vehicle_id: str) -> bool:
        with self._lock:
            return self._documents.pop(vehicle_id, None) is not None

    def health_check(self) -> Dict[str, Any]:
        with self._lock:
            active = sum(
                expires_at > self._clock()
                for expires_at, _ in self._documents.values()
            )
        return {"backend": self.backend_name, "status": "healthy", "stored_states": active}


class RedisStateStore(StateStore):
    """Redis-backed state store using one atomic ``SET EX`` per vehicle."""

    backend_name = "redis"

    def __init__(self, redis_url: str, key_prefix: str = "fuel_filter:state:"):
        if not redis_url:
            raise ValueError("REDIS_URL is required when STATE_BACKEND=redis")
        try:
            import redis
        except ImportError as exc:  # pragma: no cover - depends on optional deployment package
            raise RuntimeError(
                "Redis backend requested but package 'redis' is not installed"
            ) from exc
        self._client = redis.Redis.from_url(redis_url, decode_responses=True)
        self._key_prefix = key_prefix

    def _key(self, vehicle_id: str) -> str:
        return f"{self._key_prefix}{vehicle_id}"

    def load(self, vehicle_id: str) -> Optional[Dict[str, Any]]:
        payload = self._client.get(self._key(vehicle_id))
        return json.loads(payload) if payload else None

    def save(self, vehicle_id: str, state: Dict[str, Any], ttl_seconds: int) -> None:
        if ttl_seconds <= 0:
            raise ValueError("ttl_seconds must be positive")
        payload = json.dumps(state, separators=(",", ":"), allow_nan=False)
        self._client.set(self._key(vehicle_id), payload, ex=ttl_seconds)

    def delete(self, vehicle_id: str) -> bool:
        return bool(self._client.delete(self._key(vehicle_id)))

    def health_check(self) -> Dict[str, Any]:
        try:
            healthy = bool(self._client.ping())
            return {
                "backend": self.backend_name,
                "status": "healthy" if healthy else "unhealthy",
            }
        except Exception as exc:  # pragma: no cover - requires a live Redis outage
            return {
                "backend": self.backend_name,
                "status": "unhealthy",
                "error": type(exc).__name__,
            }


class FallbackStateStore(StateStore):
    """Keep filtering available when an optional persistent backend is down."""

    backend_name = "redis_with_memory_fallback"

    def __init__(self, primary: StateStore, fallback: Optional[StateStore] = None):
        self.primary = primary
        self.fallback = fallback or MemoryStateStore()
        self._last_primary_error: Optional[str] = None

    def load(self, vehicle_id: str) -> Optional[Dict[str, Any]]:
        try:
            document = self.primary.load(vehicle_id)
            self._last_primary_error = None
            if document is not None:
                return document
        except Exception as exc:  # pragma: no cover - exercised with deployment outages
            self._last_primary_error = type(exc).__name__
        return self.fallback.load(vehicle_id)

    def save(self, vehicle_id: str, state: Dict[str, Any], ttl_seconds: int) -> None:
        # Always keep a local recovery copy for this process.
        self.fallback.save(vehicle_id, state, ttl_seconds)
        try:
            self.primary.save(vehicle_id, state, ttl_seconds)
            self._last_primary_error = None
        except Exception as exc:  # pragma: no cover - exercised with deployment outages
            self._last_primary_error = type(exc).__name__

    def delete(self, vehicle_id: str) -> bool:
        deleted = self.fallback.delete(vehicle_id)
        try:
            deleted = self.primary.delete(vehicle_id) or deleted
            self._last_primary_error = None
        except Exception as exc:  # pragma: no cover - exercised with deployment outages
            self._last_primary_error = type(exc).__name__
        return deleted

    def health_check(self) -> Dict[str, Any]:
        try:
            primary_health = self.primary.health_check()
        except Exception as exc:  # pragma: no cover - covered through a fake backend
            self._last_primary_error = type(exc).__name__
            primary_health = {
                "backend": self.primary.backend_name,
                "status": "unhealthy",
                "error": type(exc).__name__,
            }
        if primary_health.get("status") == "healthy" and self._last_primary_error is None:
            return {"backend": self.backend_name, "status": "healthy", "primary": primary_health}
        return {
            "backend": self.backend_name,
            "status": "degraded",
            "primary": primary_health,
            "fallback": self.fallback.health_check(),
            "last_primary_error": self._last_primary_error,
        }


def serialize_vehicle_context(
    context: VehicleFilterContext,
    stats: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Convert the complete causal context to versioned JSON-compatible data."""
    return {
        "schema_version": STATE_SCHEMA_VERSION,
        "vehicle_id": context.vehicle_id,
        "context": {
            "capacity_est": context.capacity_est,
            "capacity_known": context.capacity_known,
            "capacity_mode": context.capacity_mode,
            "kalman_x": context.kalman_x,
            "kalman_p": context.kalman_p,
            "last_clean_fuel": context.last_clean_fuel,
            "last_raw_fuel": context.last_raw_fuel,
            "last_time": context.last_time.isoformat() if context.last_time else None,
            "recent_upward_steps": context.recent_upward_steps,
            "pending_downward_count": context.pending_downward_count,
            "drop_count": context.drop_count,
            "rise_count": context.rise_count,
            "upward_anchor": context.upward_anchor,
            "upward_samples": list(context.upward_samples),
            "segment_id": context.segment_id,
            "operational_state": context.operational_state,
            "stable_baseline": context.stable_baseline,
            "excursion_baseline": context.excursion_baseline,
            "excursion_min": context.excursion_min,
            "excursion_max": context.excursion_max,
            "pending_direction": context.pending_direction,
            "pending_samples": context.pending_samples,
            "pending_elapsed_min": context.pending_elapsed_min,
            "pending_stable_samples": context.pending_stable_samples,
            "max_deviation_pct": context.max_deviation_pct,
            "rebound_ratio": context.rebound_ratio,
            "pullback_ratio": context.pullback_ratio,
            "recovery_direction": context.recovery_direction,
            "recovery_samples_left": context.recovery_samples_left,
            "confirmed_direction": context.confirmed_direction,
            "confirmed_samples_left": context.confirmed_samples_left,
            "confirmed_ramp_step": context.confirmed_ramp_step,
            "reacquisition_step": context.reacquisition_step,
            "robust_noise": context.robust_noise,
            "shadow_fuel": context.shadow_fuel,
            "shadow_values": list(context.shadow_values),
            "history_fuel": list(context.history_fuel),
            "history_time": [value.isoformat() for value in context.history_time],
            "history_speed": list(context.history_speed),
            "history_coordinates": [
                list(value) if value is not None else None
                for value in context.history_coordinates
            ],
        },
        "stats": dict(stats or {}),
    }


def deserialize_vehicle_context(document: Dict[str, Any]) -> tuple[VehicleFilterContext, Dict[str, Any]]:
    """Restore a context and reject unknown future schemas safely."""
    version = document.get("schema_version")
    if version != STATE_SCHEMA_VERSION:
        raise ValueError(f"Unsupported state schema_version: {version!r}")

    values = document["context"]
    context = VehicleFilterContext(
        vehicle_id=str(document["vehicle_id"]),
        capacity_est=float(values["capacity_est"]),
        capacity_known=bool(values.get("capacity_known", False)),
        capacity_mode=str(values.get("capacity_mode", "UNKNOWN_CAPACITY_MODE")),
        kalman_x=values.get("kalman_x"),
        kalman_p=float(values.get("kalman_p", 1.0)),
        last_clean_fuel=values.get("last_clean_fuel"),
        last_raw_fuel=values.get("last_raw_fuel"),
        last_time=(
            datetime.fromisoformat(values["last_time"])
            if values.get("last_time")
            else None
        ),
        recent_upward_steps=int(
            values.get("recent_upward_steps", values.get("recent_refuel_steps", 0))
        ),
        pending_downward_count=int(
            values.get("pending_downward_count", values.get("pending_drain_count", 0))
        ),
        drop_count=int(values.get("drop_count", 0)),
        rise_count=int(values.get("rise_count", 0)),
        upward_anchor=values.get("upward_anchor", values.get("refuel_anchor")),
        upward_samples=list(
            values.get("upward_samples", values.get("refuel_samples", []))
        ),
        segment_id=values.get("segment_id"),
        operational_state=str(values.get("operational_state", "STABLE")),
        stable_baseline=values.get("stable_baseline"),
        excursion_baseline=values.get("excursion_baseline"),
        excursion_min=values.get("excursion_min"),
        excursion_max=values.get("excursion_max"),
        pending_direction=int(values.get("pending_direction", 0)),
        pending_samples=int(values.get("pending_samples", 0)),
        pending_elapsed_min=float(values.get("pending_elapsed_min", 0.0)),
        pending_stable_samples=int(values.get("pending_stable_samples", 0)),
        max_deviation_pct=float(values.get("max_deviation_pct", 0.0)),
        rebound_ratio=float(values.get("rebound_ratio", 0.0)),
        pullback_ratio=float(values.get("pullback_ratio", 0.0)),
        recovery_direction=int(values.get("recovery_direction", 0)),
        recovery_samples_left=int(values.get("recovery_samples_left", 0)),
        confirmed_direction=int(values.get("confirmed_direction", 0)),
        confirmed_samples_left=int(values.get("confirmed_samples_left", 0)),
        confirmed_ramp_step=int(values.get("confirmed_ramp_step", 0)),
        reacquisition_step=int(values.get("reacquisition_step", 0)),
        robust_noise=float(values.get("robust_noise", 0.1)),
        shadow_fuel=values.get("shadow_fuel"),
        shadow_values=deque(values.get("shadow_values", []), maxlen=5),
        history_fuel=deque(values.get("history_fuel", []), maxlen=12),
        history_time=deque(
            (datetime.fromisoformat(value) for value in values.get("history_time", [])),
            maxlen=12,
        ),
        history_speed=deque(values.get("history_speed", []), maxlen=12),
        history_coordinates=deque(
            (
                tuple(value) if value is not None else None
                for value in values.get("history_coordinates", [])
            ),
            maxlen=12,
        ),
    )
    return context, dict(document.get("stats", {}))


def create_state_store_from_env() -> StateStore:
    """Build the configured backend; memory remains the default."""
    backend = os.getenv("STATE_BACKEND", "memory").strip().lower()
    if backend == "memory":
        return MemoryStateStore()
    if backend == "redis":
        return FallbackStateStore(RedisStateStore(os.getenv("REDIS_URL", "")))
    raise ValueError(f"Unsupported STATE_BACKEND: {backend!r}")


def state_ttl_from_env() -> int:
    raw_value = os.getenv("STATE_TTL_SECONDS", str(DEFAULT_STATE_TTL_SECONDS))
    try:
        value = int(raw_value)
    except ValueError as exc:
        raise ValueError("STATE_TTL_SECONDS must be an integer") from exc
    if value <= 0:
        raise ValueError("STATE_TTL_SECONDS must be positive")
    return value
