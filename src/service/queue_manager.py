"""FIFO processing queues isolated by vehicle."""

from __future__ import annotations

import collections
import logging
import queue
import threading
from concurrent.futures import Future, TimeoutError as FutureTimeoutError
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

from src.service.state_manager import StreamingStateManager

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _QueueTask:
    """One unit of work handled by a vehicle's single FIFO worker."""

    kind: str
    future: Future
    point_data: Optional[Dict[str, Any]] = None


class VehicleQueueManager:
    """Process each vehicle serially while allowing vehicles to run in parallel.

    ``enqueue`` waits for the point's result so the existing ``/api/push``
    contract remains synchronous. Processing still happens inside the vehicle
    worker; concurrent requests for the same vehicle therefore cannot mutate
    its causal filter state at the same time.
    """

    def __init__(
        self,
        state_manager: StreamingStateManager,
        max_buffer_per_vehicle: int = 20000,
        processing_timeout_seconds: float = 30.0,
    ):
        self.state_manager = state_manager
        self.max_buffer_per_vehicle = max_buffer_per_vehicle
        self.processing_timeout_seconds = processing_timeout_seconds
        self._lock = threading.RLock()
        self._queues: Dict[str, queue.Queue] = {}
        self._workers: Dict[str, threading.Thread] = {}
        self._buffers: Dict[str, collections.deque] = collections.defaultdict(
            lambda: collections.deque(maxlen=self.max_buffer_per_vehicle)
        )
        self._stats: Dict[str, Dict[str, Any]] = {}
        self._running = True
        self.global_live_buffer = collections.deque(maxlen=20000)
        self.active_vehicle_id: Optional[str] = None

    @staticmethod
    def _initial_stats(vehicle_id: str) -> Dict[str, Any]:
        return {
            "vehicle_id": vehicle_id,
            "total_enqueued": 0,
            "total_processed": 0,
            "queue_size": 0,
            "last_enqueued_at": None,
            "last_processed_at": None,
            "current_fuel_raw": 0.0,
            "current_fuel_clean": 0.0,
            "current_fuel_smooth": 0.0,
            "current_speed": 0.0,
            "current_state": "INIT",
            "current_lat": 21.0285,
            "current_lng": 105.8542,
            "current_address": "",
        }

    def _ensure_vehicle_worker(self, vehicle_id: str) -> queue.Queue:
        with self._lock:
            if not self._running:
                raise RuntimeError("VehicleQueueManager has been stopped")

            existing = self._queues.get(vehicle_id)
            if existing is not None:
                return existing

            vehicle_queue: queue.Queue = queue.Queue()
            self._queues[vehicle_id] = vehicle_queue
            self._stats[vehicle_id] = self._initial_stats(vehicle_id)

            worker = threading.Thread(
                target=self._vehicle_worker_loop,
                args=(vehicle_id, vehicle_queue),
                name=f"FuelWorker-{vehicle_id}",
                daemon=True,
            )
            self._workers[vehicle_id] = worker
            worker.start()

            if self.active_vehicle_id is None:
                self.active_vehicle_id = vehicle_id

            logger.info("Started FIFO worker for vehicle %s", vehicle_id)
            return vehicle_queue

    def _submit(self, vehicle_id: str, task: _QueueTask) -> Any:
        with self._lock:
            vehicle_queue = self._ensure_vehicle_worker(vehicle_id)
            vehicle_queue.put(task)
            stats = self._stats.get(vehicle_id)
            if stats is not None:
                stats["queue_size"] = vehicle_queue.qsize()

        try:
            return task.future.result(timeout=self.processing_timeout_seconds)
        except FutureTimeoutError as exc:
            task.future.cancel()
            raise TimeoutError(
                f"Timed out while processing telemetry for vehicle {vehicle_id}"
            ) from exc

    def enqueue(self, vehicle_id: str, point_data: Dict[str, Any]) -> Dict[str, Any]:
        """Submit one point to its vehicle FIFO and return its processed result."""
        future: Future = Future()
        task = _QueueTask(kind="point", future=future, point_data=dict(point_data))

        with self._lock:
            vehicle_queue = self._ensure_vehicle_worker(vehicle_id)
            stats = self._stats[vehicle_id]
            stats["total_enqueued"] += 1
            stats["last_enqueued_at"] = datetime.now().isoformat()
            vehicle_queue.put(task)
            self._stats[vehicle_id]["queue_size"] = vehicle_queue.qsize()

        try:
            result = future.result(timeout=self.processing_timeout_seconds)
        except FutureTimeoutError as exc:
            future.cancel()
            raise TimeoutError(
                f"Timed out while processing telemetry for vehicle {vehicle_id}"
            ) from exc

        return {
            "status": "ok",
            "vehicle_id": vehicle_id,
            "raw_fuel": result["raw_fuel"],
            "clean_fuel": result["clean_fuel"],
            "ai_state": result["ai_state"],
            "processed_at": datetime.now().isoformat(),
        }

    def _vehicle_worker_loop(
        self,
        vehicle_id: str,
        vehicle_queue: queue.Queue,
    ) -> None:
        """Consume one vehicle queue in strict insertion order."""
        while True:
            task = vehicle_queue.get()
            if task is None:
                vehicle_queue.task_done()
                self._update_queue_size(vehicle_id, vehicle_queue)
                return

            if not task.future.set_running_or_notify_cancel():
                vehicle_queue.task_done()
                self._update_queue_size(vehicle_id, vehicle_queue)
                continue

            result = None
            error: Optional[BaseException] = None
            try:
                if task.kind == "point":
                    result = self._process_single_point(
                        vehicle_id,
                        task.point_data or {},
                    )
                elif task.kind == "reset":
                    result = self._reset_vehicle_now(vehicle_id)
                else:
                    raise ValueError(f"Unsupported queue task: {task.kind}")
            except BaseException as exc:
                error = exc
                logger.exception(
                    "Failed to process queue task for vehicle %s",
                    vehicle_id,
                )
            finally:
                vehicle_queue.task_done()
                self._update_queue_size(vehicle_id, vehicle_queue)

            # Complete last so synchronous callers see final buffers and stats.
            if error is None:
                task.future.set_result(result)
            else:
                task.future.set_exception(error)

    def _update_queue_size(
        self,
        vehicle_id: str,
        vehicle_queue: queue.Queue,
    ) -> None:
        with self._lock:
            stats = self._stats.get(vehicle_id)
            if stats is not None:
                stats["queue_size"] = vehicle_queue.qsize()

    def _process_single_point(
        self,
        vehicle_id: str,
        data: Dict[str, Any],
    ) -> Dict[str, Any]:
        raw_fuel = float(
            data.get("raw", data.get("raw_fuel", data.get("FuelLevel", 0.0)))
        )
        speed = float(data.get("speed", data.get("Speed", 0.0)))
        lat = float(data.get("lat", data.get("Lat", 0.0)))
        lng = float(data.get("lng", data.get("Lng", 0.0)))
        time_str = str(
            data.get("time", data.get("FuelTime", datetime.now().isoformat()))
        )
        address = str(data.get("address", data.get("Address", "")))
        capacity_est = float(
            data.get("capacity_est", data.get("capacity", 200.0))
        )

        if lat > 50 and lng < 50:
            lat, lng = lng, lat

        try:
            fuel_time = datetime.fromisoformat(time_str)
        except (TypeError, ValueError):
            fuel_time = datetime.now()

        result = self.state_manager.process_point(
            vehicle_id=vehicle_id,
            fuel_time=fuel_time,
            fuel_level=raw_fuel,
            speed=speed,
            lat=lat,
            lng=lng,
            capacity_est=capacity_est,
        )
        clean_fuel = float(result["clean_fuel_liters"])
        signal_state = result.get("signal_state") or result.get(
            "ai_signal_state", "UNCERTAIN"
        )
        quality_flag = result.get("quality_flag", "VALID")

        point_payload = {
            "time": time_str,
            "timestamp": time_str,
            "speed": speed,
            "raw": raw_fuel,
            "raw_fuel": raw_fuel,
            "clean": clean_fuel,
            "kalman": data.get("kalman", clean_fuel),
            "adaptive": data.get("adaptive", clean_fuel),
            "ai_enhanced": clean_fuel,
            "ai_smooth_tracking": clean_fuel,
            "smooth_tracking": clean_fuel,
            "motion_state": result.get("motion_state", "UNCERTAIN"),
            "motion_confidence": result.get("motion_confidence", 0.0),
            "gps_displacement_meters": result.get("gps_displacement_meters", 0.0),
            "lat": lat,
            "lng": lng,
            "address": address,
            "vehicle_id": vehicle_id,
            "state": signal_state,
            "signal_state": signal_state,
            "ai_state": signal_state,
            "quality_flag": quality_flag,
        }

        with self._lock:
            self._buffers[vehicle_id].append(point_payload)
            self.global_live_buffer.append(point_payload)

            stats = self._stats.get(vehicle_id)
            if stats is not None:
                stats["total_processed"] += 1
                stats["current_fuel_raw"] = round(raw_fuel, 2)
                stats["current_fuel_clean"] = round(clean_fuel, 2)
                stats["current_fuel_smooth"] = round(clean_fuel, 2)
                stats["current_speed"] = round(speed, 1)
                stats["current_state"] = signal_state
                stats["current_motion_state"] = result.get("motion_state", "UNCERTAIN")
                stats["current_lat"] = lat
                stats["current_lng"] = lng
                stats["current_address"] = address
                stats["last_processed_at"] = datetime.now().isoformat()

        return {
            "clean_fuel": round(clean_fuel, 2),
            "raw_fuel": round(raw_fuel, 2),
            "signal_state": signal_state,
            "ai_state": signal_state,
            "quality_flag": quality_flag,
        }

    def get_vehicle_data(
        self,
        vehicle_id: Optional[str] = None,
        limit: int = 300,
    ) -> List[Dict[str, Any]]:
        target_vehicle = vehicle_id or self.active_vehicle_id
        with self._lock:
            if target_vehicle and self._buffers.get(target_vehicle):
                return list(self._buffers[target_vehicle])[-limit:]

            items = list(self.global_live_buffer)
            if target_vehicle:
                filtered = [
                    point
                    for point in items
                    if point.get("vehicle_id") == target_vehicle
                ]
                if filtered:
                    return filtered[-limit:]
            return items[-limit:]

    def get_fleet_status(self) -> List[Dict[str, Any]]:
        with self._lock:
            return [dict(stats) for stats in self._stats.values()]

    def switch_active_vehicle(self, vehicle_id: str) -> bool:
        with self._lock:
            self.active_vehicle_id = vehicle_id
        return True

    def _reset_vehicle_now(self, vehicle_id: str) -> bool:
        existed = self.state_manager.reset_vehicle_state(vehicle_id)
        with self._lock:
            buffer_existed = bool(self._buffers.get(vehicle_id))
            self._buffers.pop(vehicle_id, None)
            retained = [
                point
                for point in self.global_live_buffer
                if point.get("vehicle_id") != vehicle_id
            ]
            self.global_live_buffer.clear()
            self.global_live_buffer.extend(retained)

            if vehicle_id in self._stats:
                self._stats[vehicle_id] = self._initial_stats(vehicle_id)
        return existed or buffer_existed

    def clear_vehicle(self, vehicle_id: str) -> bool:
        """Reset after all earlier points for this vehicle have completed."""
        future: Future = Future()
        task = _QueueTask(kind="reset", future=future)
        return bool(self._submit(vehicle_id, task))

    def stop(self, join_timeout_seconds: float = 2.0) -> None:
        """Finish queued work, stop workers, and reject future submissions."""
        with self._lock:
            if not self._running:
                return
            self._running = False
            queues = list(self._queues.values())
            workers = list(self._workers.values())

        for vehicle_queue in queues:
            vehicle_queue.put(None)

        current_thread = threading.current_thread()
        for worker in workers:
            if worker is not current_thread:
                worker.join(timeout=join_timeout_seconds)
