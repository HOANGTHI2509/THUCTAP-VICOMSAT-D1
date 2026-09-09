import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta

from src.service.queue_manager import VehicleQueueManager
from src.service.state_manager import StreamingStateManager


def _wait_for_queue_size(
    manager: VehicleQueueManager,
    vehicle_id: str,
    expected: int,
    timeout: float = 2.0,
) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        with manager._lock:
            current = manager._queues[vehicle_id].qsize()
        if current >= expected:
            return
        time.sleep(0.001)
    raise AssertionError(f"Queue did not reach size {expected}")


class _BlockingRecorder:
    def __init__(self):
        self.first_started = threading.Event()
        self.release_first = threading.Event()
        self.seen = []

    def process_point(self, vehicle_id, fuel_time, fuel_level, **kwargs):
        if not self.seen:
            self.first_started.set()
            assert self.release_first.wait(timeout=2.0)
        self.seen.append(fuel_level)
        return {
            "clean_fuel_liters": fuel_level,
            "ai_signal_state": "STABLE_JITTER",
        }

    def reset_vehicle_state(self, vehicle_id):
        return True


def test_same_vehicle_requests_are_processed_in_fifo_order():
    recorder = _BlockingRecorder()
    manager = VehicleQueueManager(recorder)
    values = [200.0, 199.0, 198.0, 197.0, 196.0]

    try:
        with ThreadPoolExecutor(max_workers=len(values)) as executor:
            futures = [
                executor.submit(
                    manager.enqueue,
                    "CAR-A",
                    {"FuelTime": "2026-01-01T08:00:00", "FuelLevel": values[0]},
                )
            ]
            assert recorder.first_started.wait(timeout=2.0)

            for queue_position, value in enumerate(values[1:], start=1):
                futures.append(
                    executor.submit(
                        manager.enqueue,
                        "CAR-A",
                        {"FuelTime": "2026-01-01T08:00:00", "FuelLevel": value},
                    )
                )
                _wait_for_queue_size(manager, "CAR-A", queue_position)

            recorder.release_first.set()
            results = [future.result(timeout=2.0) for future in futures]

        assert recorder.seen == values
        assert [result["raw_fuel"] for result in results] == values
        status = manager.get_fleet_status()[0]
        assert status["total_enqueued"] == len(values)
        assert status["total_processed"] == len(values)
        assert status["queue_size"] == 0
    finally:
        recorder.release_first.set()
        manager.stop()


class _ParallelVehicleRecorder:
    def __init__(self):
        self.barrier = threading.Barrier(2)
        self.worker_names = set()
        self.lock = threading.Lock()

    def process_point(self, vehicle_id, fuel_time, fuel_level, **kwargs):
        with self.lock:
            self.worker_names.add(threading.current_thread().name)
        self.barrier.wait(timeout=2.0)
        return {
            "clean_fuel_liters": fuel_level,
            "ai_signal_state": "STABLE_JITTER",
        }

    def reset_vehicle_state(self, vehicle_id):
        return True


def test_different_vehicle_queues_process_in_parallel():
    recorder = _ParallelVehicleRecorder()
    manager = VehicleQueueManager(recorder)

    try:
        with ThreadPoolExecutor(max_workers=2) as executor:
            first = executor.submit(
                manager.enqueue,
                "CAR-A",
                {"FuelTime": "2026-01-01T08:00:00", "FuelLevel": 200.0},
            )
            second = executor.submit(
                manager.enqueue,
                "CAR-B",
                {"FuelTime": "2026-01-01T08:00:00", "FuelLevel": 500.0},
            )
            assert first.result(timeout=3.0)["clean_fuel"] == 200.0
            assert second.result(timeout=3.0)["clean_fuel"] == 500.0

        assert recorder.worker_names == {"FuelWorker-CAR-A", "FuelWorker-CAR-B"}
    finally:
        manager.stop()


def test_state_is_serial_per_vehicle_but_parallel_across_vehicles():
    manager = StreamingStateManager(model_dir="")
    original_process_point = manager.engine.process_point
    tracking_lock = threading.Lock()
    active_by_vehicle = {"CAR-A": 0, "CAR-B": 0}
    max_by_vehicle = {"CAR-A": 0, "CAR-B": 0}
    active_total = 0
    max_total = 0

    def tracked_process_point(**kwargs):
        nonlocal active_total, max_total
        vehicle_id = kwargs["vehicle_id"]
        with tracking_lock:
            active_by_vehicle[vehicle_id] += 1
            max_by_vehicle[vehicle_id] = max(
                max_by_vehicle[vehicle_id],
                active_by_vehicle[vehicle_id],
            )
            active_total += 1
            max_total = max(max_total, active_total)
        try:
            time.sleep(0.01)
            return original_process_point(**kwargs)
        finally:
            with tracking_lock:
                active_by_vehicle[vehicle_id] -= 1
                active_total -= 1

    manager.engine.process_point = tracked_process_point
    start = datetime(2026, 1, 1, 8, 0)

    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = []
        for index in range(5):
            for vehicle_id, raw_fuel in (("CAR-A", 200.0), ("CAR-B", 500.0)):
                futures.append(
                    executor.submit(
                        manager.process_point,
                        vehicle_id,
                        start + timedelta(minutes=2 * index),
                        raw_fuel - index,
                        30.0,
                    )
                )
        for future in futures:
            future.result(timeout=3.0)

    assert max_by_vehicle == {"CAR-A": 1, "CAR-B": 1}
    assert max_total >= 2
    assert manager._stats["CAR-A"]["total_points"] == 5
    assert manager._stats["CAR-B"]["total_points"] == 5


def test_reset_is_ordered_after_pending_vehicle_points():
    manager = VehicleQueueManager(StreamingStateManager(model_dir=""))

    try:
        manager.enqueue(
            "CAR-A",
            {"FuelTime": "2026-01-01T08:00:00", "FuelLevel": 200.0},
        )
        manager.enqueue(
            "CAR-A",
            {"FuelTime": "2026-01-01T08:02:00", "FuelLevel": 199.0},
        )

        assert manager.clear_vehicle("CAR-A") is True
        assert manager.get_vehicle_data("CAR-A") == []
        assert "CAR-A" not in manager.state_manager._contexts
        status = manager.get_fleet_status()[0]
        assert status["total_enqueued"] == 0
        assert status["total_processed"] == 0
    finally:
        manager.stop()
