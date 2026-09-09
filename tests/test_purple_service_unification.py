from datetime import datetime, timedelta

from src.core.filters.smooth_tracking import AISmoothTrackingFilter
from src.sdk.fuel_cleaner import FuelCleanerEngine
from src.service.queue_manager import VehicleQueueManager
from src.service.state_manager import StreamingStateManager


VALUES = [200.0, 198.5, 197.0, 199.0, 196.0, 195.0]
SPEEDS = [30.0, 35.0, 40.0, 25.0, 42.0, 38.0]
CAPACITY = 300.0
START = datetime(2026, 1, 1, 8, 0)


def _direct_purple_outputs(vehicle_id: str):
    engine = AISmoothTrackingFilter(model_dir="")
    return [
        engine.process_point(
            vehicle_id=vehicle_id,
            timestamp=START + timedelta(minutes=2 * index),
            raw_fuel=value,
            speed=SPEEDS[index],
            capacity_est=CAPACITY,
        )["clean_fuel"]
        for index, value in enumerate(VALUES)
    ]


def test_streaming_state_manager_is_a_purple_engine_adapter():
    manager = StreamingStateManager(model_dir="")
    actual = []

    for index, value in enumerate(VALUES):
        result = manager.process_point(
            vehicle_id="STATE_PATH",
            fuel_time=START + timedelta(minutes=2 * index),
            fuel_level=value,
            speed=SPEEDS[index],
            capacity_est=CAPACITY,
        )
        actual.append(result["clean_fuel_liters"])

    assert actual == _direct_purple_outputs("STATE_PATH")


def test_sdk_can_share_the_official_state_manager():
    manager = StreamingStateManager(model_dir="")
    sdk = FuelCleanerEngine(state_manager=manager)

    assert sdk.state_manager is manager

    actual = [
        sdk.clean_point(
            vehicle_id="SDK_PATH",
            timestamp=START + timedelta(minutes=2 * index),
            raw_fuel=value,
            speed=SPEEDS[index],
            capacity_est=CAPACITY,
        )["clean_fuel"]
        for index, value in enumerate(VALUES)
    ]

    assert actual == _direct_purple_outputs("SDK_PATH")


def test_push_queue_processes_each_point_once_with_purple_engine():
    manager = StreamingStateManager(model_dir="")
    queue_manager = VehicleQueueManager(manager)
    actual = []

    try:
        for index, value in enumerate(VALUES):
            result = queue_manager.enqueue(
                "PUSH_PATH",
                {
                    "time": (START + timedelta(minutes=2 * index)).isoformat(),
                    "raw": value,
                    "speed": SPEEDS[index],
                    "capacity_est": CAPACITY,
                },
            )
            actual.append(result["clean_fuel"])
    finally:
        queue_manager.stop()

    assert actual == _direct_purple_outputs("PUSH_PATH")
    assert manager._stats["PUSH_PATH"]["total_points"] == len(VALUES)
