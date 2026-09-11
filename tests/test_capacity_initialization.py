from datetime import datetime, timedelta

import numpy as np
import pytest

from src.core.filters.smooth_tracking import AISmoothTrackingFilter
from src.db.database import DatabaseManager
from src.service.state_manager import StreamingStateManager
from src.service.queue_manager import VehicleQueueManager


def _run(values, capacities=None):
    engine = AISmoothTrackingFilter(model_dir="")
    start = datetime(2026, 1, 1)
    capacities = capacities or [None] * len(values)
    output = [engine.process_point("CAR", start + timedelta(minutes=2*i), value, capacity_est=capacities[i]) for i, value in enumerate(values)]
    return engine, output


@pytest.mark.parametrize("values,valid", [([0, 0, 85], 85), ([0, np.nan, -1, 120], 120)])
def test_first_valid_sample_is_init_and_invalid_prefix_is_discarded(values, valid):
    engine, output = _run(values)
    for row in output[:-1]:
        assert row["quality_flag"] == "INITIAL_INVALID_DISCARDED"
        assert row["operational_state"] == "UNINITIALIZED"
        assert row["clean_fuel"] is None
    assert output[-1]["ai_state"] == "INIT"
    assert output[-1]["clean_fuel"] == valid
    context = engine.contexts["CAR"]
    assert list(context.history_fuel) == [valid]
    assert context.kalman_x == context.last_raw_fuel == context.last_clean_fuel == valid


def test_zero_after_init_is_dropout_and_never_enters_raw_history():
    engine, output = _run([120, 119.8, 0, 0, 119.5])
    assert [output[i]["quality_flag"] for i in (2, 3)] == ["ZERO_DROPOUT_HELD"] * 2
    assert output[2]["clean_fuel"] == output[1]["clean_fuel"] == output[3]["clean_fuel"]
    assert 0 not in engine.contexts["CAR"].history_fuel


def test_missing_capacity_is_explicit_unknown_none():
    engine, output = _run([85])
    assert output[0]["capacity_mode"] == "UNKNOWN"
    assert output[0]["capacity_source"] == "NONE"
    assert output[0]["capacity_est"] is None
    assert engine.contexts["CAR"].capacity_est is None


@pytest.mark.parametrize("capacity", [80.0, 800.0])
def test_request_capacity_is_known_request(capacity):
    _, output = _run([50], [capacity])
    assert (output[0]["capacity_mode"], output[0]["capacity_source"], output[0]["capacity_est"]) == ("KNOWN", "REQUEST", capacity)


def test_master_data_precedes_request_and_is_marked_master_data():
    manager = StreamingStateManager(model_dir="", capacity_resolver=lambda vehicle_id: 100.0)
    result = manager.process_point("CAR", datetime(2026, 1, 1), 70, capacity_est=80)
    assert (result["capacity_mode"], result["capacity_source"], result["capacity_est"]) == ("KNOWN", "MASTER_DATA", 100.0)


def test_unknown_can_become_known_without_reset_or_history_rewrite():
    engine = AISmoothTrackingFilter(model_dir="")
    start = datetime(2026, 1, 1)
    first = engine.process_point("CAR", start, 50)
    context = engine.contexts["CAR"]
    first_identity = id(context)
    second = engine.process_point("CAR", start + timedelta(minutes=2), 49.8, capacity_est=80)
    assert first["capacity_mode"] == "UNKNOWN"
    assert (second["capacity_mode"], second["capacity_source"], second["capacity_est"]) == ("KNOWN", "REQUEST", 80.0)
    assert id(engine.contexts["CAR"]) == first_identity
    assert list(context.history_fuel) == [50, 49.8]
    assert second["clean_fuel"] > 49


def test_database_850_placeholder_is_not_trusted_master_data(tmp_path):
    manager = DatabaseManager(f"sqlite:///{tmp_path / 'capacity.db'}")
    manager.ensure_vehicle("PLACEHOLDER", capacity_liters=850)
    manager.ensure_vehicle("CONFIGURED", capacity_liters=100)
    assert manager.get_configured_vehicle_capacity("PLACEHOLDER") is None
    assert manager.get_configured_vehicle_capacity("CONFIGURED") == 100.0
    manager.ensure_vehicle("NO-CAPACITY")
    assert manager.get_configured_vehicle_capacity("NO-CAPACITY") is None


def test_queue_does_not_invent_capacity_when_missing():
    class Recorder:
        received = None
        def process_point(self, **kwargs):
            self.received = kwargs["capacity_est"]
            return {"clean_fuel_liters": 50.0, "signal_state": "INIT", "quality_flag": "VALID"}

    recorder = Recorder()
    queue = VehicleQueueManager(recorder)
    queue._process_single_point("CAR", {"raw": 50, "time": "2026-01-01T00:00:00"})
    assert recorder.received is None
