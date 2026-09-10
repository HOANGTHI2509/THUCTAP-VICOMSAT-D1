import sys
from datetime import datetime, timedelta
from types import SimpleNamespace

from src.service.state_manager import StreamingStateManager
from src.service.state_store import (
    FallbackStateStore,
    MemoryStateStore,
    RedisStateStore,
    STATE_SCHEMA_VERSION,
    StateStore,
)


START = datetime(2026, 1, 1, 8, 0)


def _process(manager, vehicle_id, index, value):
    return manager.process_point(
        vehicle_id=vehicle_id,
        fuel_time=START + timedelta(minutes=2 * index),
        fuel_level=value,
        speed=30.0,
        capacity_est=500.0,
    )


def test_memory_store_expires_documents_by_ttl():
    now = [1000.0]
    store = MemoryStateStore(clock=lambda: now[0])
    store.save("CAR", {"schema_version": 1}, ttl_seconds=10)
    assert store.load("CAR") == {"schema_version": 1}
    now[0] = 1011.0
    assert store.load("CAR") is None


def test_new_manager_restores_complete_vehicle_state():
    store = MemoryStateStore()
    values = [500.0, 497.0, 494.0, 491.0]

    first_manager = StreamingStateManager(
        model_dir="",
        state_store=store,
        state_ttl_seconds=3600,
    )
    for index, value in enumerate(values[:3]):
        _process(first_manager, "RESTORED", index, value)

    document = store.load("RESTORED")
    assert document["schema_version"] == STATE_SCHEMA_VERSION
    assert document["context"]["history_fuel"] == values[:3]

    restarted_manager = StreamingStateManager(
        model_dir="",
        state_store=store,
        state_ttl_seconds=3600,
    )
    restored_result = _process(restarted_manager, "RESTORED", 3, values[3])

    uninterrupted = StreamingStateManager(model_dir="")
    expected_result = None
    for index, value in enumerate(values):
        expected_result = _process(uninterrupted, "REFERENCE", index, value)

    assert restored_result["clean_fuel_liters"] == expected_result["clean_fuel_liters"]
    assert restarted_manager._stats["RESTORED"]["total_points"] == 4


def test_vehicle_states_are_isolated_and_reset_deletes_persisted_state():
    store = MemoryStateStore()
    manager = StreamingStateManager(model_dir="", state_store=store)
    _process(manager, "CAR-A", 0, 200.0)
    _process(manager, "CAR-B", 0, 600.0)

    assert store.load("CAR-A")["context"]["last_clean_fuel"] == 200.0
    assert store.load("CAR-B")["context"]["last_clean_fuel"] == 600.0
    assert manager.reset_vehicle_state("CAR-A") is True
    assert store.load("CAR-A") is None
    assert store.load("CAR-B") is not None


def test_memory_backend_health_is_reported_without_secrets():
    manager = StreamingStateManager(model_dir="", state_store=MemoryStateStore())
    health = manager.state_store_health()
    assert health["backend"] == "memory"
    assert health["status"] == "healthy"


class _UnavailableStore(StateStore):
    backend_name = "unavailable_test_backend"

    def load(self, vehicle_id):
        raise ConnectionError(vehicle_id)

    def save(self, vehicle_id, state, ttl_seconds):
        raise ConnectionError(vehicle_id)

    def delete(self, vehicle_id):
        raise ConnectionError(vehicle_id)

    def health_check(self):
        raise ConnectionError("offline")


def test_unavailable_persistent_store_degrades_to_memory_without_stopping_filter():
    store = FallbackStateStore(_UnavailableStore())
    manager = StreamingStateManager(model_dir="", state_store=store)

    result = _process(manager, "FALLBACK-CAR", 0, 300.0)

    assert result["clean_fuel_liters"] == 300.0
    assert store.load("FALLBACK-CAR")["context"]["last_clean_fuel"] == 300.0
    health = store.health_check()
    assert health["status"] == "degraded"
    assert health["primary"]["status"] == "unhealthy"
    assert health["fallback"]["status"] == "healthy"


def test_redis_store_uses_versioned_json_and_expiry(monkeypatch):
    class FakeClient:
        def __init__(self):
            self.values = {}
            self.last_expiry = None

        def get(self, key):
            return self.values.get(key)

        def set(self, key, value, ex):
            self.values[key] = value
            self.last_expiry = ex

        def delete(self, key):
            return int(self.values.pop(key, None) is not None)

        def ping(self):
            return True

    client = FakeClient()

    class FakeRedis:
        @staticmethod
        def from_url(redis_url, decode_responses):
            assert redis_url == "redis://unit-test/0"
            assert decode_responses is True
            return client

    monkeypatch.setitem(sys.modules, "redis", SimpleNamespace(Redis=FakeRedis))
    store = RedisStateStore("redis://unit-test/0")
    document = {"schema_version": STATE_SCHEMA_VERSION, "value": 250.0}

    store.save("CAR", document, ttl_seconds=90)

    assert client.last_expiry == 90
    assert store.load("CAR") == document
    assert store.health_check()["status"] == "healthy"
    assert store.delete("CAR") is True
    assert store.load("CAR") is None
