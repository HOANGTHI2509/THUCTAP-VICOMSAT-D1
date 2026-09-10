from datetime import datetime

from fastapi.testclient import TestClient

from src.core.filters.smooth_tracking.contracts import (
    MOTION_STATES,
    QUALITY_FLAGS,
    SIGNAL_STATES,
    normalize_quality_flag,
    normalize_signal_state,
)
from src.service.api import CleanFuelOutput, app, state_manager


def test_topic1_contract_normalizes_legacy_business_names():
    assert normalize_signal_state("DRAIN") == "DOWNWARD_SHIFT"
    assert normalize_signal_state("REFUEL") == "UPWARD_SHIFT"
    assert normalize_quality_flag("DRAIN_CONFIRMED") == "DOWNWARD_SHIFT_TRACKED"
    assert normalize_quality_flag("REFUEL_TRACKED") == "UPWARD_SHIFT_TRACKED"


def test_contract_vocabularies_do_not_expose_topic2_events():
    public_values = SIGNAL_STATES + QUALITY_FLAGS + MOTION_STATES
    assert "REFUEL" not in public_values
    assert "DRAIN" not in public_values


def test_clean_output_accepts_legacy_ai_state_but_serializes_signal_state():
    output = CleanFuelOutput.model_validate(
        {
            "VehicleID": "CAR-ALIAS",
            "FuelTime": datetime(2026, 1, 1, 8, 0).isoformat(),
            "RawFuel": 200.0,
            "CleanFuel": 200.0,
            "AI_State": "STABLE_JITTER",
            "Confidence": 1.0,
            "QualityFlag": "VALID",
            "LatencyMs": 1.0,
        }
    )
    serialized = output.model_dump(by_alias=True)
    assert serialized["SignalState"] == "STABLE_JITTER"
    assert "AI_State" not in serialized


def test_realtime_api_returns_only_topic1_signal_contract():
    vehicle_id = "TOPIC1-API-CONTRACT"
    state_manager.reset_vehicle_state(vehicle_id)
    client = TestClient(app)
    response = client.post(
        "/api/v1/fuel/clean-point",
        json={
            "VehicleID": vehicle_id,
            "FuelTime": "2026-01-01T08:00:00",
            "FuelLevel": 200.0,
            "Speed": 0.0,
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["CleanFuel"] == 200.0
    assert payload["SignalState"] == "INIT"
    assert payload["QualityFlag"] == "VALID"
    assert payload["MotionState"] == "UNCERTAIN"
    assert "AI_State" not in payload
    assert "is_refuel" not in payload
    assert "is_drain" not in payload

