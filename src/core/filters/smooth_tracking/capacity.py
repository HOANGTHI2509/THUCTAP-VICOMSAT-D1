"""Authoritative calibrated tank capacities keyed by normalized VehicleID."""

from __future__ import annotations

from typing import Optional


VEHICLE_CAPACITIES_LITERS = {
    "24H04650": 800.0,
    "29E45520": 200.0,
    "29E45560": 200.0,
    "29E51878": 200.0,
    "29H41394": 350.0,
    "29H75028": 100.0,
    "35H09245": 400.0,
    "90H03494": 600.0,
    "92H03625": 200.0,
}


def normalize_vehicle_id(vehicle_id: object) -> str:
    return "".join(character for character in str(vehicle_id).upper() if character.isalnum())


def capacity_for_vehicle(vehicle_id: object) -> Optional[float]:
    return VEHICLE_CAPACITIES_LITERS.get(normalize_vehicle_id(vehicle_id))
