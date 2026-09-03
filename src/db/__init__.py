"""
Module Cơ sở dữ liệu chuẩn 3NF cho hệ thống giám sát nhiên liệu.
"""
from src.db.models import Base, Vehicle, AISignalState, FuelLog, FuelEvent
from src.db.database import DatabaseManager, get_db_manager

__all__ = [
    "Base",
    "Vehicle",
    "AISignalState",
    "FuelLog",
    "FuelEvent",
    "DatabaseManager",
    "get_db_manager",
]
