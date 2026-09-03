"""
Quản lý kết nối và thao tác Cơ sở dữ liệu chuẩn 3NF sử dụng SQLAlchemy.
Hỗ trợ đa hệ quản trị (SQLite, PostgreSQL, MySQL, SQL Server) hoặc chế độ Không dùng CSDL (Zero-DB).
"""
import os
import logging
from datetime import datetime
from typing import Optional, List, Dict, Any
from sqlalchemy import create_engine, select, desc
from sqlalchemy.orm import sessionmaker, scoped_session

from src.db.models import Base, Vehicle, AISignalState, FuelLog, FuelEvent

logger = logging.getLogger("FuelDatabase")

# Dữ liệu từ điển khởi tạo mặc định cho 5 trạng thái tín hiệu AI
DEFAULT_SIGNAL_STATES = [
    {
        "state_code": "UPWARD_SHIFT",
        "state_name_vi": "Bơm/Đổ nhiên liệu",
        "description": "Mức nhiên liệu tăng vọt thực tế do được tiếp thêm dầu vào bình.",
    },
    {
        "state_code": "DOWNWARD_SHIFT",
        "state_name_vi": "Hụt dầu đột ngột",
        "description": "Mức nhiên liệu tụt giảm nhanh bất thường, nghi ngờ rút trộm hoặc rò rỉ.",
    },
    {
        "state_code": "GRADUAL_CHANGE",
        "state_name_vi": "Tiêu thụ khi chạy",
        "description": "Nhiên liệu giảm dần đều theo thời gian do động cơ tiêu hao khi xe di chuyển.",
    },
    {
        "state_code": "STABLE_JITTER",
        "state_name_vi": "Xe đỗ ổn định",
        "description": "Xe đứng yên hoặc đỗ, phao xăng có rung động nhỏ quanh giá trị thực tế.",
    },
    {
        "state_code": "OSCILLATION_NOISE",
        "state_name_vi": "Nhiễu sóng sánh/xung",
        "description": "Nhiễu cơ học do quán tính dầu sóng sánh trong bình hoặc xung điện cảm biến.",
    },
]


class DatabaseManager:
    """
    Lớp quản lý kết nối CSDL chuẩn 3NF độc lập và an toàn luồng (thread-safe).
    """

    def __init__(self, database_url: Optional[str] = None):
        """
        Khởi tạo kết nối CSDL.
        Nếu database_url là None, rỗng hoặc 'NONE', hệ thống chạy ở chế độ Zero-DB (không lưu trữ).
        """
        raw_url = database_url if database_url is not None else os.getenv("DATABASE_URL", "sqlite:///./fuel_records.db")
        raw_url = raw_url.strip() if raw_url else ""

        if raw_url.upper() in {"", "NONE", "FALSE", "OFF"}:
            self.enabled = False
            self.engine = None
            self.session_factory = None
            logger.info("DatabaseManager khởi động ở chế độ: ZERO-DB (Không sử dụng CSDL)")
            return

        self.enabled = True
        self.database_url = raw_url

        connect_args = {}
        if self.database_url.startswith("sqlite"):
            connect_args["check_same_thread"] = False
            # Tự động tạo thư mục chứa file SQLite nếu đường dẫn có thư mục
            if "///" in self.database_url:
                file_path = self.database_url.split("///", 1)[1]
                dir_name = os.path.dirname(file_path)
                if dir_name and not os.path.exists(dir_name):
                    os.makedirs(dir_name, exist_ok=True)

        self.engine = create_engine(
            self.database_url,
            connect_args=connect_args,
            pool_pre_ping=True if not self.database_url.startswith("sqlite") else False,
        )
        self.session_factory = scoped_session(sessionmaker(bind=self.engine, expire_on_commit=False))

        # Tự động tạo bảng và nạp dữ liệu từ điển
        self._init_db()

    def _init_db(self):
        """Tự động tạo các bảng 3NF và nạp sẵn từ điển trạng thái."""
        if not self.enabled or not self.engine:
            return
        try:
            Base.metadata.create_all(bind=self.engine)
            self._seed_signal_states()
            logger.info(f"DatabaseManager kết nối thành công tới: {self.database_url.split('@')[-1]}")
        except Exception as e:
            logger.error(f"Lỗi khởi tạo CSDL: {e}")

    def _seed_signal_states(self):
        """Nạp sẵn danh mục 5 trạng thái tín hiệu nếu chưa có."""
        if not self.enabled:
            return
        session = self.session_factory()
        try:
            for item in DEFAULT_SIGNAL_STATES:
                existing = session.get(AISignalState, item["state_code"])
                if not existing:
                    state = AISignalState(**item)
                    session.add(state)
            session.commit()
        except Exception as e:
            session.rollback()
            logger.warning(f"Lỗi seed từ điển trạng thái: {e}")
        finally:
            session.close()

    def ensure_vehicle(self, vehicle_id: str, capacity_liters: Optional[float] = 850.0, vehicle_name: Optional[str] = None):
        """Đảm bảo xe đã tồn tại trong bảng vehicles (Chuẩn 3NF)."""
        if not self.enabled:
            return
        session = self.session_factory()
        try:
            vehicle = session.get(Vehicle, vehicle_id)
            if not vehicle:
                vehicle = Vehicle(
                    vehicle_id=vehicle_id,
                    vehicle_name=vehicle_name or f"Xe {vehicle_id}",
                    capacity_liters=capacity_liters or 850.0,
                )
                session.add(vehicle)
                session.commit()
            elif capacity_liters and vehicle.capacity_liters != capacity_liters:
                vehicle.capacity_liters = capacity_liters
                session.commit()
        except Exception as e:
            session.rollback()
            logger.warning(f"Lỗi ensure_vehicle '{vehicle_id}': {e}")
        finally:
            session.close()

    def save_measurement(
        self,
        vehicle_id: str,
        timestamp: datetime | str,
        raw_fuel: float,
        clean_fuel: float,
        speed: float = 0.0,
        lat: Optional[float] = None,
        lng: Optional[float] = None,
        state_code: str = "STABLE_JITTER",
        capacity_est: Optional[float] = 850.0,
    ) -> Optional[int]:
        """
        Lưu một điểm đo đạc sau khi lọc sạch vào bảng fuel_logs chuẩn 3NF.
        """
        if not self.enabled:
            return None

        if isinstance(timestamp, str):
            try:
                dt = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
            except Exception:
                try:
                    dt = datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S")
                except Exception:
                    dt = datetime.utcnow()
        else:
            dt = timestamp

        # Đảm bảo xe đã có trong bảng vehicles
        self.ensure_vehicle(vehicle_id=vehicle_id, capacity_liters=capacity_est)

        session = self.session_factory()
        try:
            log_entry = FuelLog(
                vehicle_id=vehicle_id,
                timestamp=dt,
                raw_fuel=float(raw_fuel),
                clean_fuel=float(clean_fuel),
                speed=float(speed) if speed is not None else 0.0,
                lat=float(lat) if lat is not None else None,
                lng=float(lng) if lng is not None else None,
                state_code=state_code if state_code in {s["state_code"] for s in DEFAULT_SIGNAL_STATES} else "STABLE_JITTER",
            )
            session.add(log_entry)
            session.commit()
            return log_entry.id
        except Exception as e:
            session.rollback()
            logger.error(f"Lỗi lưu điểm đo fuel_logs cho xe {vehicle_id}: {e}")
            return None
        finally:
            session.close()

    def save_event(
        self,
        vehicle_id: str,
        event_type: str,
        start_time: datetime | str,
        end_time: datetime | str,
        start_fuel: float,
        end_fuel: float,
        change_liters: float,
        lat: Optional[float] = None,
        lng: Optional[float] = None,
        address: Optional[str] = None,
        capacity_est: Optional[float] = 850.0,
    ) -> Optional[int]:
        """
        Lưu sự kiện đặc biệt (REFUEL hoặc DRAIN) vào bảng fuel_events chuẩn 3NF.
        """
        if not self.enabled:
            return None

        def parse_dt(val):
            if isinstance(val, str):
                for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S"):
                    try:
                        return datetime.strptime(val.split(".")[0], fmt)
                    except Exception:
                        pass
                return datetime.utcnow()
            return val

        dt_start = parse_dt(start_time)
        dt_end = parse_dt(end_time)

        self.ensure_vehicle(vehicle_id=vehicle_id, capacity_liters=capacity_est)

        session = self.session_factory()
        try:
            event_entry = FuelEvent(
                vehicle_id=vehicle_id,
                event_type=event_type.upper(),
                start_time=dt_start,
                end_time=dt_end,
                start_fuel=float(start_fuel),
                end_fuel=float(end_fuel),
                change_liters=float(change_liters),
                lat=float(lat) if lat is not None else None,
                lng=float(lng) if lng is not None else None,
                address=address,
            )
            session.add(event_entry)
            session.commit()
            return event_entry.id
        except Exception as e:
            session.rollback()
            logger.error(f"Lỗi lưu sự kiện fuel_events cho xe {vehicle_id}: {e}")
            return None
        finally:
            session.close()

    def get_logs(
        self,
        vehicle_id: str,
        start_time: Optional[datetime | str] = None,
        end_time: Optional[datetime | str] = None,
        limit: int = 1000,
    ) -> List[Dict[str, Any]]:
        """Lấy danh sách điểm đo sạch phục vụ báo cáo hoặc vẽ biểu đồ."""
        if not self.enabled:
            return []
        session = self.session_factory()
        try:
            query = select(FuelLog).where(FuelLog.vehicle_id == vehicle_id)
            if start_time:
                query = query.where(FuelLog.timestamp >= start_time)
            if end_time:
                query = query.where(FuelLog.timestamp <= end_time)
            query = query.order_by(FuelLog.timestamp.asc()).limit(limit)
            results = session.execute(query).scalars().all()
            return [r.to_dict() for r in results]
        finally:
            session.close()

    def get_events(self, vehicle_id: str, limit: int = 100) -> List[Dict[str, Any]]:
        """Lấy danh sách các sự kiện đổ xăng / rút dầu."""
        if not self.enabled:
            return []
        session = self.session_factory()
        try:
            query = (
                select(FuelEvent)
                .where(FuelEvent.vehicle_id == vehicle_id)
                .order_by(desc(FuelEvent.start_time))
                .limit(limit)
            )
            results = session.execute(query).scalars().all()
            return [r.to_dict() for r in results]
        finally:
            session.close()

    def close(self):
        """Đóng toàn bộ kết nối và giải phóng engine."""
        if hasattr(self, "session_factory") and self.session_factory:
            self.session_factory.remove()
        if hasattr(self, "engine") and self.engine:
            self.engine.dispose()


# Singleton instance để sử dụng toàn hệ thống
_global_db_manager: Optional[DatabaseManager] = None


def get_db_manager(database_url: Optional[str] = None) -> DatabaseManager:
    """Hàm lấy đối tượng DatabaseManager duy nhất toàn ứng dụng."""
    global _global_db_manager
    if _global_db_manager is None:
        _global_db_manager = DatabaseManager(database_url)
    return _global_db_manager
