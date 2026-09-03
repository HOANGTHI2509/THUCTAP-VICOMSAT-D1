"""
Định nghĩa các bảng Cơ sở dữ liệu chuẩn 3NF (Third Normal Form) bằng SQLAlchemy.
"""
from datetime import datetime
from sqlalchemy import Column, String, Float, DateTime, Text, ForeignKey, BigInteger, Integer
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()


class Vehicle(Base):
    """
    Bảng lưu thông tin xe.
    Chuẩn 3NF: Lưu trữ dung tích bình chuẩn (capacity_liters) tại đây một lần duy nhất,
    tránh trùng lặp dữ liệu trên từng bản ghi đo đạc.
    """
    __tablename__ = "vehicles"

    vehicle_id = Column(String(50), primary_key=True, index=True, doc="Biển số hoặc mã định danh xe")
    vehicle_name = Column(String(100), nullable=True, doc="Tên gợi nhớ hoặc loại phương tiện")
    capacity_liters = Column(Float, nullable=True, default=850.0, doc="Dung tích bình chứa chuẩn (Lít)")
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    # Relationships
    logs = relationship("FuelLog", back_populates="vehicle", cascade="all, delete-orphan")
    events = relationship("FuelEvent", back_populates="vehicle", cascade="all, delete-orphan")

    def to_dict(self):
        return {
            "vehicle_id": self.vehicle_id,
            "vehicle_name": self.vehicle_name,
            "capacity_liters": self.capacity_liters,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


class AISignalState(Base):
    """
    Bảng từ điển mã trạng thái tín hiệu nhiên liệu do AI phân loại.
    Chuẩn 3NF: Tách riêng tên diễn giải tiếng Việt để tránh phụ thuộc bắc cầu:
    log_id -> state_code -> state_name_vi.
    """
    __tablename__ = "ai_signal_states"

    state_code = Column(String(50), primary_key=True, index=True, doc="Mã trạng thái chuẩn tiếng Anh")
    state_name_vi = Column(String(100), nullable=False, doc="Tên diễn giải tiếng Việt")
    description = Column(Text, nullable=True, doc="Mô tả kỹ thuật hành vi tín hiệu")

    # Relationships
    logs = relationship("FuelLog", back_populates="signal_state")

    def to_dict(self):
        return {
            "state_code": self.state_code,
            "state_name_vi": self.state_name_vi,
            "description": self.description,
        }


class FuelLog(Base):
    """
    Bảng nhật ký đo đạc chuỗi thời gian sau khi đã được AI-Kalman lọc sạch.
    Chuẩn 3NF: Mọi thuộc tính (raw_fuel, clean_fuel, speed, lat, lng) đều phụ thuộc
    trực tiếp vào khóa chính id của lần đo đạc.
    """
    __tablename__ = "fuel_logs"

    id = Column(BigInteger().with_variant(Integer, "sqlite"), primary_key=True, autoincrement=True)
    vehicle_id = Column(String(50), ForeignKey("vehicles.vehicle_id", ondelete="CASCADE"), nullable=False, index=True)
    timestamp = Column(DateTime, nullable=False, index=True, doc="Thời điểm đo đạc")
    raw_fuel = Column(Float, nullable=False, doc="Mức nhiên liệu thô từ cảm biến (Lít)")
    clean_fuel = Column(Float, nullable=False, doc="Mức nhiên liệu sau lọc sạch AI-Kalman (Lít)")
    speed = Column(Float, nullable=False, default=0.0, doc="Vận tốc GPS (km/h)")
    lat = Column(Float, nullable=True, doc="Vĩ độ GPS")
    lng = Column(Float, nullable=True, doc="Kinh độ GPS")
    state_code = Column(String(50), ForeignKey("ai_signal_states.state_code"), nullable=False, default="STABLE_JITTER")
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    # Relationships
    vehicle = relationship("Vehicle", back_populates="logs")
    signal_state = relationship("AISignalState", back_populates="logs")

    def to_dict(self):
        return {
            "id": self.id,
            "vehicle_id": self.vehicle_id,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "raw_fuel": self.raw_fuel,
            "clean_fuel": self.clean_fuel,
            "speed": self.speed,
            "lat": self.lat,
            "lng": self.lng,
            "state_code": self.state_code,
            "state_name_vi": self.signal_state.state_name_vi if self.signal_state else self.state_code,
        }


class FuelEvent(Base):
    """
    Bảng lưu trữ các sự kiện đặc biệt (Bơm đổ xăng, Rút dầu đột ngột).
    Chuẩn 3NF: Tách riêng bảng biến cố giúp tra cứu sự kiện nhanh mà không cần quét toàn bảng logs.
    """
    __tablename__ = "fuel_events"

    id = Column(BigInteger().with_variant(Integer, "sqlite"), primary_key=True, autoincrement=True)
    vehicle_id = Column(String(50), ForeignKey("vehicles.vehicle_id", ondelete="CASCADE"), nullable=False, index=True)
    event_type = Column(String(20), nullable=False, doc="REFUEL (Đổ xăng) hoặc DRAIN (Rút dầu)")
    start_time = Column(DateTime, nullable=False, doc="Thời điểm bắt đầu sự kiện")
    end_time = Column(DateTime, nullable=False, doc="Thời điểm kết thúc sự kiện")
    start_fuel = Column(Float, nullable=False, doc="Mức nhiên liệu trước sự kiện (Lít)")
    end_fuel = Column(Float, nullable=False, doc="Mức nhiên liệu sau sự kiện (Lít)")
    change_liters = Column(Float, nullable=False, doc="Độ chênh lệch nhiên liệu ròng (Lít)")
    lat = Column(Float, nullable=True)
    lng = Column(Float, nullable=True)
    address = Column(String(255), nullable=True, doc="Địa điểm diễn ra sự kiện")
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    # Relationships
    vehicle = relationship("Vehicle", back_populates="events")

    def to_dict(self):
        return {
            "id": self.id,
            "vehicle_id": self.vehicle_id,
            "event_type": self.event_type,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "start_fuel": self.start_fuel,
            "end_fuel": self.end_fuel,
            "change_liters": self.change_liters,
            "lat": self.lat,
            "lng": self.lng,
            "address": self.address,
        }
