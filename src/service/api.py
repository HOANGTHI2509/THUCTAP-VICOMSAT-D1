from __future__ import annotations

import os
import time
import io
import logging
import collections
from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Request, Response, status
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, ConfigDict, AliasChoices

from src.service.state_manager import StreamingStateManager
from src.service.queue_manager import VehicleQueueManager
from src.db.database import get_db_manager
from src.sdk.fuel_cleaner import FuelCleanerEngine

logger = logging.getLogger("FuelAPI")

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# 1. Khởi tạo FastAPI App & Cấu hình bảo mật
API_KEY = os.getenv("API_KEY", "vcomsat_secret_key_2026")
REQUIRE_API_KEY = os.getenv("REQUIRE_API_KEY", "false").lower() in ("true", "1", "yes")
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./fuel_records.db")

app = FastAPI(
    title="VCOMSAT Real-time Fuel Denoising & Filtering API",
    description="Microservice thời gian thực (Causal AI + Adaptive Kalman) khử nhiễu dữ liệu cảm biến nhiên liệu cho VCOMSAT.",
    version="1.0.0",
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def security_api_key_middleware(request: Request, call_next):
    """
    Middleware xác thực API Key cho môi trường Doanh nghiệp.
    Mặc định REQUIRE_API_KEY=false để môi trường Local Demo không bị chặn.
    Khi REQUIRE_API_KEY=true, bảo vệ toàn bộ các endpoint ghi/lọc dữ liệu:
    - Mọi endpoint /api/v1/* (ngoại trừ /api/v1/health)
    - /api/push, /api/switch-vehicle, /api/export_history
    """
    path = request.url.path
    if REQUIRE_API_KEY:
        # Whitelist các endpoint công khai không cần API Key
        is_public = (
            path == "/api/v1/health"
            or path in ("/docs", "/redoc", "/openapi.json")
            or path.startswith(("/docs/", "/static/"))
        )
        is_protected = not is_public and (
            path.startswith("/api/v1/")
            or path in ("/api/push", "/api/switch-vehicle", "/api/export_history")
            or path.startswith("/api/push")
        )
        if is_protected:
            api_key_header = request.headers.get("X-API-Key")
            auth_header = request.headers.get("Authorization")
            token = None
            if api_key_header:
                token = api_key_header.strip()
            elif auth_header and auth_header.startswith("Bearer "):
                token = auth_header.split(" ", 1)[1].strip()

            if not token or token != API_KEY:
                return JSONResponse(
                    status_code=401,
                    content={
                        "status": "error",
                        "error_code": "UNAUTHORIZED",
                        "message": "API Key không hợp lệ hoặc thiếu trong Header X-API-Key (hoặc Authorization: Bearer <KEY>).",
                    },
                )
    return await call_next(request)


# 2. Khởi tạo In-Memory State Manager & SDK & DB 3NF
state_manager = StreamingStateManager()
queue_manager = VehicleQueueManager(state_manager=state_manager)
sdk_engine = FuelCleanerEngine(state_manager=state_manager)
db_manager = get_db_manager(DATABASE_URL)


# 3. Pydantic Schemas Doanh Nghiệp (Hỗ trợ Bí danh Tiếng Anh / Tiếng Việt)
class EnterprisePointInput(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    vehicle_id: str = Field(..., validation_alias=AliasChoices("vehicle_id", "bien_so", "car_id", "VehicleID"), json_schema_extra={"example": "29H-75028"})
    timestamp: str = Field(..., validation_alias=AliasChoices("timestamp", "time", "thoi_gian", "FuelTime"), json_schema_extra={"example": "2026-08-15 09:16:00"})
    raw_fuel: float = Field(..., validation_alias=AliasChoices("raw_fuel", "fuel", "xang_tho", "FuelLevel", "raw"), json_schema_extra={"example": 93.3})
    speed: float = Field(0.0, validation_alias=AliasChoices("speed", "van_toc", "Speed"), json_schema_extra={"example": 61.0})
    distance_m: float = Field(0.0, validation_alias=AliasChoices("distance_m", "quang_duong", "DistanceMeters"), json_schema_extra={"example": 0.0})
    lat: Optional[float] = Field(None, validation_alias=AliasChoices("lat", "vi_do", "Lat"), json_schema_extra={"example": 21.0285})
    lng: Optional[float] = Field(None, validation_alias=AliasChoices("lng", "kinh_do", "Lng"), json_schema_extra={"example": 105.8542})
    address: Optional[str] = Field(None, validation_alias=AliasChoices("address", "dia_chi", "Address"))
    capacity_est: Optional[float] = Field(None, validation_alias=AliasChoices("capacity_est", "CapacityEst", "dung_tich", "capacity"), json_schema_extra={"example": 95.0})


class EnterpriseBatchInput(BaseModel):
    points: List[EnterprisePointInput]


class FuelPointInput(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    vehicle_id: str = Field(..., alias="VehicleID", json_schema_extra={"example": "29E-45520"}, description="Mã định danh hoặc biển số xe")
    fuel_time: datetime = Field(..., alias="FuelTime", json_schema_extra={"example": "2026-08-27T10:00:00"}, description="Thời điểm ghi nhận telemetry")
    fuel_level: float = Field(..., alias="FuelLevel", json_schema_extra={"example": 105.2}, description="Mức nhiên liệu đo thô từ cảm biến (Lít)")
    speed: float = Field(0.0, alias="Speed", json_schema_extra={"example": 45.0}, description="Vận tốc GPS của xe (km/h)")
    lat: Optional[float] = Field(None, alias="Lat", json_schema_extra={"example": 21.0285}, description="Vĩ độ GPS")
    lng: Optional[float] = Field(None, alias="Lng", json_schema_extra={"example": 105.8542}, description="Kinh độ GPS")
    address: Optional[str] = Field(None, alias="Address", description="Địa chỉ GPS (nếu có)")
    
    # Các trường mở rộng (Không bắt buộc Vcomsat phải gửi, tự fallback mặc định)
    distance_meters: float = Field(0.0, json_schema_extra={"example": 350.0}, description="Quãng đường (không bắt buộc)")
    segment_id: Optional[str] = Field(None, json_schema_extra={"example": "SEG_01"})
    capacity_est: Optional[float] = Field(
        None,
        validation_alias=AliasChoices("capacity_est", "CapacityEst", "capacity"),
        json_schema_extra={"example": 200.0},
    )
    noise_sigma_liters: Optional[float] = Field(0.8, json_schema_extra={"example": 0.8})


class CleanFuelOutput(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    vehicle_id: str = Field(..., alias="VehicleID")
    fuel_time: str = Field(..., alias="FuelTime")
    raw_fuel_liters: float = Field(..., alias="RawFuel")
    clean_fuel_liters: float = Field(..., alias="CleanFuel")
    signal_state: str = Field(
        ...,
        validation_alias=AliasChoices(
            "signal_state",
            "ai_signal_state",
            "SignalState",
            "AI_State",
        ),
        serialization_alias="SignalState",
    )
    confidence: float = Field(..., alias="Confidence")
    quality_flag: str = Field(..., alias="QualityFlag")
    latency_ms: float = Field(..., alias="LatencyMs")
    motion_state: str = Field("UNCERTAIN", alias="MotionState")
    motion_confidence: float = Field(0.0, alias="MotionConfidence")
    gps_displacement_meters: float = Field(0.0, alias="GpsDisplacementMeters")


class FuelBatchInput(BaseModel):
    vehicle_id: str = Field(..., json_schema_extra={"example": "29E-45520"})
    points: List[FuelPointInput]


class CleanBatchOutput(BaseModel):
    vehicle_id: str
    total_points: int
    results: List[CleanFuelOutput]
    total_latency_ms: float


# 4. API Endpoints
@app.get("/api/v1/health", tags=["System"])
def health_check() -> Dict[str, Any]:
    """Kiểm tra trạng thái sức khỏe của Microservice."""
    return {
        "status": "healthy",
        "service": "vicomsat-fuel-cleaning-service",
        "version": "1.0.0",
        "model_loaded": state_manager.model is not None,
        "model_type": "AI_Smooth_Tracking_Causal",
        "active_vehicles_in_memory": state_manager.active_vehicle_count,
        "state_store": state_manager.state_store_health(),
        "database_enabled": db_manager.enabled,
        "api_key_required": REQUIRE_API_KEY,
        "timestamp": datetime.now().isoformat(),
    }


@app.post("/api/v1/clean", response_model=CleanFuelOutput, tags=["Enterprise API"])
def enterprise_clean_point(point: EnterprisePointInput) -> CleanFuelOutput:
    """
    Endpoint chuẩn Doanh nghiệp: Nhận 1 điểm đo thô (hỗ trợ bí danh tiếng Anh/Việt)
    -> Lọc AI-Kalman thời gian thực -> Tự lưu phép đo vào CSDL 3NF (nếu bật DB)
    -> Trả ngay JSON theo contract Đề tài 1.
    """
    res = sdk_engine.clean_point(
        vehicle_id=point.vehicle_id,
        timestamp=point.timestamp,
        raw_fuel=point.raw_fuel,
        speed=point.speed,
        distance_m=point.distance_m,
        lat=point.lat,
        lng=point.lng,
        capacity_est=point.capacity_est,
    )

    if db_manager.enabled:
        db_manager.save_measurement(
            vehicle_id=point.vehicle_id,
            timestamp=point.timestamp,
            raw_fuel=point.raw_fuel,
            clean_fuel=res["clean_fuel"],
            speed=point.speed,
            lat=point.lat,
            lng=point.lng,
            state_code=res["signal_state"],
            capacity_est=point.capacity_est,
        )
    return CleanFuelOutput(
        vehicle_id=res["vehicle_id"],
        fuel_time=res["timestamp"],
        raw_fuel_liters=res["raw_fuel"],
        clean_fuel_liters=res["clean_fuel"],
        signal_state=res["signal_state"],
        confidence=res["confidence"],
        quality_flag=res["quality_flag"],
        latency_ms=res["processing_time_ms"],
        motion_state=res["motion_state"],
        motion_confidence=res["motion_confidence"],
        gps_displacement_meters=res["gps_displacement_meters"],
    )


@app.post("/api/v1/clean-batch", response_model=List[CleanFuelOutput], tags=["Enterprise API"])
def enterprise_clean_batch(batch: EnterpriseBatchInput) -> List[CleanFuelOutput]:
    """
    Endpoint chuẩn Doanh nghiệp: Xử lý hàng loạt theo mảng (10-500 điểm)
    cho các thiết bị truyền dữ liệu theo cụm.
    """
    results = []
    for pt in batch.points:
        r = enterprise_clean_point(pt)
        results.append(r)
    return results


@app.get(
    "/api/v1/events",
    tags=["Legacy"],
    deprecated=True,
    include_in_schema=False,
)
def get_vehicle_events(vehicle_id: str, limit: int = 100) -> List[Dict[str, Any]]:
    """Legacy downstream records; the Topic 1 service never creates them."""
    if not db_manager.enabled:
        return []
    return db_manager.get_events(vehicle_id=vehicle_id, limit=limit)


@app.post("/api/v1/fuel/clean-point", response_model=CleanFuelOutput, tags=["Denoising Pipeline"])
def clean_fuel_point(point: FuelPointInput) -> CleanFuelOutput:
    """Nhận 1 điểm dữ liệu telemetry streaming và trả về giá trị đã làm sạch trong thời gian thực."""
    try:
        res = state_manager.process_point(
            vehicle_id=point.vehicle_id,
            fuel_time=point.fuel_time,
            fuel_level=point.fuel_level,
            speed=point.speed,
            lat=point.lat,
            lng=point.lng,
            distance_meters=point.distance_meters,
            segment_id=point.segment_id,
            capacity_est=point.capacity_est,
            noise_sigma_liters=point.noise_sigma_liters,
        )
        LIVE_STREAM_BUFFER.append({
            "time": point.fuel_time.isoformat() if hasattr(point.fuel_time, "isoformat") else str(point.fuel_time),
            "speed": point.speed,
            "raw": point.fuel_level,
            "kalman": res["clean_fuel_liters"],
            "adaptive": res["clean_fuel_liters"],
            "ai_enhanced": res["clean_fuel_liters"],
            "ai_smooth_tracking": res["clean_fuel_liters"],
            "smooth_tracking": res["clean_fuel_liters"],
            "signal_state": res["signal_state"],
            "quality_flag": res["quality_flag"],
            "motion_state": res["motion_state"],
            "motion_confidence": res["motion_confidence"],
            "gps_displacement_meters": res["gps_displacement_meters"],
            "lat": point.lat or 0.0,
            "lng": point.lng or 0.0,
            "address": point.address or "",
        })
        return CleanFuelOutput(**res)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Lỗi xử lý điểm dữ liệu: {str(e)}",
        )


@app.post("/api/v1/fuel/clean-batch", response_model=CleanBatchOutput, tags=["Denoising Pipeline"])
def clean_fuel_batch(batch: FuelBatchInput) -> CleanBatchOutput:
    """Nhận một gói danh sách các điểm telemetry liên tiếp của 1 xe và xử lý theo luồng thời gian thực."""
    t_start = time.perf_counter()
    results: List[CleanFuelOutput] = []

    # Sort theo thời gian để đảm bảo thứ tự streaming
    sorted_points = sorted(batch.points, key=lambda p: p.fuel_time)

    for pt in sorted_points:
        res = state_manager.process_point(
            vehicle_id=batch.vehicle_id,
            fuel_time=pt.fuel_time,
            fuel_level=pt.fuel_level,
            speed=pt.speed,
            lat=pt.lat,
            lng=pt.lng,
            distance_meters=pt.distance_meters,
            segment_id=pt.segment_id,
            capacity_est=pt.capacity_est,
            noise_sigma_liters=pt.noise_sigma_liters,
        )
        results.append(CleanFuelOutput(**res))

    total_latency = (time.perf_counter() - t_start) * 1000.0

    return CleanBatchOutput(
        vehicle_id=batch.vehicle_id,
        total_points=len(results),
        results=results,
        total_latency_ms=round(total_latency, 3),
    )


# Quản lý luồng và hàng chờ đa xe qua VehicleQueueManager
LIVE_STREAM_BUFFER = queue_manager.global_live_buffer
VEHICLE_STREAM_BUFFERS = queue_manager._buffers


@app.get("/api/data", tags=["Web App Live Demo"])
def get_live_data(vehicle_id: Optional[str] = None, limit: int = 300) -> List[Dict[str, Any]]:
    """Lấy dữ liệu telemetry realtime mới nhất theo xe được chọn từ hàng chờ/buffer."""
    return queue_manager.get_vehicle_data(vehicle_id=vehicle_id, limit=limit)


@app.post("/api/push", tags=["Web App Live Demo"])
def push_live_point(data: Dict[str, Any]) -> Dict[str, Any]:
    """Nhận điểm dữ liệu từ script mô phỏng hoặc thiết bị xe và đưa vào hàng chờ xử lý độc lập."""
    vid = str(data.get("vehicle_id", data.get("VehicleID", "24H-04650")))
    res = queue_manager.enqueue(vid, data)
    return res


@app.get("/api/fleet/status", tags=["Web App Live Demo"])
def get_fleet_status() -> List[Dict[str, Any]]:
    """Giám sát tình trạng hàng đợi và dữ liệu realtime của toàn bộ các xe trong đội xe."""
    return queue_manager.get_fleet_status()


@app.post("/api/switch-vehicle", tags=["Web App Live Demo"])
def switch_live_vehicle(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Chuyển đổi xe hiển thị trực tiếp từ giao diện Web App."""
    vehicle_id = str(payload.get("vehicle_id", "24H-04650"))
    queue_manager.switch_active_vehicle(vehicle_id)
    cur_data = queue_manager.get_vehicle_data(vehicle_id)
    return {"status": "switched", "vehicle_id": vehicle_id, "total_points": len(cur_data)}


@app.get("/api/vehicles", tags=["Web App Live Demo"])
def list_available_vehicles() -> List[Dict[str, Any]]:
    """Trả về danh sách các xe thực tế đang có luồng dữ liệu bắn từ simulate_live_car hoặc thiết bị."""
    vehicles = []
    with queue_manager._lock:
        for vid, buf in queue_manager._buffers.items():
            if len(buf) > 0:
                first_pt = buf[0]
                last_pt = buf[-1]
                s_time = str(first_pt.get("time", first_pt.get("timestamp", "")))
                e_time = str(last_pt.get("time", last_pt.get("timestamp", "")))
                vehicles.append({
                    "id": vid,
                    "name": f"Xe {vid}",
                    "start_date": s_time[:10] if len(s_time) >= 10 else "2026-08-10",
                    "end_date": e_time[:10] if len(e_time) >= 10 else "2026-08-18",
                    "start_time": s_time,
                    "end_time": e_time,
                    "total_points": len(buf),
                    "is_streaming": True,
                })
    return vehicles


@app.get("/api/current-vehicle", tags=["Web App Live Demo"])
def get_current_vehicle() -> Dict[str, Any]:
    """Trả về biển số xe đang nhận luồng dữ liệu telemetry thời gian thực."""
    with queue_manager._lock:
        active_vid = queue_manager.active_vehicle_id
        if active_vid and active_vid in queue_manager._buffers and len(queue_manager._buffers[active_vid]) > 0:
            return {"vehicle_id": active_vid, "total_points": len(queue_manager._buffers[active_vid])}
        for vid, buf in queue_manager._buffers.items():
            if len(buf) > 0:
                return {"vehicle_id": vid, "total_points": len(buf)}
    return {"vehicle_id": None, "total_points": 0}


@app.get("/api/history", tags=["Web App Live Demo"])
def get_history(vehicle_id: str = "24H-04650", start_date: str = "", end_date: str = "", limit: int = 20000) -> List[Dict[str, Any]]:
    """
    Tra cứu dữ liệu thời gian thực đã bắn và đã qua bộ lọc AI của xe.
    Chỉ hiển thị các điểm thực tế đã nhận được, không nạp trước dữ liệu tĩnh.
    """
    with queue_manager._lock:
        buf = queue_manager._buffers.get(vehicle_id)
        if not buf or len(buf) == 0:
            # Thử đối soát không phân biệt hoa thường
            found_vid = None
            for vid in queue_manager._buffers:
                if vid.lower() == vehicle_id.lower() or vehicle_id in vid or vid in vehicle_id:
                    found_vid = vid
                    break
            if found_vid and len(queue_manager._buffers[found_vid]) > 0:
                pts = list(queue_manager._buffers[found_vid])
            else:
                return []
        else:
            pts = list(buf)

    # Lọc theo khoảng ngày nếu người dùng chọn
    if start_date or end_date:
        filtered = []
        for p in pts:
            t = str(p.get("time", p.get("timestamp", "")))
            if not t:
                filtered.append(p)
                continue
            p_date = t[:10]
            if start_date and p_date < start_date:
                continue
            if end_date and p_date > end_date:
                continue
            filtered.append(p)
        return filtered[-limit:]

    return pts[-limit:]


@app.get("/api/export_history", tags=["Web App Live Demo"])
def export_history(vehicle_id: str = "24H-04650", start_date: str = "", end_date: str = ""):
    """Xuất dữ liệu lịch sử đã lọc ra file Excel chuẩn doanh nghiệp (.xlsx)."""
    import io
    import pandas as pd
    from fastapi.responses import Response
    
    history_data = get_history(vehicle_id=vehicle_id, start_date=start_date, end_date=end_date)
    if not history_data:
        raise HTTPException(status_code=404, detail="Không tìm thấy dữ liệu để xuất")

    # Chuẩn hóa các cột báo cáo chuyên nghiệp
    rows = []
    for idx, pt in enumerate(history_data, 1):
        raw_val = round(float(pt.get("raw_fuel", pt.get("raw", 0.0))), 2)
        clean_val = round(float(pt.get("ai_enhanced", pt.get("clean", raw_val))), 2)
        noise_val = round(raw_val - clean_val, 2)
        
        # Đánh giá trạng thái tín hiệu
        if abs(noise_val) > 4.0:
            status_label = "Nhiễu cảm biến mạnh (Spike)"
        elif abs(noise_val) > 1.5:
            status_label = "Sóng sánh dao động (Sloshing)"
        else:
            status_label = "Ổn định (Normal)"

        rows.append({
            "STT": idx,
            "Biển số xe": pt.get("vehicle_id", vehicle_id),
            "Thời gian": pt.get("timestamp", pt.get("time", "")),
            "Mức xăng thô (Lít)": raw_val,
            "Mức xăng AI-Kalman (Lít)": clean_val,
            "Độ lệch nhiễu (Lít)": noise_val,
            "Trạng thái tín hiệu": status_label,
            "Vận tốc (km/h)": pt.get("speed", 0),
            "Vĩ độ (Lat)": pt.get("lat", 0.0),
            "Kinh độ (Lng)": pt.get("lng", 0.0),
            "Địa chỉ / Vị trí": pt.get("address", "")
        })

    df_clean = pd.DataFrame(rows)
    
    excel_buffer = io.BytesIO()
    with pd.ExcelWriter(excel_buffer, engine="openpyxl") as writer:
        df_clean.to_excel(writer, index=False, sheet_name="Báo Cáo Nhiên Liệu")

    excel_buffer.seek(0)
    filename = f"Bao_Cao_Nhien_Lieu_{vehicle_id}.xlsx"
    
    return Response(
        content=excel_buffer.getvalue(),
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": f"attachment; filename={filename}"}
    )


@app.get("/api/v1/vehicles", tags=["Vehicle State Management"])
def list_vehicles() -> List[Dict[str, Any]]:
    """Liệt kê các xe có context đang hoạt động trong tiến trình service."""
    return state_manager.list_active_vehicles()


@app.post("/api/v1/vehicles/{vehicle_id}/reset-state", tags=["Vehicle State Management"])
def reset_vehicle_state(vehicle_id: str) -> Dict[str, Any]:
    """Reset trạng thái bộ lọc và xóa sạch buffer cũ của một xe."""
    queue_manager.clear_vehicle(vehicle_id)
    return {
        "status": "success",
        "message": f"Đã reset trạng thái và xóa sạch buffer thành công cho xe {vehicle_id}",
        "vehicle_id": vehicle_id,
        "timestamp": datetime.now().isoformat(),
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.service.api:app", host="0.0.0.0", port=8000, reload=True)
