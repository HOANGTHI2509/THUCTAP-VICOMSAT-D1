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
from src.core.filters.ai_state_filter import load_fuel_state_classifier, filter_with_ai_state
from src.db.database import get_db_manager
from src.sdk.fuel_cleaner import FuelCleanerEngine

logger = logging.getLogger("FuelAPI")

# 1. Khởi tạo FastAPI App & Cấu hình bảo mật
API_KEY = os.getenv("API_KEY", "vicomsat_secret_key_2026")
REQUIRE_API_KEY = os.getenv("REQUIRE_API_KEY", "false").lower() in ("true", "1", "yes")
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./fuel_records.db")

app = FastAPI(
    title="VICOMSAT Real-time Fuel Denoising & Filtering API",
    description="Microservice thời gian thực (Causal AI + Adaptive Kalman) khử nhiễu dữ liệu cảm biến nhiên liệu cho Vcomsat.",
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
    Middleware xác thực API Key linh hoạt cho môi trường Doanh nghiệp.
    Mặc định REQUIRE_API_KEY=false để môi trường Local Demo không bị chặn.
    """
    path = request.url.path
    if REQUIRE_API_KEY and (path.startswith("/api/v1/clean") or path.startswith("/api/v1/events")):
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
ai_state_model, ai_state_metadata = load_fuel_state_classifier("models/fuel_state_classifier")
sdk_engine = FuelCleanerEngine()
db_manager = get_db_manager(DATABASE_URL)


# 3. Pydantic Schemas Doanh Nghiệp (Hỗ trợ Bí danh Tiếng Anh / Tiếng Việt)
class EnterprisePointInput(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    vehicle_id: str = Field(..., validation_alias=AliasChoices("vehicle_id", "bien_so", "car_id", "VehicleID"), example="29H-75028")
    timestamp: str = Field(..., validation_alias=AliasChoices("timestamp", "time", "thoi_gian", "FuelTime"), example="2026-08-15 09:16:00")
    raw_fuel: float = Field(..., validation_alias=AliasChoices("raw_fuel", "fuel", "xang_tho", "FuelLevel", "raw"), example=93.3)
    speed: float = Field(0.0, validation_alias=AliasChoices("speed", "van_toc", "Speed"), example=61.0)
    distance_m: float = Field(0.0, validation_alias=AliasChoices("distance_m", "quang_duong", "DistanceMeters"), example=0.0)
    lat: Optional[float] = Field(None, validation_alias=AliasChoices("lat", "vi_do", "Lat"), example=21.0285)
    lng: Optional[float] = Field(None, validation_alias=AliasChoices("lng", "kinh_do", "Lng"), example=105.8542)
    address: Optional[str] = Field(None, validation_alias=AliasChoices("address", "dia_chi", "Address"))
    capacity_est: Optional[float] = Field(None, validation_alias=AliasChoices("capacity_est", "dung_tich", "capacity"), example=95.0)


class EnterpriseCleanResponse(BaseModel):
    vehicle_id: str
    timestamp: str
    raw_fuel: float
    clean_fuel: float
    speed: float
    ai_state: str
    ai_state_desc: str
    event_label: str
    confidence: float
    is_refuel: bool
    is_drain: bool
    is_spike: bool
    processing_time_ms: float
    saved_to_db: bool


class EnterpriseBatchInput(BaseModel):
    points: List[EnterprisePointInput]
class FuelPointInput(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    vehicle_id: str = Field(..., alias="VehicleID", example="29E-45520", description="Mã định danh hoặc biển số xe")
    fuel_time: datetime = Field(..., alias="FuelTime", example="2026-08-27T10:00:00", description="Thời điểm ghi nhận telemetry")
    fuel_level: float = Field(..., alias="FuelLevel", example=105.2, description="Mức nhiên liệu đo thô từ cảm biến (Lít)")
    speed: float = Field(0.0, alias="Speed", example=45.0, description="Vận tốc GPS của xe (km/h)")
    lat: Optional[float] = Field(None, alias="Lat", example=21.0285, description="Vĩ độ GPS")
    lng: Optional[float] = Field(None, alias="Lng", example=105.8542, description="Kinh độ GPS")
    address: Optional[str] = Field(None, alias="Address", description="Địa chỉ GPS (nếu có)")
    
    # Các trường mở rộng (Không bắt buộc Vcomsat phải gửi, tự fallback mặc định)
    distance_meters: float = Field(0.0, example=350.0, description="Quãng đường (không bắt buộc)")
    segment_id: Optional[str] = Field(None, example="SEG_01")
    capacity_est: Optional[float] = Field(200.0, example=200.0)
    noise_sigma_liters: Optional[float] = Field(0.8, example=0.8)


class CleanFuelOutput(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    vehicle_id: str = Field(..., alias="VehicleID")
    fuel_time: str = Field(..., alias="FuelTime")
    raw_fuel_liters: float = Field(..., alias="RawFuel")
    clean_fuel_liters: float = Field(..., alias="CleanFuel")
    ai_signal_state: str = Field(..., alias="AI_State")
    confidence: float = Field(..., alias="Confidence")
    quality_flag: str = Field(..., alias="QualityFlag")
    latency_ms: float = Field(..., alias="LatencyMs")


class FuelBatchInput(BaseModel):
    vehicle_id: str = Field(..., example="29E-45520")
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
        "model_type": "RandomForest_Causal_v3",
        "active_vehicles_in_memory": len(state_manager._contexts),
        "database_enabled": db_manager.enabled,
        "api_key_required": REQUIRE_API_KEY,
        "timestamp": datetime.now().isoformat(),
    }


@app.post("/api/v1/clean", response_model=EnterpriseCleanResponse, tags=["Enterprise API"])
def enterprise_clean_point(point: EnterprisePointInput) -> EnterpriseCleanResponse:
    """
    Endpoint chuẩn Doanh nghiệp: Nhận 1 điểm đo thô (hỗ trợ bí danh tiếng Anh/Việt)
    -> Lọc AI-Kalman thời gian thực -> Tự lưu vào CSDL 3NF (nếu bật DB)
    -> Trả ngay JSON kết quả song ngữ (< 5ms).
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

    saved_to_db = False
    if db_manager.enabled:
        log_id = db_manager.save_measurement(
            vehicle_id=point.vehicle_id,
            timestamp=point.timestamp,
            raw_fuel=point.raw_fuel,
            clean_fuel=res["clean_fuel"],
            speed=point.speed,
            lat=point.lat,
            lng=point.lng,
            state_code=res["ai_state"],
            capacity_est=point.capacity_est,
        )
        if log_id:
            saved_to_db = True

        # Tự động lưu biến cố vào bảng fuel_events
        if res["is_refuel"]:
            db_manager.save_event(
                vehicle_id=point.vehicle_id,
                event_type="REFUEL",
                start_time=point.timestamp,
                end_time=point.timestamp,
                start_fuel=point.raw_fuel,
                end_fuel=res["clean_fuel"],
                change_liters=round(res["clean_fuel"] - point.raw_fuel, 2),
                lat=point.lat,
                lng=point.lng,
                address=point.address,
                capacity_est=point.capacity_est,
            )
        elif res["is_drain"]:
            db_manager.save_event(
                vehicle_id=point.vehicle_id,
                event_type="DRAIN",
                start_time=point.timestamp,
                end_time=point.timestamp,
                start_fuel=point.raw_fuel,
                end_fuel=res["clean_fuel"],
                change_liters=round(point.raw_fuel - res["clean_fuel"], 2),
                lat=point.lat,
                lng=point.lng,
                address=point.address,
                capacity_est=point.capacity_est,
            )

    return EnterpriseCleanResponse(**res, saved_to_db=saved_to_db)


@app.post("/api/v1/clean-batch", response_model=List[EnterpriseCleanResponse], tags=["Enterprise API"])
def enterprise_clean_batch(batch: EnterpriseBatchInput) -> List[EnterpriseCleanResponse]:
    """
    Endpoint chuẩn Doanh nghiệp: Xử lý hàng loạt theo mảng (10-500 điểm)
    cho các thiết bị truyền dữ liệu theo cụm.
    """
    results = []
    for pt in batch.points:
        r = enterprise_clean_point(pt)
        results.append(r)
    return results


@app.get("/api/v1/events", tags=["Enterprise API"])
def get_vehicle_events(vehicle_id: str, limit: int = 100) -> List[Dict[str, Any]]:
    """Tra cứu các sự kiện đổ xăng hoặc nghi ngờ rút trộm dầu từ bảng CSDL 3NF."""
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
    if not cur_data:
        history_pts = get_history(vehicle_id=vehicle_id)
        for pt in history_pts[:150]:
            queue_manager.enqueue(vehicle_id, pt)
    return {"status": "switched", "vehicle_id": vehicle_id, "total_points": len(cur_data)}


BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

DATASET_FILE_MAP = {
    "21H-03221": os.path.join(BASE_DIR, "TienXuLy", "21H-03221_processed.csv"),
    "24H-04650": os.path.join(BASE_DIR, "TienXuLy", "24H-04650_processed.csv"),
    "29E-45520": os.path.join(BASE_DIR, "TienXuLy", "29E-45520_processed.csv"),
    "90H-03494": os.path.join(BASE_DIR, "TienXuLy", "90H-03494_processed.csv"),
}


VEHICLE_INFO_CACHE: List[Dict[str, Any]] = []


@app.get("/api/vehicles", tags=["Web App Live Demo"])
def list_available_vehicles() -> List[Dict[str, Any]]:
    """Trả về danh sách tất cả các xe kèm ngày bắt đầu, ngày kết thúc và tổng số điểm đo."""
    global VEHICLE_INFO_CACHE
    if VEHICLE_INFO_CACHE:
        return VEHICLE_INFO_CACHE

    import glob
    import pandas as pd
    files = glob.glob(os.path.join(BASE_DIR, "TienXuLy", "*_processed.csv"))
    vehicles = []
    for f in sorted(files):
        vid = os.path.basename(f).replace("_processed.csv", "")
        try:
            df_time = pd.read_csv(f, usecols=["FuelTime"])
            s_date = str(df_time["FuelTime"].min())[:10]
            e_date = str(df_time["FuelTime"].max())[:10]
            t_pts = len(df_time)
        except Exception:
            s_date = "2026-08-10"
            e_date = "2026-08-18"
            t_pts = 0

        vehicles.append({
            "id": vid,
            "name": f"Xe {vid}",
            "start_date": s_date,
            "end_date": e_date,
            "total_points": t_pts,
        })
    VEHICLE_INFO_CACHE = vehicles
    return vehicles


@app.get("/api/current-vehicle", tags=["Web App Live Demo"])
def get_current_vehicle() -> Dict[str, Any]:
    """Trả về biển số xe đang nhận luồng dữ liệu telemetry thời gian thực."""
    active_vid = LIVE_STREAM_BUFFER[-1].get("vehicle_id") if LIVE_STREAM_BUFFER else "21H-03221"
    return {"vehicle_id": active_vid, "total_points": len(LIVE_STREAM_BUFFER)}


HISTORY_CACHE = {}


@app.get("/api/history", tags=["Web App Live Demo"])
def get_history(vehicle_id: str = "21H-03221", start_date: str = "", end_date: str = "", limit: int = 10000) -> List[Dict[str, Any]]:
    """Tra cứu lịch sử dữ liệu lọc theo xe và khoảng ngày được chọn."""
    import glob
    # Tự động chuẩn hóa nếu người dùng chọn ngày bắt đầu lớn hơn ngày kết thúc
    if start_date and end_date and start_date > end_date:
        start_date, end_date = end_date, start_date

    cache_key = f"{vehicle_id}_{start_date}_{end_date}_{limit}"
    if cache_key in HISTORY_CACHE:
        return HISTORY_CACHE[cache_key]

    # 1. Tìm file dữ liệu của xe
    target_file = None
    for vid, path in DATASET_FILE_MAP.items():
        if vid in vehicle_id:
            target_file = path
            break

    # Nếu không thuộc các xe ghim sẵn, tìm trong toàn bộ thư mục TienXuLy
    if not target_file:
        cand = os.path.join(BASE_DIR, "TienXuLy", f"{vehicle_id}_processed.csv")
        if os.path.exists(cand):
            target_file = cand
        else:
            for f in glob.glob(os.path.join(BASE_DIR, "TienXuLy", "*_processed.csv")):
                if vehicle_id in f:
                    target_file = f
                    break

    if not target_file or not os.path.exists(target_file):
        return list(LIVE_STREAM_BUFFER)

    try:
        import numpy as np
        import pandas as pd
        from src.core.filters.ai_enhanced_adaptive_realtime import filter_ai_enhanced_adaptive_realtime

        if target_file.endswith(".xlsx"):
            df = pd.read_excel(target_file)
        else:
            df = pd.read_csv(target_file)

        time_col = "FuelTime" if "FuelTime" in df.columns else "time"
        fuel_col = "FuelLevel" if "FuelLevel" in df.columns else ("raw" if "raw" in df.columns else "fuel_level")
        speed_col = "Speed" if "Speed" in df.columns else "speed"

        df["_dt"] = pd.to_datetime(df[time_col], errors="coerce")
        df_filtered = df.dropna(subset=["_dt"]).copy()

        # Lọc theo khoảng ngày người dùng chọn
        if start_date:
            try:
                st = pd.to_datetime(start_date)
                df_filtered = df_filtered[df_filtered["_dt"] >= st]
            except Exception:
                pass

        if end_date:
            try:
                et = pd.to_datetime(end_date) + pd.Timedelta(days=1)
                df_filtered = df_filtered[df_filtered["_dt"] < et]
            except Exception:
                pass

        if len(df_filtered) == 0:
            df_sample = df.head(limit).copy()
        else:
            df_sample = df_filtered.head(limit).copy()

        df_sample["FuelLevel"] = pd.to_numeric(df_sample[fuel_col].astype(str).str.replace(",", "."), errors="coerce").fillna(100.0)
        df_sample["Speed"] = pd.to_numeric(df_sample[speed_col].astype(str).str.replace(",", "."), errors="coerce").fillna(0.0)
        df_sample["FuelTime"] = df_sample[time_col].astype(str)

        # 1. Trích đặc trưng & Phân loại trạng thái AI (Random Forest)
        if ai_state_model is not None and ai_state_metadata is not None:
            df_sample = filter_with_ai_state(
                df_sample,
                model=ai_state_model,
                metadata=ai_state_metadata,
                mode="realtime",
            )

        # 2. Chạy lọc Adaptive Kalman theo nhãn AI
        clean_fuels = filter_ai_enhanced_adaptive_realtime(df_sample, config={"source_col": "FuelLevel"})

        results = []
        for i, (_, row) in enumerate(df_sample.iterrows()):
            time_val = str(row.get(time_col, ""))
            raw_val = float(row.get("FuelLevel", 0.0))
            clean_val = float(clean_fuels[i]) if i < len(clean_fuels) and not np.isnan(clean_fuels[i]) else raw_val
            speed_val = float(row.get("Speed", 0.0))
            ai_st = str(row.get("AI_State", "NORMAL"))
            
            lat_raw = str(row.get("Lat", 21.0285)).replace(",", ".").strip()
            lng_raw = str(row.get("Lng", 105.8542)).replace(",", ".").strip()
            try:
                lat_val = float(lat_raw)
            except Exception:
                lat_val = 21.0285
            try:
                lng_val = float(lng_raw)
            except Exception:
                lng_val = 105.8542
            
            if lat_val > 50 and lng_val < 50:
                lat_val, lng_val = lng_val, lat_val

            results.append({
                "time": time_val,
                "timestamp": time_val,
                "raw": raw_val,
                "raw_fuel": raw_val,
                "ai_enhanced": round(clean_val, 2),
                "clean": round(clean_val, 2),
                "kalman": round(clean_val, 2),
                "speed": speed_val,
                "lat": lat_val,
                "lng": lng_val,
                "address": str(row.get("Address", "")),
                "vehicle_id": vehicle_id,
                "state": ai_st,
            })

        HISTORY_CACHE[cache_key] = results
        return results
    except Exception as e:
        return list(LIVE_STREAM_BUFFER)


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
    """Liệt kê danh sách các xe và trạng thái bộ lọc đang được duy trì trong bộ nhớ In-Memory."""
    return state_manager.list_active_vehicles()


@app.post("/api/v1/vehicles/{vehicle_id}/reset-state", tags=["Vehicle State Management"])
def reset_vehicle_state(vehicle_id: str) -> Dict[str, Any]:
    """Reset trạng thái bộ lọc của một xe (khi xe bảo dưỡng hoặc thay đổi cảm biến)."""
    success = state_manager.reset_vehicle_state(vehicle_id)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Xe '{vehicle_id}' chưa có trạng thái trong bộ nhớ.",
        )
    return {
        "status": "success",
        "message": f"Đã reset trạng thái bộ lọc thành công cho xe {vehicle_id}",
        "vehicle_id": vehicle_id,
        "timestamp": datetime.now().isoformat(),
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.service.api:app", host="0.0.0.0", port=8000, reload=True)
