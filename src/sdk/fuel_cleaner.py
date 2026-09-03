"""
Vicomsat Fuel Cleaner SDK (Python Library Engine)
Đóng gói toàn diện AI Random Forest Causal + Adaptive Kalman thành thư viện độc lập.
Cho phép nhà phát triển Python gọi trực tiếp bằng 3 dòng code.
"""
from __future__ import annotations

import os
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Union
import pandas as pd

from src.service.state_manager import StreamingStateManager

# Từ điển ánh xạ trạng thái sang tiếng Việt
AI_STATE_DESCRIPTIONS = {
    "UPWARD_SHIFT": "Bơm/Đổ nhiên liệu",
    "DOWNWARD_SHIFT": "Hụt dầu đột ngột",
    "GRADUAL_CHANGE": "Tiêu thụ khi chạy",
    "STABLE_JITTER": "Xe đỗ ổn định",
    "OSCILLATION_NOISE": "Nhiễu sóng sánh/xung",
}


class FuelCleanerEngine:
    """
    Engine xử lý và lọc dữ liệu nhiên liệu theo thời gian thực (Causal AI-Kalman).
    Quản lý bộ nhớ trạng thái cách ly cho từng xe, an toàn luồng (thread-safe).
    """

    def __init__(self, model_dir: Optional[str] = None):
        """
        Khởi tạo Engine lọc dầu.
        Tự động tìm và nạp mô hình Random Forest tại models/fuel_state_classifier nếu không truyền đường dẫn.
        """
        if model_dir is None:
            # Tìm đường dẫn model tương đối so với thư mục gốc dự án
            base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
            candidate = os.path.join(base_dir, "models", "fuel_state_classifier")
            if os.path.exists(candidate):
                model_dir = candidate
            else:
                model_dir = "models/fuel_state_classifier"

        self.state_manager = StreamingStateManager(model_dir=model_dir)

    def clean_point(
        self,
        vehicle_id: str,
        timestamp: Union[str, datetime],
        raw_fuel: float,
        speed: float = 0.0,
        distance_m: float = 0.0,
        lat: Optional[float] = None,
        lng: Optional[float] = None,
        capacity_est: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Xử lý và lọc sạch 1 điểm đo đạc theo thời gian thực (Streaming Causal).

        Tham số:
            vehicle_id: Biển số hoặc mã xe (ví dụ: '29H-75028').
            timestamp: Thời gian đo ('2026-08-15 09:16:00' hoặc datetime).
            raw_fuel: Mức nhiên liệu thô từ cảm biến phao xăng (Lít).
            speed: Vận tốc di chuyển của xe (km/h).
            distance_m: Quãng đường GPS tích lũy (mét).
            lat, lng: Tọa độ GPS (tùy chọn).
            capacity_est: Dung tích bình chứa ước tính (Lít, mặc định 850L nếu không truyền).

        Trả về:
            Dictionary chứa mức nhiên liệu sạch (clean_fuel), trạng thái AI song ngữ,
            cờ cảnh báo đổ xăng/rút dầu và thời gian xử lý.
        """
        t_start = time.perf_counter()

        # Chuẩn hóa timestamp sang kiểu datetime
        if isinstance(timestamp, str):
            dt = None
            for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S.%f"):
                try:
                    dt = datetime.strptime(timestamp.split("+")[0].split("Z")[0], fmt)
                    break
                except Exception:
                    pass
            if dt is None:
                dt = datetime.utcnow()
        else:
            dt = timestamp

        # Lấy giá trị sạch trước đó của xe để phát hiện bước nhảy
        ctx = self.state_manager.get_or_create_context(vehicle_id, capacity_est=capacity_est)
        prev_clean = ctx.last_clean_fuel if ctx.last_clean_fuel is not None else float(raw_fuel)

        # Chạy pipeline AI + Adaptive Kalman qua StreamingStateManager
        res = self.state_manager.process_point(
            vehicle_id=vehicle_id,
            fuel_time=dt,
            fuel_level=float(raw_fuel),
            speed=float(speed) if speed is not None else 0.0,
            distance_meters=float(distance_m) if distance_m is not None else 0.0,
            lat=lat,
            lng=lng,
            capacity_est=capacity_est,
        )

        clean_val = float(res.get("clean_fuel_liters", raw_fuel))
        ai_state = str(res.get("ai_signal_state", "STABLE_JITTER"))
        conf = float(res.get("confidence", 1.0))
        ai_state_desc = AI_STATE_DESCRIPTIONS.get(ai_state, "Ổn định")

        # Xác định các cờ biến cố
        is_refuel = (ai_state == "UPWARD_SHIFT") or ((clean_val - prev_clean) >= 5.0)
        is_drain = (ai_state == "DOWNWARD_SHIFT") and ((prev_clean - clean_val) >= 8.0)
        is_spike = (ai_state == "OSCILLATION_NOISE") and (abs(float(raw_fuel) - clean_val) >= 2.5)

        # Nhãn tiếng Việt tóm tắt biến cố
        if is_refuel:
            event_label = "Đổ xăng"
        elif is_drain:
            event_label = "Nghi rút trộm dầu"
        elif is_spike:
            event_label = "Nhiễu xung cảm biến"
        elif speed > 1.0:
            event_label = "Tiêu thụ khi chạy"
        else:
            event_label = "Xe đỗ ổn định"

        elapsed_ms = (time.perf_counter() - t_start) * 1000.0

        return {
            "vehicle_id": vehicle_id,
            "timestamp": dt.strftime("%Y-%m-%d %H:%M:%S"),
            "raw_fuel": round(float(raw_fuel), 2),
            "clean_fuel": round(clean_val, 2),
            "speed": round(float(speed), 1) if speed is not None else 0.0,
            "ai_state": ai_state,
            "ai_state_desc": ai_state_desc,
            "event_label": event_label,
            "confidence": round(conf, 4),
            "is_refuel": is_refuel,
            "is_drain": is_drain,
            "is_spike": is_spike,
            "processing_time_ms": round(elapsed_ms, 3),
        }

    def clean_batch(
        self,
        df: pd.DataFrame,
        vehicle_id_col: str = "vehicle_id",
        time_col: str = "timestamp",
        fuel_col: str = "raw_fuel",
        speed_col: str = "speed",
        distance_col: Optional[str] = None,
        capacity_est: Optional[float] = None,
    ) -> pd.DataFrame:
        """
        Xử lý hàng loạt toàn bộ DataFrame (thích hợp cho các file CSV, Excel phân tích dữ liệu lịch sử).
        """
        out_df = df.copy()

        # Tự động tìm tên cột nếu người dùng đặt khác
        col_map = {c.lower(): c for c in out_df.columns}
        v_col = out_df.columns[0]
        for candidate in (vehicle_id_col, "vehicle_id", "car_id", "bien_so", "vehicleid"):
            if candidate.lower() in col_map:
                v_col = col_map[candidate.lower()]
                break

        t_col = None
        for candidate in (time_col, "timestamp", "fueltime", "time", "thoi_gian"):
            if candidate.lower() in col_map:
                t_col = col_map[candidate.lower()]
                break

        f_col = None
        for candidate in (fuel_col, "raw_fuel", "fuellevel", "fuel", "raw", "xang_tho"):
            if candidate.lower() in col_map:
                f_col = col_map[candidate.lower()]
                break

        s_col = None
        for candidate in (speed_col, "speed", "van_toc"):
            if candidate.lower() in col_map:
                s_col = col_map[candidate.lower()]
                break

        d_col = None
        if distance_col:
            for candidate in (distance_col, "distancemeters", "distance"):
                if candidate.lower() in col_map:
                    d_col = col_map[candidate.lower()]
                    break

        clean_fuels = []
        ai_states = []
        ai_state_descs = []
        event_labels = []
        is_refuels = []
        is_drains = []

        for _, row in out_df.iterrows():
            vid = str(row[v_col]) if v_col in row else "DEFAULT_VEHICLE"
            tm = row[t_col] if t_col in row else datetime.utcnow()
            fl = float(row[f_col]) if f_col in row and pd.notnull(row[f_col]) else 0.0
            sp = float(row[s_col]) if s_col in row and pd.notnull(row[s_col]) else 0.0
            dist = float(row[d_col]) if d_col and d_col in row and pd.notnull(row[d_col]) else 0.0

            res = self.clean_point(
                vehicle_id=vid,
                timestamp=tm,
                raw_fuel=fl,
                speed=sp,
                distance_m=dist,
                capacity_est=capacity_est,
            )

            clean_fuels.append(res["clean_fuel"])
            ai_states.append(res["ai_state"])
            ai_state_descs.append(res["ai_state_desc"])
            event_labels.append(res["event_label"])
            is_refuels.append(res["is_refuel"])
            is_drains.append(res["is_drain"])

        out_df["clean_fuel"] = clean_fuels
        out_df["ai_state"] = ai_states
        out_df["ai_state_desc"] = ai_state_descs
        out_df["event_label"] = event_labels
        out_df["is_refuel"] = is_refuels
        out_df["is_drain"] = is_drains

        return out_df

    def reset_vehicle(self, vehicle_id: str) -> bool:
        """Xóa trạng thái trong RAM của một xe cụ thể."""
        with self.state_manager._lock:
            if vehicle_id in self.state_manager._contexts:
                del self.state_manager._contexts[vehicle_id]
                return True
        return False

    def get_active_vehicles(self) -> List[str]:
        """Lấy danh sách các xe đang có trạng thái lưu trong RAM."""
        with self.state_manager._lock:
            return list(self.state_manager._contexts.keys())
