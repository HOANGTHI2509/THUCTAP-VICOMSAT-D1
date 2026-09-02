from __future__ import annotations

import collections
import logging
import queue
import threading
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

from src.service.state_manager import StreamingStateManager

logger = logging.getLogger(__name__)


class VehicleQueueManager:
    """Quản lý hàng đợi (Queue) phân luồng theo từng biển số xe (vehicle_id).
    
    Đặc điểm:
    - Mỗi xe sở hữu 1 hàng đợi FIFO riêng (queue.Queue).
    - Mỗi xe có 1 luồng worker độc lập để xử lý theo đúng thứ tự thời gian.
    - Non-blocking ingestion: API nhận telemetry và trả về ngay (< 1ms).
    - Tách biệt hoàn toàn trạng thái Kalman và AI giữa các xe.
    """

    def __init__(self, state_manager: StreamingStateManager, max_buffer_per_vehicle: int = 500):
        self.state_manager = state_manager
        self.max_buffer_per_vehicle = max_buffer_per_vehicle
        self._lock = threading.Lock()
        self._queues: Dict[str, queue.Queue] = {}
        self._workers: Dict[str, threading.Thread] = {}
        self._buffers: Dict[str, collections.deque] = collections.defaultdict(
            lambda: collections.deque(maxlen=self.max_buffer_per_vehicle)
        )
        self._stats: Dict[str, Dict[str, Any]] = {}
        self._running = True
        self.global_live_buffer = collections.deque(maxlen=500)
        self.active_vehicle_id: Optional[str] = None

    def _ensure_vehicle_worker(self, vehicle_id: str) -> queue.Queue:
        with self._lock:
            if vehicle_id not in self._queues:
                q = queue.Queue()
                self._queues[vehicle_id] = q
                self._stats[vehicle_id] = {
                    "vehicle_id": vehicle_id,
                    "total_enqueued": 0,
                    "total_processed": 0,
                    "queue_size": 0,
                    "last_enqueued_at": None,
                    "last_processed_at": None,
                    "current_fuel_raw": 0.0,
                    "current_fuel_clean": 0.0,
                    "current_speed": 0.0,
                    "current_state": "INIT",
                    "current_lat": 21.0285,
                    "current_lng": 105.8542,
                    "current_address": "",
                }

                worker = threading.Thread(
                    target=self._vehicle_worker_loop,
                    args=(vehicle_id,),
                    name=f"Worker-{vehicle_id}",
                    daemon=True,
                )
                self._workers[vehicle_id] = worker
                worker.start()
                logger.info(f"Khởi động worker thread cho xe {vehicle_id}")

            if self.active_vehicle_id is None:
                self.active_vehicle_id = vehicle_id

            return self._queues[vehicle_id]

    def enqueue(self, vehicle_id: str, point_data: Dict[str, Any]) -> Dict[str, Any]:
        """Đưa điểm dữ liệu telemetry vào hàng chờ của xe."""
        q = self._ensure_vehicle_worker(vehicle_id)
        q.put(point_data)

        with self._lock:
            stats = self._stats.get(vehicle_id)
            if stats:
                stats["total_enqueued"] += 1
                stats["queue_size"] = q.qsize()
                stats["last_enqueued_at"] = datetime.now().isoformat()

        return {
            "status": "queued",
            "vehicle_id": vehicle_id,
            "queue_size": q.qsize(),
            "enqueued_at": datetime.now().isoformat(),
        }

    def _vehicle_worker_loop(self, vehicle_id: str):
        """Worker thread vòng lặp xử lý từng gói dữ liệu trong hàng chờ của xe."""
        q = self._queues[vehicle_id]

        while self._running:
            try:
                point_data = q.get(timeout=0.5)
            except queue.Empty:
                continue

            try:
                self._process_single_point(vehicle_id, point_data)
            except Exception as e:
                logger.error(f"Lỗi khi xử lý điểm cho xe {vehicle_id}: {e}", exc_info=True)
            finally:
                q.task_done()
                with self._lock:
                    if vehicle_id in self._stats:
                        self._stats[vehicle_id]["queue_size"] = q.qsize()

    def _process_single_point(self, vehicle_id: str, data: Dict[str, Any]):
        raw_fuel = float(data.get("raw", data.get("raw_fuel", data.get("FuelLevel", 0.0))))
        speed = float(data.get("speed", data.get("Speed", 0.0)))
        lat = float(data.get("lat", data.get("Lat", 0.0)))
        lng = float(data.get("lng", data.get("Lng", 0.0)))
        time_str = str(data.get("time", data.get("FuelTime", datetime.now().isoformat())))
        address = str(data.get("address", data.get("Address", "")))
        capacity_est = float(data.get("capacity_est", data.get("capacity", 200.0)))

        # Sửa đảo ngược kinh độ/vĩ độ nếu có
        if lat > 50 and lng < 50:
            lat, lng = lng, lat

        try:
            f_time = datetime.fromisoformat(time_str)
        except Exception:
            f_time = datetime.now()

        # Xử lý qua Causal AI + Adaptive Kalman trong state_manager
        res = self.state_manager.process_point(
            vehicle_id=vehicle_id,
            fuel_time=f_time,
            fuel_level=raw_fuel,
            speed=speed,
            lat=lat,
            lng=lng,
            capacity_est=capacity_est,
        )
        clean_val = res["clean_fuel_liters"]
        ai_state = res["ai_signal_state"]

        point_payload = {
            "time": time_str,
            "speed": speed,
            "raw": raw_fuel,
            "kalman": data.get("kalman", clean_val),
            "adaptive": data.get("adaptive", clean_val),
            "ai_enhanced": clean_val,
            "lat": lat,
            "lng": lng,
            "address": address,
            "vehicle_id": vehicle_id,
            "state": ai_state,
        }

        # Lưu vào buffer riêng của xe và buffer tổng
        with self._lock:
            self._buffers[vehicle_id].append(point_payload)
            self.global_live_buffer.append(point_payload)

            stats = self._stats.get(vehicle_id)
            if stats:
                stats["total_processed"] += 1
                stats["current_fuel_raw"] = round(raw_fuel, 2)
                stats["current_fuel_clean"] = round(clean_val, 2)
                stats["current_speed"] = round(speed, 1)
                stats["current_state"] = ai_state
                stats["current_lat"] = lat
                stats["current_lng"] = lng
                stats["current_address"] = address
                stats["last_processed_at"] = datetime.now().isoformat()

    def get_vehicle_data(self, vehicle_id: Optional[str] = None, limit: int = 300) -> List[Dict[str, Any]]:
        """Lấy danh sách điểm dữ liệu realtime của xe chỉ định."""
        target_vid = vehicle_id or self.active_vehicle_id
        with self._lock:
            if target_vid and target_vid in self._buffers and len(self._buffers[target_vid]) > 0:
                return list(self._buffers[target_vid])[-limit:]
            if self.global_live_buffer:
                items = list(self.global_live_buffer)
                if target_vid:
                    filtered = [p for p in items if p.get("vehicle_id") == target_vid]
                    if filtered:
                        return filtered[-limit:]
                return items[-limit:]
            return []

    def get_fleet_status(self) -> List[Dict[str, Any]]:
        """Trả về tình trạng hàng đợi và telemetry của toàn bộ các xe đang streaming."""
        with self._lock:
            return list(self._stats.values())

    def switch_active_vehicle(self, vehicle_id: str) -> bool:
        with self._lock:
            self.active_vehicle_id = vehicle_id
            return True

    def stop(self):
        self._running = False
