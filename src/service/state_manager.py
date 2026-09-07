from __future__ import annotations

import json
import os
import pickle
import math
import statistics
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.core.filters.ai_enhanced_adaptive_realtime import (
    RealtimeAdaptiveKalmanState,
    filter_ai_enhanced_adaptive_realtime,
)



def load_fuel_state_classifier(model_dir: str = "models/fuel_state_classifier"):
    if not os.path.isabs(model_dir):
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        candidate = os.path.join(base_dir, model_dir)
        if os.path.exists(candidate):
            model_dir = candidate
    model_path = os.path.join(model_dir, "fuel_state_classifier.pkl")
    metadata_path = os.path.join(model_dir, "metadata.json")
    if not os.path.exists(model_path) or not os.path.exists(metadata_path):
        return None, None
    try:
        with open(metadata_path, encoding="utf-8") as handle:
            metadata = json.load(handle)
        with open(model_path, "rb") as handle:
            model = pickle.load(handle)
        return model, metadata
    except Exception:
        return None, None


@dataclass
class VehicleStreamContext:
    vehicle_id: str
    capacity_est: float = 200.0
    noise_sigma_liters: float = 0.8
    flat_jitter_threshold: float = 0.8
    spike_threshold: float = 2.5
    event_threshold: float = 5.0
    current_segment_id: Optional[str] = None
    kalman_state: RealtimeAdaptiveKalmanState = field(default_factory=RealtimeAdaptiveKalmanState)
    history_fuel: deque = field(default_factory=lambda: deque(maxlen=12))
    history_time: deque = field(default_factory=lambda: deque(maxlen=12))
    history_speed: deque = field(default_factory=lambda: deque(maxlen=12))
    last_seen_time: Optional[datetime] = None
    last_clean_fuel: Optional[float] = None
    last_ai_state: str = "UNKNOWN"
    last_confidence: float = 1.0
    total_points_processed: int = 0


class StreamingStateManager:
    """Thread-safe In-Memory State & Feature Extraction Manager per Vehicle."""

    def __init__(self, model_dir: Optional[str] = None):
        self._global_lock = threading.Lock()
        self._contexts: Dict[str, VehicleStreamContext] = {}
        self._vehicle_locks: Dict[str, threading.RLock] = {}
        
        # Load AI Model (RF Causal Classifier)
        if model_dir is None:
            model_dir = str(Path("models/fuel_state_classifier"))
        self.model, self.metadata = load_fuel_state_classifier(model_dir)
        self.feature_columns = self.metadata.get("feature_columns", []) if self.metadata else []
        self.labels = self.metadata.get("labels", []) if self.metadata else []

    @staticmethod
    def _compute_thresholds(capacity_est: float, noise_sigma: float) -> Tuple[float, float, float]:
        cap = capacity_est if capacity_est and capacity_est > 10.0 else 200.0
        sigma = noise_sigma if noise_sigma and noise_sigma > 0.05 else 0.8
        flat = max(2.5 * sigma, 0.005 * cap, 0.5)
        spike = max(4.5 * sigma, 0.012 * cap, flat * 1.8)
        event = max(8.0 * sigma, 0.035 * cap, spike * 1.8)
        return flat, spike, event

    def _get_vehicle_lock(self, vehicle_id: str) -> threading.RLock:
        with self._global_lock:
            if vehicle_id not in self._vehicle_locks:
                self._vehicle_locks[vehicle_id] = threading.RLock()
            return self._vehicle_locks[vehicle_id]

    def get_or_create_context(
        self,
        vehicle_id: str,
        capacity_est: Optional[float] = None,
        noise_sigma_liters: Optional[float] = None,
        segment_id: Optional[str] = None,
    ) -> VehicleStreamContext:
        v_lock = self._get_vehicle_lock(vehicle_id)
        with v_lock:
            with self._global_lock:
                if vehicle_id not in self._contexts:
                    cap = capacity_est if capacity_est and capacity_est > 10.0 else 200.0
                    sigma = noise_sigma_liters if noise_sigma_liters and noise_sigma_liters > 0.05 else 0.8
                    flat, spike, event = self._compute_thresholds(cap, sigma)
                    ctx = VehicleStreamContext(
                        vehicle_id=vehicle_id,
                        capacity_est=cap,
                        noise_sigma_liters=sigma,
                        flat_jitter_threshold=flat,
                        spike_threshold=spike,
                        event_threshold=event,
                        current_segment_id=segment_id,
                    )
                    self._contexts[vehicle_id] = ctx
                    return ctx
                else:
                    ctx = self._contexts[vehicle_id]
                    updated = False
                    if capacity_est and capacity_est > 10.0 and abs(capacity_est - ctx.capacity_est) > 1e-3:
                        ctx.capacity_est = capacity_est
                        updated = True
                    if noise_sigma_liters and noise_sigma_liters > 0.05 and abs(noise_sigma_liters - ctx.noise_sigma_liters) > 1e-4:
                        ctx.noise_sigma_liters = noise_sigma_liters
                        updated = True
                    if updated:
                        ctx.flat_jitter_threshold, ctx.spike_threshold, ctx.event_threshold = (
                            self._compute_thresholds(ctx.capacity_est, ctx.noise_sigma_liters)
                        )
                    if segment_id is not None and ctx.current_segment_id is None:
                        ctx.current_segment_id = segment_id
                    return ctx

    def reset_vehicle_state(self, vehicle_id: str) -> bool:
        v_lock = self._get_vehicle_lock(vehicle_id)
        with v_lock:
            with self._global_lock:
                if vehicle_id in self._contexts:
                    del self._contexts[vehicle_id]
                    return True
                return False

    def list_active_vehicles(self) -> List[Dict[str, Any]]:
        with self._global_lock:
            result = []
            for vid, ctx in self._contexts.items():
                result.append({
                    "vehicle_id": vid,
                    "capacity_est": ctx.capacity_est,
                    "noise_sigma": ctx.noise_sigma_liters,
                    "total_points": ctx.total_points_processed,
                    "last_seen": ctx.last_seen_time.isoformat() if ctx.last_seen_time else None,
                    "last_clean_fuel": ctx.last_clean_fuel,
                    "last_state": ctx.last_ai_state,
                    "current_segment_id": ctx.current_segment_id,
                })
            return result

    def process_point(
        self,
        vehicle_id: str,
        fuel_time: datetime,
        fuel_level: float,
        speed: float = 0.0,
        lat: Optional[float] = None,
        lng: Optional[float] = None,
        distance_meters: float = 0.0,
        segment_id: Optional[str] = None,
        capacity_est: Optional[float] = None,
        noise_sigma_liters: Optional[float] = None,
        trace_collector: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Process a single incoming telemetry data point through Causal AI + Kalman filter."""
        t_start = time.perf_counter()
        v_lock = self._get_vehicle_lock(vehicle_id)
        with v_lock:
            ctx = self.get_or_create_context(vehicle_id, capacity_est, noise_sigma_liters, segment_id)
            cap = ctx.capacity_est
            sigma = ctx.noise_sigma_liters
            flat = ctx.flat_jitter_threshold
            spike = ctx.spike_threshold
            event = ctx.event_threshold

            is_raw_nan = pd.isna(fuel_level)
            is_raw_zero = (not is_raw_nan) and (fuel_level <= 0.0)

            prev_time = ctx.history_time[-1] if ctx.history_time else None
            prev_fuel = ctx.history_fuel[-1] if ctx.history_fuel else None

            # 1. Out-of-order check (fuel_time < prev_time): khong cap nhat state realtime bang du lieu cu
            if prev_time is not None and fuel_time < prev_time:
                latency_ms = (time.perf_counter() - t_start) * 1000.0
                return {
                    "vehicle_id": vehicle_id,
                    "fuel_time": fuel_time.isoformat(),
                    "raw_fuel_liters": None if is_raw_nan else round(float(fuel_level), 2),
                    "clean_fuel_liters": round(float(ctx.last_clean_fuel), 2) if ctx.last_clean_fuel is not None else None,
                    "ai_signal_state": "OUT_OF_ORDER",
                    "confidence": 0.0,
                    "quality_flag": "OUT_OF_ORDER",
                    "latency_ms": round(latency_ms, 3),
                }

            # 2. Duplicate timestamp check (fuel_time == prev_time): tranh dem xac nhan 2 lan
            if prev_time is not None and fuel_time == prev_time:
                is_exact_dup = (
                    (is_raw_nan and pd.isna(prev_fuel)) or
                    (not is_raw_nan and prev_fuel is not None and not pd.isna(prev_fuel) and abs(fuel_level - prev_fuel) < 1e-4)
                )
                flag = "DUPLICATE_IGNORED" if is_exact_dup else "DUPLICATE_TIME_CONFLICT"
                latency_ms = (time.perf_counter() - t_start) * 1000.0
                return {
                    "vehicle_id": vehicle_id,
                    "fuel_time": fuel_time.isoformat(),
                    "raw_fuel_liters": None if is_raw_nan else round(float(fuel_level), 2),
                    "clean_fuel_liters": round(float(ctx.last_clean_fuel), 2) if ctx.last_clean_fuel is not None else None,
                    "ai_signal_state": "DUPLICATE",
                    "confidence": 1.0 if is_exact_dup else 0.0,
                    "quality_flag": flag,
                    "latency_ms": round(latency_ms, 3),
                }

            # 3. Check segment change or long gap (>120 minutes)
            time_gap_min = (fuel_time - prev_time).total_seconds() / 60.0 if prev_time is not None else 2.0
            is_segment_change = (
                segment_id is not None and
                ctx.current_segment_id is not None and
                segment_id != ctx.current_segment_id
            )
            is_long_gap = time_gap_min > 120.0

            if is_segment_change or is_long_gap:
                ctx.kalman_state = RealtimeAdaptiveKalmanState()
                ctx.history_fuel.clear()
                ctx.history_time.clear()
                ctx.history_speed.clear()
                ctx.last_clean_fuel = None
                ctx.last_ai_state = "UNKNOWN"
                ctx.last_confidence = 1.0
                if segment_id is not None:
                    ctx.current_segment_id = segment_id
                time_gap_min = 2.0
            elif segment_id is not None and ctx.current_segment_id is None:
                ctx.current_segment_id = segment_id

            # 4. Feature Extraction
            valid_history = [f for f in ctx.history_fuel if (f is not None and not pd.isna(f) and f > 5.0)]
            if valid_history:
                prev_valid_fuel = valid_history[-1]
            elif not is_raw_nan and fuel_level > 5.0:
                prev_valid_fuel = fuel_level
            else:
                prev_valid_fuel = cap

            if not is_raw_nan and fuel_level > 5.0:
                delta_fuel = fuel_level - prev_valid_fuel
            else:
                delta_fuel = 0.0
            abs_delta_fuel = abs(delta_fuel)

            fuel_val_safe = 0.0 if (is_raw_nan or is_raw_zero) else fuel_level
            fuel_pct = fuel_val_safe / cap if cap else 0.0
            delta_pct = delta_fuel / cap if cap else 0.0
            delta_over_noise = delta_fuel / sigma if sigma else 0.0

            # Update history deques
            ctx.history_fuel.append(fuel_level)
            ctx.history_time.append(fuel_time)
            ctx.history_speed.append(speed)

            valid_with_current = [f for f in ctx.history_fuel if (f is not None and not pd.isna(f) and f > 5.0)]
            if len(valid_with_current) >= 3:
                rolling_std_12 = float(statistics.pstdev(valid_with_current))
            else:
                rolling_std_12 = 0.0
            rolling_std_pct = rolling_std_12 / cap if cap else 0.0

            last3 = valid_with_current[-3:] if valid_with_current else ([fuel_level] if not is_raw_nan else [cap])
            prev_median3 = float(statistics.median(last3)) if last3 else (fuel_level if not is_raw_nan else cap)

            motion_speed = speed
            has_gps = 1.0 if (lat is not None and lng is not None) else 0.0
            gps_speed = speed

            valid_5 = valid_with_current[-5:] if len(valid_with_current) >= 5 else valid_with_current
            local_range5 = float(max(valid_5) - min(valid_5)) if valid_5 else 0.0
            valid_7 = valid_with_current[-7:] if len(valid_with_current) >= 7 else valid_with_current
            local_range7 = float(max(valid_7) - min(valid_7)) if valid_7 else local_range5

            # 5. AI Inference
            feature_dict = {
                "FuelLevel": fuel_val_safe,
                "FuelPct": fuel_pct,
                "Speed": speed,
                "MotionSpeedKmh": motion_speed,
                "TimeGapMinutes": time_gap_min,
                "DeltaFuel": delta_fuel,
                "DeltaPct": delta_pct,
                "AbsDeltaFuel": abs_delta_fuel,
                "DeltaOverNoise": delta_over_noise,
                "RollingStd12": rolling_std_12,
                "RollingStdPct": rolling_std_pct,
                "DistanceMeters": distance_meters,
                "GpsSpeedKmh": gps_speed,
                "HasGPS": has_gps,
                "capacity_est": cap,
                "noise_sigma_liters": sigma,
                "flat_jitter_threshold": flat,
                "spike_threshold": spike,
                "event_threshold": event,
                "PrevMedian3": prev_median3,
                "FutureMedian3": prev_median3,
                "FutureMedian5": prev_median3,
                "ReturnToPrevLevel": 0.0,
                "LocalRange5": local_range5,
                "LocalRange7": local_range7,
                "PeakReversalFlag": 0,
                "ValleyReversalFlag": 0,
                "TransientScore": 0.0,
            }

            if self.model is not None and self.feature_columns:
                feat_vector = np.array([[feature_dict.get(col, 0.0) for col in self.feature_columns]], dtype=float)
                if hasattr(self.model, "predict_proba"):
                    probs = self.model.predict_proba(feat_vector)[0]
                    idx = probs.argmax()
                    ai_signal_state = str(self.model.classes_[idx])
                    conf = float(probs[idx])
                else:
                    pred_raw = self.model.predict(feat_vector)[0]
                    ai_signal_state = str(pred_raw)
                    conf = 0.95
            else:
                if is_raw_nan or fuel_level <= 5.0:
                    ai_signal_state = "OSCILLATION_NOISE"
                elif speed > 1.0 and delta_fuel < -0.3 * flat:
                    ai_signal_state = "GRADUAL_CHANGE"
                else:
                    ai_signal_state = "STABLE_JITTER"
                conf = 0.90

            # 6. Realtime Adaptive Kalman Filtering
            pt_df = pd.DataFrame([{
                "FuelLevel": np.nan if is_raw_nan else fuel_level,
                "FuelTime": fuel_time,
                "Speed": speed,
                "DistanceMeters": distance_meters,
                "SegmentID": segment_id if segment_id is not None else ctx.current_segment_id,
                "capacity_est": cap,
                "noise_sigma_liters": sigma,
                "flat_jitter_threshold": flat,
                "spike_threshold": spike,
                "event_threshold": event,
                "RollingStd": rolling_std_12,
                "AI_State": ai_signal_state,
                "AI_SignalState": ai_signal_state,
                "AI_State_Confidence": conf,
                "QualityFlag": 1 if (is_raw_nan or is_raw_zero) else 0,
                "QualityReason": "FUEL_ZERO" if is_raw_zero else ("INVALID_RAW" if is_raw_nan else "VALID"),
            }])

            cfg_filter = {"source_col": "FuelLevel"}
            if trace_collector is not None:
                cfg_filter["trace_collector"] = trace_collector
            clean_fuels, next_state = filter_ai_enhanced_adaptive_realtime(
                pt_df,
                config=cfg_filter,
                state=ctx.kalman_state,
                return_state=True,
            )

            raw_clean = clean_fuels[0] if clean_fuels else np.nan
            if not pd.isna(raw_clean):
                clean_fuel_val = float(raw_clean)
            elif ctx.last_clean_fuel is not None and not pd.isna(ctx.last_clean_fuel):
                clean_fuel_val = float(ctx.last_clean_fuel)
            else:
                clean_fuel_val = None

            # Update context
            ctx.kalman_state = next_state
            ctx.last_seen_time = fuel_time
            if clean_fuel_val is not None:
                ctx.last_clean_fuel = clean_fuel_val
            ctx.last_ai_state = ai_signal_state
            ctx.last_confidence = conf
            ctx.total_points_processed += 1

            latency_ms = (time.perf_counter() - t_start) * 1000.0

            if clean_fuel_val is None:
                quality_flag = "NO_VALID_ESTIMATE"
            elif is_raw_zero:
                quality_flag = "FUEL_ZERO_HOLD"
            elif is_raw_nan:
                quality_flag = "INVALID_RAW_HOLD"
            else:
                quality_flag = "VALID"

            return {
                "vehicle_id": vehicle_id,
                "fuel_time": fuel_time.isoformat(),
                "raw_fuel_liters": None if is_raw_nan else round(float(fuel_level), 2),
                "clean_fuel_liters": round(float(clean_fuel_val), 2) if clean_fuel_val is not None else None,
                "ai_signal_state": ai_signal_state,
                "confidence": round(float(conf), 4),
                "quality_flag": quality_flag,
                "latency_ms": round(latency_ms, 3),
                # Causal features used for this exact prediction.  They are
                # returned so diagnostic tools can inspect the streaming path
                # without rebuilding features from an offline DataFrame.
                "rolling_std": round(float(rolling_std_12), 4),
                "capacity_est": round(float(cap), 4),
                "noise_sigma_liters": round(float(sigma), 4),
                "flat_jitter_threshold": round(float(flat), 4),
                "event_threshold": round(float(event), 4),
                "time_gap_minutes": round(float(time_gap_min), 4),
            }
