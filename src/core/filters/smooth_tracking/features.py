"""Trich xuat dac trung causal cho AI va rule engine."""

import math
import statistics
from typing import Dict, Optional, Tuple

from .config import SmoothTrackingConfig
from .state import MotionEvidence, TrendEvidence, VehicleFilterContext, WindowEvidence


class CausalFeatureExtractor:
    """Tinh dac trung chi tu diem hien tai va lich su cua cung xe."""

    def __init__(self, config: SmoothTrackingConfig):
        self.config = config

    @staticmethod
    def _valid_coordinate(
        latitude: Optional[float],
        longitude: Optional[float],
    ) -> Optional[Tuple[float, float]]:
        try:
            lat = float(latitude)
            lng = float(longitude)
        except (TypeError, ValueError):
            return None
        if math.isnan(lat) or math.isnan(lng):
            return None
        if not (-90.0 <= lat <= 90.0 and -180.0 <= lng <= 180.0):
            return None
        # Queue/API defaults use (0, 0) to mean "coordinate unavailable".
        if lat == 0.0 and lng == 0.0:
            return None
        return lat, lng

    @staticmethod
    def _haversine_meters(
        first: Tuple[float, float],
        second: Tuple[float, float],
    ) -> float:
        lat1, lng1 = map(math.radians, first)
        lat2, lng2 = map(math.radians, second)
        dlat = lat2 - lat1
        dlng = lng2 - lng1
        a = math.sin(dlat / 2.0) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlng / 2.0) ** 2
        return 6_371_000.0 * 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))

    def motion_evidence(
        self,
        context: VehicleFilterContext,
        speed: float,
        latitude: Optional[float],
        longitude: Optional[float],
    ) -> MotionEvidence:
        """Classify motion quality without assigning any business event meaning."""
        current = self._valid_coordinate(latitude, longitude)
        coordinates = [
            coordinate
            for coordinate in list(context.history_coordinates)[-self.config.motion_window_points + 1 :]
            if coordinate is not None
        ]
        if current is not None:
            coordinates.append(current)

        count = len(coordinates)
        if count < 2:
            return MotionEvidence(
                state="UNCERTAIN",
                confidence=0.0,
                gps_displacement_meters=0.0,
                gps_radius_meters=0.0,
                gps_sample_count=count,
                has_gps=count > 0,
            )

        displacement = self._haversine_meters(coordinates[0], coordinates[-1])
        center = (
            sum(coordinate[0] for coordinate in coordinates) / count,
            sum(coordinate[1] for coordinate in coordinates) / count,
        )
        radius = max(
            self._haversine_meters(coordinate, center)
            for coordinate in coordinates
        )
        enough_points = count >= self.config.low_motion_min_gps_points
        low_speed = speed <= self.config.parked_speed_kmh
        high_speed = speed > self.config.moving_speed_kmh
        gps_moving = displacement >= self.config.moving_min_gps_displacement_meters
        gps_clustered = radius <= self.config.low_motion_radius_meters

        if enough_points and low_speed and gps_clustered:
            state = "LOW_MOTION"
            confidence = min(1.0, 0.65 + 0.1 * count)
        elif enough_points and high_speed and gps_moving:
            state = "MOVING"
            confidence = min(1.0, 0.65 + 0.1 * count)
        else:
            state = "UNCERTAIN"
            confidence = 0.4 if enough_points else 0.2

        return MotionEvidence(
            state=state,
            confidence=confidence,
            gps_displacement_meters=displacement,
            gps_radius_meters=radius,
            gps_sample_count=count,
            has_gps=True,
        )

    def window_evidence(
        self,
        context: VehicleFilterContext,
        raw_fuel: float,
        jitter: float,
    ) -> WindowEvidence:
        values = list(context.history_fuel)[-5:] + [raw_fuel]
        changes = [values[index] - values[index - 1] for index in range(1, len(values))]
        total_variation = sum(abs(change) for change in changes)
        net_change = values[-1] - values[0] if len(values) >= 2 else 0.0
        directionality = abs(net_change) / total_variation if total_variation > 1e-6 else 0.0
        local_std = float(statistics.pstdev(values)) if len(values) >= 3 else 0.0
        local_range = float(max(values) - min(values)) if values else 0.0
        directional_down = (
            net_change <= -max(1.5, jitter * 0.75)
            and directionality >= self.config.downward_directionality_min
        )
        return WindowEvidence(
            recent_values=values,
            directionality=directionality,
            local_std=local_std,
            local_range=local_range,
            directional_down=directional_down,
        )

    def trend_evidence(
        self,
        context: VehicleFilterContext,
        raw_fuel: float,
        speed: float,
        jitter: float,
    ) -> TrendEvidence:
        values = list(context.history_fuel)[-9:] + [raw_fuel]
        speeds = list(context.history_speed)[-9:] + [speed]
        if len(values) < 6:
            return TrendEvidence(robust_downtrend=False, target=raw_fuel)

        middle = len(values) // 2
        first_level = float(statistics.median(values[:middle]))
        second_level = float(statistics.median(values[middle:]))
        level_drop = first_level - second_level
        moving_ratio = sum(value > self.config.moving_speed_kmh for value in speeds) / len(speeds)

        mean_index = (len(values) - 1) * 0.5
        mean_level = sum(values) / len(values)
        denominator = sum((index - mean_index) ** 2 for index in range(len(values)))
        slope = (
            sum(
                (index - mean_index) * (value - mean_level)
                for index, value in enumerate(values)
            ) / denominator
            if denominator > 0.0
            else 0.0
        )
        target = mean_level + slope * (len(values) - 1 - mean_index)
        has_moving = any(value > self.config.moving_speed_kmh for value in speeds)
        robust_downtrend = (
            (moving_ratio >= 0.15 or has_moving)
            and level_drop >= max(2.0, jitter * 1.25)
            and slope <= -max(0.2, jitter * 0.12)
        )
        return TrendEvidence(robust_downtrend=robust_downtrend, target=target)

    def model_features(
        self,
        context: VehicleFilterContext,
        raw_fuel: float,
        speed: float,
        dt_minutes: float,
        motion: Optional[MotionEvidence] = None,
    ) -> Dict[str, float]:
        """Tao vector dac trung tuong thich model hien tai."""
        valid_history = [
            value
            for value in context.history_fuel
            if value is not None and not math.isnan(value) and value > 0
        ]
        previous_valid = valid_history[-1] if valid_history else raw_fuel
        delta_fuel = raw_fuel - previous_valid
        capacity = context.capacity_est
        sigma = max(
            self.config.noise_sigma_floor,
            self.config.noise_sigma_capacity_ratio * capacity,
        )

        all_valid = valid_history + [raw_fuel]
        rolling_std = float(statistics.pstdev(all_valid)) if len(all_valid) >= 3 else 0.0
        median3 = float(statistics.median(all_valid[-3:]))
        last5 = all_valid[-5:]
        range5 = float(max(last5) - min(last5)) if last5 else 0.0
        last7 = all_valid[-7:]
        range7 = float(max(last7) - min(last7)) if last7 else range5
        flat_jitter = max(
            self.config.jitter_floor,
            self.config.jitter_capacity_ratio * capacity,
        )
        spike_threshold = max(
            self.config.spike_floor,
            self.config.spike_capacity_ratio * capacity,
        )
        event_threshold = max(
            self.config.event_floor,
            self.config.event_capacity_ratio * capacity,
        )

        return {
            "FuelLevel": raw_fuel,
            "FuelPct": raw_fuel / capacity if capacity else 0.0,
            "Speed": speed,
            "MotionSpeedKmh": speed,
            "TimeGapMinutes": dt_minutes,
            "DeltaFuel": delta_fuel,
            "DeltaPct": delta_fuel / capacity if capacity else 0.0,
            "AbsDeltaFuel": abs(delta_fuel),
            "DeltaOverNoise": delta_fuel / sigma if sigma else 0.0,
            "RollingStd12": rolling_std,
            "RollingStdPct": rolling_std / capacity if capacity else 0.0,
            "DistanceMeters": motion.gps_displacement_meters if motion else 0.0,
            "GpsSpeedKmh": speed,
            "HasGPS": 1.0 if motion and motion.has_gps else 0.0,
            "capacity_est": capacity,
            "noise_sigma_liters": sigma,
            "flat_jitter_threshold": flat_jitter,
            "spike_threshold": spike_threshold,
            "event_threshold": event_threshold,
            "PrevMedian3": median3,
            "FutureMedian3": median3,
            "FutureMedian5": median3,
            "ReturnToPrevLevel": 0.0,
            "LocalRange5": range5,
            "LocalRange7": range7,
            "PeakReversalFlag": 0,
            "ValleyReversalFlag": 0,
            "TransientScore": 0.0,
        }
