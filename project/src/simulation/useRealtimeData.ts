import { useEffect, useState } from "react";
import type { TripPoint } from "./types";
import { findTrip, DEFAULT_TRIP_ID } from "./trips";

export function useRealtimeData(targetVehicleId?: string) {
  const [windowPoints, setWindowPoints] = useState<TripPoint[]>([]);
  const [current, setCurrent] = useState<TripPoint | null>(null);
  const [events, setEvents] = useState<TripPoint[]>([]);
  const [cursor, setCursor] = useState(0);
  const [windowSec, setWindowSec] = useState(10 * 60);
  const [startTimeStr, setStartTimeStr] = useState<string>("");

  const fetchLatest = async () => {
    try {
      const url = targetVehicleId
        ? `http://localhost:8000/api/data?vehicle_id=${encodeURIComponent(targetVehicleId)}&limit=300`
        : `http://localhost:8000/api/data?limit=300`;
      const response = await fetch(url);
      if (!response.ok) return;
      const data = await response.json();
      
      if (!data || data.length === 0) return;

      // Sắp xếp các điểm theo thời gian tăng dần
      const sortedData = [...data].sort((a, b) => {
        const ta = new Date(a.time).getTime();
        const tb = new Date(b.time).getTime();
        return ta - tb;
      });

      // Lấy các điểm thuộc chiếc xe được chỉ định (hoặc xe cuối cùng nếu không chỉ định)
      const selectedVid = targetVehicleId || sortedData[sortedData.length - 1].vehicle_id;
      const filteredData = sortedData.filter((d: any) => !selectedVid || d.vehicle_id === selectedVid);

      if (filteredData.length === 0) return;

      // Lấy thời gian trung vị (median) để giữ lại toàn bộ cụm dữ liệu xe đang chạy
      // và loại bỏ chính xác các điểm rác bị lệch thời gian > 24 giờ
      let cleanData = filteredData;
      if (filteredData.length > 2) {
        const times = filteredData.map((d: any) => new Date(d.time).getTime()).sort((a, b) => a - b);
        const medianTime = times[Math.floor(times.length / 2)];
        cleanData = filteredData.filter((d: any) => {
          const t = new Date(d.time).getTime();
          return Math.abs(t - medianTime) <= 24 * 3600 * 1000;
        });
      }

      if (cleanData.length === 0) cleanData = filteredData;

      const baseMs = new Date(cleanData[0].time).getTime();
      setStartTimeStr(cleanData[0].time);
      
      // Map backend format to TripPoint format
      const mappedData: TripPoint[] = cleanData.map((d: any) => {
        const ptTime = new Date(d.time).getTime();
        const cleanFuel = d.ai_enhanced !== undefined ? d.ai_enhanced : (d.adaptive !== undefined ? d.adaptive : d.raw);
        const smoothFuel = d.ai_smooth_tracking !== undefined
          ? d.ai_smooth_tracking
          : (d.smooth_tracking !== undefined ? d.smooth_tracking : cleanFuel);
        const rawFuel = d.raw !== undefined ? d.raw : cleanFuel;
        const noise = rawFuel - cleanFuel;
        const isSpike = Math.abs(noise) > 4.0;

        return {
          t: Math.max(0, (ptTime - baseMs) / 1000),
          speed: d.speed || 0,
          rawFuel: rawFuel,
          traditionalKalman: d.kalman !== undefined ? d.kalman : cleanFuel,
          adaptiveKalman: d.adaptive !== undefined ? d.adaptive : cleanFuel,
          mlKalman: cleanFuel,
          aiSmoothTracking: smoothFuel,
          smoothTracking: smoothFuel,
          noise: noise,
          isSpike: isSpike,
          vehicleId: d.vehicle_id || d.VehicleID || targetVehicleId || "21H-03221",
          aiState: d.state || "STABLE_JITTER",
          lat: d.lat,
          lng: d.lng,
          address: d.address,
        };
      });
      
      if (mappedData.length > 0) {
        const lastT = mappedData[mappedData.length - 1].t;
        setWindowPoints(mappedData);
        setCurrent(mappedData[mappedData.length - 1]);
        setEvents(mappedData.filter(p => p.isSpike).slice(-10));
        setCursor(lastT);
        // Cửa sổ cuộn sóng tối đa 20 phút (1200 giây) để biểu đồ cuộn liên tục
        const span = lastT - mappedData[0].t;
        setWindowSec(Math.min(1200, Math.max(600, span)));
      }
    } catch (err) {
      console.error("Lỗi lấy dữ liệu Realtime:", err);
    }
  };

  useEffect(() => {
    fetchLatest();
    const interval = setInterval(fetchLatest, 1000);
    return () => clearInterval(interval);
  }, [targetVehicleId]);

  return {
    playing: true,
    speed: 1,
    cursor,
    duration: cursor,
    current,
    window: windowPoints,
    windowSec,
    events,
    loading: false,
    refetch: fetchLatest,
    startTimeStr,
    
    // Dummy functions
    play: () => {},
    pause: () => {},
    toggle: () => {},
    setSpeed: () => {},
    seek: () => {},
    reset: () => {}
  };
}
