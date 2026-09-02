import { useEffect, useState } from "react";
import type { TripPoint } from "./types";
import { findTrip, DEFAULT_TRIP_ID } from "./trips";

export function useRealtimeData() {
  const [windowPoints, setWindowPoints] = useState<TripPoint[]>([]);
  const [current, setCurrent] = useState<TripPoint | null>(null);
  const [events, setEvents] = useState<TripPoint[]>([]);
  const [cursor, setCursor] = useState(0);
  const [windowSec, setWindowSec] = useState(10 * 60);

  const fetchLatest = async () => {
    try {
      const response = await fetch("http://localhost:8000/api/data?limit=300");
      if (!response.ok) return;
      const data = await response.json();
      
      if (!data || data.length === 0) return;

      // Sắp xếp các điểm theo thời gian tăng dần
      const sortedData = [...data].sort((a, b) => {
        const ta = new Date(a.time).getTime();
        const tb = new Date(b.time).getTime();
        return ta - tb;
      });

      // Chỉ lấy các điểm thuộc cùng chiếc xe đang active để không bị trộn lẫn dữ liệu giữa 2 xe
      const latestVid = sortedData[sortedData.length - 1].vehicle_id;
      const filteredData = sortedData.filter((d: any) => d.vehicle_id === latestVid);

      const baseMs = new Date(filteredData[0].time).getTime();
      
      // Map backend format to TripPoint format
      const mappedData: TripPoint[] = filteredData.map((d: any) => {
        const ptTime = new Date(d.time).getTime();
        const cleanFuel = d.ai_enhanced !== undefined ? d.ai_enhanced : (d.adaptive !== undefined ? d.adaptive : d.raw);
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
          noise: noise,
          isSpike: isSpike,
          vehicleId: d.vehicle_id || d.VehicleID || "24H-04650",
          aiState: d.state || "STABLE_JITTER"
        };
      });
      
      if (mappedData.length > 0) {
        const lastT = mappedData[mappedData.length - 1].t;
        setWindowPoints(mappedData);
        setCurrent(mappedData[mappedData.length - 1]);
        setEvents(mappedData.filter(p => p.isSpike).slice(-10));
        setCursor(lastT);
        setWindowSec(Math.max(600, lastT));
      }
    } catch (err) {
      console.error("Lỗi lấy dữ liệu Realtime:", err);
    }
  };

  useEffect(() => {
    fetchLatest();
    const interval = setInterval(fetchLatest, 1000);
    return () => clearInterval(interval);
  }, []);

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
    
    // Dummy functions
    play: () => {},
    pause: () => {},
    toggle: () => {},
    setSpeed: () => {},
    seek: () => {},
    reset: () => {}
  };
}
