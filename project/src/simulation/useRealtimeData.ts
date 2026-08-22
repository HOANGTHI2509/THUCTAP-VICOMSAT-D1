import { useEffect, useState } from "react";
import type { TripPoint } from "./types";
import { findTrip, DEFAULT_TRIP_ID } from "./trips";

export function useRealtimeData() {
  const [windowPoints, setWindowPoints] = useState<TripPoint[]>([]);
  const [current, setCurrent] = useState<TripPoint | null>(null);
  const [events, setEvents] = useState<TripPoint[]>([]);
  const [cursor, setCursor] = useState(0);
  
  // Fake properties to satisfy UI components that expect Simulation behavior
  const playing = true; 
  const speed = 1;
  const duration = 0;
  const windowSec = 10 * 60; // 10 minutes window

  const defaultTrip = findTrip(DEFAULT_TRIP_ID);
  const startMs = defaultTrip ? new Date(defaultTrip.startTimeStr).getTime() : 0;

  useEffect(() => {
    let interval = setInterval(async () => {
      try {
        const response = await fetch("http://localhost:8000/api/data?limit=200");
        if (!response.ok) return;
        const data = await response.json();
        
        // Map backend format to TripPoint format
        const mappedData: TripPoint[] = data.map((d: any, idx: number) => {
          return {
            t: (new Date(d.time).getTime() - startMs) / 1000,
            speed: d.speed || 0, // Include speed if available
            rawFuel: d.raw,
            traditionalKalman: d.kalman,
            adaptiveKalman: d.adaptive,
            mlKalman: d.ai_enhanced,
            noise: d.raw - d.ai_enhanced,
            isSpike: Math.abs(d.raw - d.ai_enhanced) > 5 // simple spike logic
          };
        });
        
        if (mappedData.length > 0) {
          setWindowPoints(mappedData);
          setCurrent(mappedData[mappedData.length - 1]);
          setEvents(mappedData.filter(p => p.isSpike).slice(-10)); // keep last 10 spikes
          setCursor(mappedData[mappedData.length - 1].t);
        }
      } catch (err) {
        console.error("Lỗi lấy dữ liệu Realtime:", err);
      }
    }, 2000);
    
    return () => clearInterval(interval);
  }, []);

  return {
    playing,
    speed,
    cursor,
    duration,
    current,
    window: windowPoints,
    windowSec,
    events,
    loading: false,
    
    // Dummy functions
    play: () => {},
    pause: () => {},
    toggle: () => {},
    setSpeed: () => {},
    seek: () => {},
    reset: () => {}
  };
}
