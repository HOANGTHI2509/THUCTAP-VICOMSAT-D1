import { useCallback, useEffect, useRef, useState } from "react";
import type { Trip, TripPoint } from "./types";

const WINDOW_SEC = 10 * 60 * 24; // Hiển thị 24 tiếng trên cửa sổ trượt (do dữ liệu kéo dài hàng tháng)
const STEP_SEC = 300; // Nhảy 5 phút (300 giây) mỗi bước lặp để chạy mượt với dữ liệu thực

export interface SimulationState {
  playing: boolean;
  speed: number;
  cursor: number;
  duration: number;
  current: TripPoint | null;
  window: TripPoint[];
  windowSec: number;
  events: TripPoint[];
  loading: boolean;
}

export function useSimulation(trip: Trip | null): SimulationState & {
  play: () => void;
  pause: () => void;
  toggle: () => void;
  setSpeed: (s: number) => void;
  seek: (t: number) => void;
  reset: () => void;
} {
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeedState] = useState(100); // Tốc độ x100 vì dữ liệu rất dài
  const [cursor, setCursor] = useState(0);
  const [current, setCurrent] = useState<TripPoint | null>(null);
  const [windowPoints, setWindowPoints] = useState<TripPoint[]>([]);
  const [events, setEvents] = useState<TripPoint[]>([]);
  const [loading, setLoading] = useState(false);
  const [realData, setRealData] = useState<TripPoint[]>([]);

  const tripRef = useRef(trip);
  const cursorRef = useRef(0);
  const dataIndexRef = useRef(0);
  const windowRef = useRef<TripPoint[]>([]);
  const eventsRef = useRef<TripPoint[]>([]);
  const playingRef = useRef(false);
  const speedRef = useRef(speed);
  const rafRef = useRef<number | null>(null);
  const lastTsRef = useRef<number | null>(null);
  const accRef = useRef(0);

  // Fetch real data
  useEffect(() => {
    tripRef.current = trip;
    if (trip) {
      setLoading(true);
      setPlaying(false);
      
      const carId = trip.vehicleId; // e.g. "car1"
      fetch(`/data_${carId}.json`)
        .then(res => res.json())
        .then((data: TripPoint[]) => {
            setRealData(data);
            setLoading(false);
            cursorRef.current = 0;
            dataIndexRef.current = 0;
            windowRef.current = [];
            setCursor(0);
            setCurrent(data.length > 0 ? data[0] : null);
            setWindowPoints([]);
            eventsRef.current = (data.length > 0 && data[0].isSpike) ? [data[0]] : [];
            setEvents(eventsRef.current);
            accRef.current = 0;
        })
        .catch(err => {
            console.error("Failed to load real data", err);
            setLoading(false);
        });
    } else {
        setRealData([]);
    }
  }, [trip]);

  useEffect(() => { playingRef.current = playing; }, [playing]);
  useEffect(() => { speedRef.current = speed; }, [speed]);

  const stepOnce = useCallback(() => {
    const t = tripRef.current;
    if (!t || realData.length === 0) return;
    
    let idx = dataIndexRef.current + 1;
    if (idx >= realData.length) {
      setPlaying(false);
      return;
    }
    
    dataIndexRef.current = idx;
    const pt = realData[idx];
    setCurrent(pt);
    
    const w = windowRef.current;
    w.push(pt);
    // Giữ lại các điểm trong cửa sổ thời gian
    while (w.length > 0 && w[0].t < pt.t - WINDOW_SEC) w.shift();
    setWindowPoints([...w]);
    
    if (pt.isSpike) {
      eventsRef.current.push(pt);
      setEvents([...eventsRef.current]);
    }
    
    cursorRef.current = pt.t;
    setCursor(pt.t);
  }, [realData]);

  useEffect(() => {
    if (!playing) {
      if (rafRef.current) cancelAnimationFrame(rafRef.current);
      rafRef.current = null;
      lastTsRef.current = null;
      return;
    }
    const tick = (ts: number) => {
      if (lastTsRef.current == null) lastTsRef.current = ts;
      const dt = (ts - lastTsRef.current) / 1000;
      lastTsRef.current = ts;
      accRef.current += dt * speedRef.current;
      
      let steps = 0;
      // 1 đơn vị accRef tương ứng với 1 dòng dữ liệu (index)
      while (accRef.current >= 1 && steps < 600) {
        stepOnce();
        accRef.current -= 1;
        steps++;
      }
      
      rafRef.current = requestAnimationFrame(tick);
    };
    rafRef.current = requestAnimationFrame(tick);
    return () => {
      if (rafRef.current) cancelAnimationFrame(rafRef.current);
      rafRef.current = null;
      lastTsRef.current = null;
    };
  }, [playing, stepOnce]);

  const play = useCallback(() => { if(!loading) setPlaying(true); }, [loading]);
  const pause = useCallback(() => { setPlaying(false); }, []);
  const toggle = useCallback(() => { if(!loading) setPlaying((p) => !p); }, [loading]);
  const setSpeed = useCallback((s: number) => { setSpeedState(s); }, []);

  const seek = useCallback((tSec: number) => {
    const tr = tripRef.current;
    if (!tr || realData.length === 0) return;
    const target = Math.max(0, Math.min(tr.durationSec, Math.floor(tSec)));
    cursorRef.current = target;
    accRef.current = 0;
    setCursor(target);
    
    if (target === 0) {
      dataIndexRef.current = 0;
      setCurrent(realData[0]);
      windowRef.current = [];
      setWindowPoints([]);
      eventsRef.current = (realData.length > 0 && realData[0].isSpike) ? [realData[0]] : [];
      setEvents(eventsRef.current);
      return;
    }
    
    // Tìm điểm dữ liệu tại target
    let idx = 0;
    while (idx < realData.length - 1 && realData[idx+1].t <= target) {
        idx++;
    }
    dataIndexRef.current = idx;
    setCurrent(realData[idx]);
    
    // Tái tạo lại window windowRef
    const w: TripPoint[] = [];
    let startIdx = idx;
    while (startIdx > 0 && realData[startIdx].t >= target - WINDOW_SEC) {
        startIdx--;
    }
    for(let i = startIdx; i <= idx; i++) {
        if(realData[i].t >= target - WINDOW_SEC && realData[i].t <= target) {
            w.push(realData[i]);
        }
    }
    
    
    windowRef.current = w;
    setWindowPoints([...w]);
    
    const ev = realData.slice(0, idx + 1).filter(p => p.isSpike);
    eventsRef.current = ev;
    setEvents(ev);
  }, [realData]);

  const reset = useCallback(() => {
    setPlaying(false);
    cursorRef.current = 0;
    dataIndexRef.current = 0;
    windowRef.current = [];
    setCursor(0);
    setCurrent(realData.length > 0 ? realData[0] : null);
    setWindowPoints([]);
    eventsRef.current = (realData.length > 0 && realData[0].isSpike) ? [realData[0]] : [];
    setEvents(eventsRef.current);
    accRef.current = 0;
  }, [realData]);

  return {
    playing, speed, cursor,
    duration: trip?.durationSec ?? 0,
    current,
    window: windowPoints,
    windowSec: WINDOW_SEC,
    events,
    loading,
    play, pause, toggle, setSpeed, seek, reset,
  };
}
