import { useEffect, useRef, useState } from "react";
import { Activity, Gauge, Waves, ZoomIn } from "lucide-react";
import type { Trip, TripPoint } from "@/simulation/types";

interface Props {
  trip: Trip;
  points: TripPoint[];
  cursor: number;
  windowSec: number;
  playing: boolean;
}

type ChartMode = "speed" | "fuel";
const ZOOM_LEVELS = [1, 2, 4, 8];
const DEFAULT_ZOOM = 2;

export default function RollingChart({ trip, points, cursor, windowSec, playing }: Props) {
  const [zoom, setZoom] = useState(DEFAULT_ZOOM);

  return (
    <div className="relative flex-1 min-w-0 rounded-2xl bg-cockpit-900 border border-cockpit-700 shadow-2xl overflow-hidden flex flex-col transition-colors duration-300">
      <ChartPanel mode="speed" points={points} cursor={cursor} windowSec={windowSec} playing={playing} trip={trip} />
      <div className="flex items-center justify-between px-5 py-2 bg-cockpit-900 border-y border-cockpit-700 shrink-0">
        <div className="flex items-center gap-2">
          <Activity size={16} className="text-fuel-filter" />
          <span className="text-xs font-semibold text-cockpit-100">Nhiên liệu — Gốc và sau lọc</span>
        </div>
        <div className="flex items-center gap-1 text-[11px]">
          <ZoomIn size={13} className="text-cockpit-400 mr-1" />
          <span className="text-[10px] uppercase tracking-widest text-cockpit-400 mr-1">Phóng đại</span>
          {ZOOM_LEVELS.map((level) => (
            <button
              key={level}
              onClick={() => setZoom(level)}
              className={`px-2 py-0.5 rounded text-[11px] font-mono transition-all ${zoom === level ? "bg-fuel-filter text-white font-semibold" : "bg-cockpit-700 text-cockpit-300 hover:bg-cockpit-600"}`}
            >
              x{level}
            </button>
          ))}
        </div>
      </div>
      <ChartPanel mode="fuel" points={points} cursor={cursor} windowSec={windowSec} playing={playing} zoom={zoom} trip={trip} />
    </div>
  );
}

function ChartPanel({ mode, points, cursor, windowSec, playing, zoom = 1, trip }: {
  mode: ChartMode;
  points: TripPoint[];
  cursor: number;
  windowSec: number;
  playing: boolean;
  zoom?: number;
  trip: Trip;
}) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const render = () => {
      const canvas = canvasRef.current;
      const container = containerRef.current;
      if (!canvas || !container) return;
      const dpr = window.devicePixelRatio || 1;
      const width = container.clientWidth;
      const height = container.clientHeight;
      canvas.width = width * dpr;
      canvas.height = height * dpr;
      canvas.style.width = `${width}px`;
      canvas.style.height = `${height}px`;
      const ctx = canvas.getContext("2d");
      if (!ctx) return;
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      draw(ctx, width, height, points, cursor, windowSec, mode, zoom, trip);
    };
    render();
    window.addEventListener("resize", render);
    return () => window.removeEventListener("resize", render);
  }, [points, cursor, windowSec, mode, zoom, playing]);

  return (
    <div ref={containerRef} className={`relative w-full ${mode === "speed" ? "h-[150px]" : "h-[230px]"} shrink-0`}>
      <canvas ref={canvasRef} className="block w-full h-full" />
      {mode === "speed" && <div className="absolute top-2 left-5 flex items-center gap-2 pointer-events-none"><Gauge size={15} className="text-speed" /><span className="text-xs font-semibold text-cockpit-200">Vận tốc xe</span><span className="text-[10px] text-cockpit-400">km/h</span></div>}
      {mode === "fuel" && <div className="absolute top-2 left-5 flex items-center gap-4 text-[11px] pointer-events-none"><span className="flex items-center gap-1.5"><span className="w-3 h-0.5 bg-fuel-raw" /><span className="text-cockpit-300">Xăng gốc</span></span><span className="flex items-center gap-1.5"><span className="w-3 h-0.5 bg-fuel-filter" /><span className="text-cockpit-300">Sau lọc</span></span></div>}
      {!playing && points.length > 0 && <div className="absolute top-2 right-4 text-[10px] uppercase tracking-widest text-cockpit-400 bg-cockpit-800/80 px-2 py-1 rounded">Đã tạm dừng</div>}
      {points.length === 0 && <div className="absolute inset-0 flex flex-col items-center justify-center gap-2 text-cockpit-500"><Waves size={34} className="opacity-40" /><span className="text-sm">Bấm Play để bắt đầu đo</span></div>}
    </div>
  );
}

function draw(ctx: CanvasRenderingContext2D, w: number, h: number, points: TripPoint[], cursor: number, windowSec: number, mode: ChartMode, zoom: number, trip: Trip) {
  const padL = 52;
  const padR = 14;
  const padT = 14;
  const padB = 24;
  const plotW = w - padL - padR;
  const plotH = h - padT - padB;
  const tMax = Math.max(cursor, windowSec);
  const tMin = tMax - windowSec;
  const visible = points.filter((point) => point.t >= tMin && point.t <= tMax);
  const speedHex = getCssVar("--color-speed");
  const speedGridHex = getCssVar("--chart-speed-grid");
  const rawHex = getCssVar("--color-fuel-raw");
  const filterHex = getCssVar("--color-fuel-filter");
  const gridColor = mode === "speed" ? speedGridHex : getCssVar("--chart-grid");
  const labelColor = mode === "speed" ? withAlpha(speedHex, 0.7) : getCssVar("--chart-label");
  const xOf = (t: number) => padL + ((t - tMin) / windowSec) * plotW;

  ctx.clearRect(0, 0, w, h);
  ctx.font = "10px ui-monospace, monospace";
  ctx.fillStyle = labelColor;
  ctx.strokeStyle = gridColor;
  ctx.lineWidth = 1;

  const allVisibleValues = visible.flatMap((point) => [point.filteredFuel, point.rawFuel]);
  const minVisible = allVisibleValues.length ? Math.min(...allVisibleValues) : 0;
  const maxVisible = allVisibleValues.length ? Math.max(...allVisibleValues) : 100;
  
  const center = (minVisible + maxVisible) / 2;
  const dataSpan = (maxVisible - minVisible) / 2;
  
  // Cửa sổ Y tối thiểu là 10 lít để không phóng đại quá mức các nhiễu nhỏ, chia cho độ zoom
  const span = Math.max(15, dataSpan * 1.5) / zoom;
  
  const minFuel = Math.max(0, center - span);
  const maxFuel = center + span;
  const yOf = mode === "speed"
    ? (value: number) => padT + (1 - Math.max(0, Math.min(90, value)) / 90) * plotH
    : (value: number) => padT + (1 - (Math.max(minFuel, Math.min(maxFuel, value)) - minFuel) / (maxFuel - minFuel)) * plotH;

  const rows = mode === "speed" ? 4 : 8;
  for (let row = 0; row <= rows; row++) {
    const y = padT + (row / rows) * plotH;
    ctx.beginPath();
    ctx.moveTo(padL, y);
    ctx.lineTo(padL + plotW, y);
    ctx.stroke();
    const value = mode === "speed" ? 90 - row * 22.5 : maxFuel - (row / rows) * (maxFuel - minFuel);
    ctx.textAlign = "right";
    ctx.fillText(value.toFixed(0), padL - 6, y + 3);
  }

  for (let column = 0; column <= 10; column++) {
    const x = padL + (column / 10) * plotW;
    ctx.beginPath();
    ctx.moveTo(x, padT);
    ctx.lineTo(x, padT + plotH);
    ctx.stroke();
    const time = tMin + (column / 10) * windowSec;
    ctx.textAlign = "center";
    
    // Tính toán thời gian thực tế
    const startMs = new Date(trip.startTimeStr).getTime();
    const dt = new Date(startMs + time * 1000);
    
    const timeStr = `${dt.getHours().toString().padStart(2, "0")}:${dt.getMinutes().toString().padStart(2, "0")}`;
    const dateStr = `${dt.getDate().toString().padStart(2, "0")}/${(dt.getMonth() + 1).toString().padStart(2, "0")}`;
    
    ctx.fillText(timeStr, x, padT + plotH + 15);
    ctx.fillText(dateStr, x, padT + plotH + 28);
  }

  if (visible.length === 0) return;
  const accent = mode === "speed" ? speedHex : filterHex;
  const accentAlpha = withAlpha(accent, 0.4);

  if (mode === "fuel") {
    for (const point of visible) {
      if (!point.isSpike) continue;
      ctx.fillStyle = getCssVar("--chart-spike");
      ctx.fillRect(xOf(point.t) - 1.5, padT, 3, plotH);
    }
    drawLine(ctx, visible, (point) => yOf(point.rawFuel), xOf, withAlpha(rawHex, 0.35), 1, 0);
    drawLine(ctx, visible, (point) => yOf(point.rawFuel), xOf, rawHex, 1.6, 6);
    drawLine(ctx, visible, (point) => yOf(point.filteredFuel), xOf, filterHex, 2.4, 8);
  } else {
    drawLine(ctx, visible, (point) => yOf(point.speed), xOf, speedHex, 2.4, 6);
  }

  const last = visible[visible.length - 1];
  const lastX = xOf(last.t);
  ctx.strokeStyle = accentAlpha;
  ctx.setLineDash([4, 4]);
  ctx.beginPath();
  ctx.moveTo(lastX, padT);
  ctx.lineTo(lastX, padT + plotH);
  ctx.stroke();
  ctx.setLineDash([]);
  ctx.fillStyle = accent;
  ctx.beginPath();
  ctx.arc(lastX, yOf(mode === "speed" ? last.speed : last.filteredFuel), 4, 0, Math.PI * 2);
  ctx.fill();
}

function drawLine(ctx: CanvasRenderingContext2D, points: TripPoint[], yOf: (point: TripPoint) => number, xOf: (t: number) => number, color: string, width: number, blur: number) {
  ctx.strokeStyle = color;
  ctx.lineWidth = width;
  ctx.shadowColor = color;
  ctx.shadowBlur = blur;
  ctx.beginPath();
  points.forEach((point, index) => {
    const x = xOf(point.t);
    const y = yOf(point);
    if (index === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
  });
  ctx.stroke();
  ctx.shadowBlur = 0;
}

function getCssVar(name: string): string {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}

function withAlpha(hex: string, alpha: number): string {
  const value = hex.replace("#", "");
  if (value.length !== 6) return hex;
  return `rgba(${parseInt(value.slice(0, 2), 16)}, ${parseInt(value.slice(2, 4), 16)}, ${parseInt(value.slice(4, 6), 16)}, ${alpha})`;
}
