import { Play, Pause, Truck, ChevronDown, Gauge, RotateCcw } from "lucide-react";
import { useEffect, useRef, useState } from "react";
import { VEHICLES } from "@/simulation/trips";
import type { Trip } from "@/simulation/types";

interface Props {
  trip: Trip;
  playing: boolean;
  speed: number;
  cursor: number;
  duration: number;
  onToggle: () => void;
  onSpeed: (s: number) => void;
  onSeek: (t: number) => void;
  onReset: () => void;
  onTrip: (t: Trip) => void;
}

const SPEEDS = [1, 2, 5, 10, 20];

function fmt(s: number): string {
  const m = Math.floor(s / 60);
  const sec = Math.floor(s % 60);
  return `${m}:${sec.toString().padStart(2, "0")}`;
}

function Dropdown({
  label, value, sub, items, onSelect,
}: {
  label: string;
  value: string;
  sub?: string;
  items: { id: string; label: string; sub?: string }[];
  onSelect: (id: string) => void;
}) {
  const [open, setOpen] = useState(false);
  const ref = useRef<HTMLDivElement>(null);
  useEffect(() => {
    const h = (e: MouseEvent) => {
      if (ref.current && !ref.current.contains(e.target as Node)) setOpen(false);
    };
    document.addEventListener("mousedown", h);
    return () => document.removeEventListener("mousedown", h);
  }, []);
  return (
    <div className="relative" ref={ref}>
      <div className="text-[10px] uppercase tracking-widest text-cockpit-400 mb-1">{label}</div>
      <button
        onClick={() => setOpen((o) => !o)}
        className="w-full flex items-center justify-between gap-2 px-3 py-2 rounded-lg
                   bg-cockpit-700 hover:bg-cockpit-600 border border-cockpit-600
                   text-left transition-colors"
      >
        <div className="min-w-0">
          <div className="text-sm text-cockpit-100 truncate">{value}</div>
          {sub && <div className="text-[10px] text-cockpit-400 truncate">{sub}</div>}
        </div>
        <ChevronDown size={16} className={`text-cockpit-300 shrink-0 transition-transform ${open ? "rotate-180" : ""}`} />
      </button>
      {open && (
        <div className="absolute z-30 mt-1 w-full rounded-lg bg-cockpit-800 border border-cockpit-600 shadow-2xl overflow-hidden max-h-72 overflow-y-auto">
          {items.map((it) => (
            <button
              key={it.id}
              onClick={() => { onSelect(it.id); setOpen(false); }}
              className="w-full px-3 py-2 text-left hover:bg-cockpit-700 transition-colors border-b border-cockpit-700 last:border-0"
            >
              <div className="text-sm text-cockpit-100 truncate">{it.label}</div>
              {it.sub && <div className="text-[10px] text-fuel-raw truncate">{it.sub}</div>}
            </button>
          ))}
        </div>
      )}
    </div>
  );
}

export default function RemoteControl({
  trip, playing, speed, cursor, duration,
  onToggle, onSpeed, onSeek, onReset, onTrip,
}: Props) {
  const tripItems = VEHICLES.flatMap((v) =>
    v.trips.map((t) => ({ id: t.id, label: `${v.name} — ${t.tripName}`, sub: t.noiseLabel }))
  );
  const vehicleItems = VEHICLES.map((v) => ({ id: v.id, label: v.name, sub: `${v.trips.length} chuyến` }));

  const [selVehicle, setSelVehicle] = useState(trip.vehicleId);
  const [selTrip, setSelTrip] = useState(trip.id);

  return (
    <div className="w-[260px] shrink-0 rounded-2xl bg-cockpit-850 border border-cockpit-700 shadow-2xl p-4 flex flex-col gap-4">
      <div className="flex items-center gap-2 pb-3 border-b border-cockpit-700">
        <div className="w-9 h-9 rounded-lg bg-cockpit-700 flex items-center justify-center">
          <Gauge size={20} className="text-fuel-filter" />
        </div>
        <div>
          <div className="text-sm font-semibold text-cockpit-100">Remote</div>
          <div className="text-[10px] text-cockpit-400 uppercase tracking-wider">Điều khiển mô phỏng</div>
        </div>
      </div>

      <Dropdown
        label="Chọn xe"
        value={VEHICLES.find((v) => v.id === selVehicle)?.name ?? ""}
        sub={`${VEHICLES.find((v) => v.id === selVehicle)?.trips.length ?? 0} chuyến`}
        items={vehicleItems}
        onSelect={(id) => {
          setSelVehicle(id);
          const v = VEHICLES.find((v) => v.id === id);
          if (v && v.trips[0]) {
            setSelTrip(v.trips[0].id);
            onTrip(v.trips[0]);
          }
        }}
      />

      <Dropdown
        label="Chọn chuyến đi"
        value={trip.tripName}
        sub={trip.noiseLabel}
        items={tripItems}
        onSelect={(id) => {
          setSelTrip(id);
          const t = VEHICLES.flatMap((v) => v.trips).find((t) => t.id === id);
          if (t) onTrip(t);
        }}
      />

      <div className="flex items-center gap-2 text-xs text-cockpit-300">
        <Truck size={14} className="text-cockpit-400" />
        <span className="truncate">{trip.vehicleName}</span>
      </div>

      <button
        onClick={onToggle}
        className={`relative w-full h-14 rounded-xl font-semibold text-base flex items-center justify-center gap-2 transition-all
          ${playing
            ? "bg-fuel-filter/20 border border-fuel-filter/50 text-fuel-filter hover:bg-fuel-filter/30"
            : "bg-fuel-filter text-cockpit-950 hover:brightness-110"}`}
      >
        {playing ? <Pause size={22} /> : <Play size={22} />}
        <span>{playing ? "Tạm dừng" : "Bắt đầu"}</span>
      </button>

      <div>
        <div className="flex items-center justify-between mb-2">
          <span className="text-[10px] uppercase tracking-widest text-cockpit-400">Tốc độ tua</span>
          <span className="text-sm font-mono text-fuel-filter">x{speed}</span>
        </div>
        <div className="grid grid-cols-5 gap-1">
          {SPEEDS.map((s) => (
            <button
              key={s}
              onClick={() => onSpeed(s)}
              className={`py-1.5 rounded-md text-xs font-mono transition-all
                ${speed === s
                  ? "bg-fuel-filter text-cockpit-950 font-semibold"
                  : "bg-cockpit-700 text-cockpit-300 hover:bg-cockpit-600"}`}
            >
              x{s}
            </button>
          ))}
        </div>
      </div>

      <div>
        <div className="flex items-center justify-between mb-2 text-[10px] uppercase tracking-widest text-cockpit-400">
          <span>Vị trí</span>
          <span className="font-mono text-cockpit-200 normal-case tracking-normal">
            {fmt(cursor)} / {fmt(duration)}
          </span>
        </div>
        <input
          type="range"
          min={0}
          max={duration}
          value={cursor}
          onChange={(e) => onSeek(Number(e.target.value))}
          className="w-full accent-fuel-filter h-1.5"
        />
      </div>

      <button
        onClick={onReset}
        className="flex items-center justify-center gap-2 py-2 rounded-lg text-xs
                   bg-cockpit-700 text-cockpit-300 hover:bg-cockpit-600 transition-colors"
      >
        <RotateCcw size={14} />
        <span>Về đầu chuyến</span>
      </button>
    </div>
  );
}
