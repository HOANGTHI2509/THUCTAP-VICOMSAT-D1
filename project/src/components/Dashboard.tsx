import { useMemo } from "react";
import { Gauge as GaugeIcon, Fuel, ShieldCheck, AlertTriangle, Radio } from "lucide-react";
import type { TripPoint } from "@/simulation/types";

interface Props {
  current: TripPoint | null;
  playing: boolean;
}

function MiniGauge({ value, min, max, unit, label, color, icon }: {
  value: number;
  min: number;
  max: number;
  unit: string;
  label: string;
  color: string;
  icon: React.ReactNode;
}) {
  const pct = Math.max(0, Math.min(1, (value - min) / (max - min)));
  const angle = -120 + pct * 240;
  const cx = 40, cy = 40, r = 30;
  const startA = (-120 - 90) * (Math.PI / 180);
  const endA = (120 - 90) * (Math.PI / 180);
  const sweep = endA - startA;
  const valA = startA + pct * sweep;
  const p0 = { x: cx + r * Math.cos(startA), y: cy + r * Math.sin(startA) };
  const p1 = { x: cx + r * Math.cos(endA), y: cy + r * Math.sin(endA) };
  const pv = { x: cx + r * Math.cos(valA), y: cy + r * Math.sin(valA) };
  const largeArc = sweep > Math.PI ? 1 : 0;
  const largeArcVal = (valA - startA) > Math.PI ? 1 : 0;

  return (
    <div className="flex items-center gap-3">
      <div className="relative shrink-0">
        <svg viewBox="0 0 80 80" className="w-[72px] h-[72px]">
          <path d={`M ${p0.x} ${p0.y} A ${r} ${r} 0 ${largeArc} 1 ${p1.x} ${p1.y}`}
                fill="none" stroke="var(--color-cockpit-700)" strokeWidth="5" strokeLinecap="round" />
          {pct > 0.001 && (
            <path d={`M ${p0.x} ${p0.y} A ${r} ${r} 0 ${largeArcVal} 1 ${pv.x} ${pv.y}`}
                  fill="none" stroke={color} strokeWidth="5" strokeLinecap="round"
                  style={{ filter: `drop-shadow(0 0 3px ${color})` }} />
          )}
          <line x1={cx} y1={cy} x2={cx + (r - 7) * Math.cos(valA)} y2={cy + (r - 7) * Math.sin(valA)}
                stroke={color} strokeWidth="2" strokeLinecap="round" />
          <circle cx={cx} cy={cy} r="3" fill={color} />
        </svg>
        <div className="absolute top-1 left-1/2 -translate-x-1/2" style={{ color }}>{icon}</div>
      </div>
      <div className="min-w-0">
        <div className="text-xl font-mono font-semibold leading-none" style={{ color }}>
          {value.toFixed(0)}
          <span className="text-[10px] text-cockpit-400 ml-1">{unit}</span>
        </div>
        <div className="text-[10px] uppercase tracking-widest text-cockpit-400 mt-1">{label}</div>
      </div>
    </div>
  );
}

function FuelReadout({ value, label, color, flashing }: {
  value: number;
  label: string;
  color: string;
  flashing: boolean;
}) {
  return (
    <div className="flex items-center gap-3">
      <div className="w-[72px] h-[72px] rounded-full border-[3px] flex flex-col items-center justify-center relative shrink-0"
           style={{ borderColor: color, boxShadow: flashing ? `0 0 14px ${color}` : `0 0 4px ${color}55` }}>
        <Fuel size={16} style={{ color }} />
        <div className={`text-lg font-mono font-semibold leading-none mt-0.5 ${flashing ? "animate-alertflash" : ""}`} style={{ color }}>
          {value.toFixed(1)}
        </div>
      </div>
      <div className="min-w-0">
        <div className="text-[10px] uppercase tracking-widest text-cockpit-400">{label}</div>
      </div>
    </div>
  );
}

export default function Dashboard({ current, playing }: Props) {
  const speed = current?.speed ?? 0;
  const raw = current?.rawFuel ?? 0;
  const filtered = current?.adaptiveKalman ?? 0;
  const isSpike = current?.isSpike ?? false;
  const alert = isSpike && playing;

  const statusText = useMemo(() => {
    if (!playing) return "Đỗ im";
    if (!current) return "Sẵn sàng";
    if (alert) return "Đang chống nhiễu";
    return "Bình thường";
  }, [playing, current, alert]);

  return (
    <div className="rounded-2xl bg-cockpit-850 border border-cockpit-700 shadow-2xl overflow-hidden">
      <div className="flex items-center justify-between px-5 py-2 border-b border-cockpit-700">
        <div className="flex items-center gap-2">
          <Radio size={15} className="text-cockpit-400" />
          <span className="text-xs uppercase tracking-widest text-cockpit-400">Táp-lô trực tiếp</span>
        </div>
        <div className="text-[10px] text-cockpit-500 font-mono">{statusText}</div>
      </div>

      <div className="flex items-center justify-around gap-4 px-5 py-3 flex-wrap">
        <MiniGauge value={speed} min={0} max={90} unit="km/h" label="Vận tốc"
          color="var(--color-speed)" icon={<GaugeIcon size={12} />} />
        <FuelReadout value={raw} label="Xăng gốc (nhiễu)" color="var(--color-fuel-raw)" flashing={alert} />
        <FuelReadout value={filtered} label="Xăng sau lọc" color="var(--color-fuel-filter)" flashing={false} />
        <div className="flex items-center gap-3">
          <div className={`w-[72px] h-[72px] rounded-2xl flex flex-col items-center justify-center gap-1 transition-all shrink-0
              ${alert
                ? "bg-signal-warn/20 border-2 border-signal-warn animate-redglow"
                : "bg-signal-safe/15 border-2 border-signal-safe/60 animate-pulseglow"}`}>
            {alert ? <AlertTriangle size={22} className="text-signal-warn" /> : <ShieldCheck size={22} className="text-signal-safe" />}
            <span className={`text-[10px] font-semibold uppercase tracking-wider ${alert ? "text-signal-warn" : "text-signal-safe"}`}>
              {alert ? "Cảnh báo" : "An toàn"}
            </span>
          </div>
          <div className="text-[10px] uppercase tracking-widest text-cockpit-400">Tín hiệu</div>
        </div>
      </div>

    </div>
  );
}
