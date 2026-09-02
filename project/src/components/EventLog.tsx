import { ScrollText } from "lucide-react";
import type { TripPoint, Trip } from "@/simulation/types";

interface Props {
  points: TripPoint[];
  trip: Trip | null;
}

export default function EventLog({ points, trip }: Props) {
  const startMs = trip ? new Date(trip.startTimeStr).getTime() : 0;

  const events = points
    .filter((point) => point.isSpike)
    .slice(-6)
    .reverse()
    .map((point) => {
      let timeStr = "";
      if (startMs > 0) {
        const pointTime = new Date(startMs + point.t * 1000);
        timeStr = pointTime.toLocaleString("vi-VN", {
          day: "2-digit",
          month: "2-digit",
          hour: "2-digit",
          minute: "2-digit",
        });
      } else {
        const minutes = Math.floor(point.t / 60);
        const seconds = Math.floor(point.t % 60);
        timeStr = `${minutes}:${seconds.toString().padStart(2, "0")}`;
      }
      
      const noiseAbs = Math.abs(point.noise);
      return {
        time: timeStr,
        kind: noiseAbs > 10 ? "Đột biến" : noiseAbs > 5 ? "Nhiễu gai" : "Hố rớt",
        noise: point.noise,
        speed: point.speed,
      };
    });

  return (
    <section className="rounded-2xl bg-cockpit-900 border border-cockpit-700 shadow-2xl px-5 py-3">
      <div className="flex items-center gap-2 mb-2">
        <ScrollText size={14} className="text-cockpit-400" />
        <span className="text-[11px] uppercase tracking-widest text-cockpit-400">Nhật ký ghi nhận</span>
        <span className="text-[10px] text-cockpit-500 ml-auto">{events.length} sự kiện</span>
      </div>
      <div className="flex flex-col gap-0.5 max-h-28 overflow-y-auto">
        {events.length === 0 ? (
          <div className="text-xs text-cockpit-500 italic py-1">Chưa có sự kiện nhiễu nào trong khung giờ này.</div>
        ) : (
          events.map((event, index) => (
            <div key={`${event.time}-${index}`} className="flex flex-wrap items-center gap-x-3 gap-y-0.5 text-[11px] font-mono py-0.5">
              <span className="text-cockpit-400 w-24 shrink-0">{event.time}</span>
              <span className={`shrink-0 w-20 ${event.noise > 0 ? "text-fuel-raw" : "text-signal-warn"}`}>{event.kind}</span>
              <span className="text-cockpit-300">nhiễu {event.noise > 0 ? "+" : ""}{event.noise.toFixed(1)} L</span>
              <span className="text-cockpit-500">· vận tốc {event.speed.toFixed(0)} km/h</span>
            </div>
          ))
        )}
      </div>
    </section>
  );
}
