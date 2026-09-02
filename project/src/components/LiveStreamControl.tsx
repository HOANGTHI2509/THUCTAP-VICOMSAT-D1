import { useState } from "react";
import { Activity, Wifi, Radio, Cpu, CheckCircle2, ShieldCheck, MapPin, Gauge, Loader2 } from "lucide-react";
import type { TripPoint } from "@/simulation/types";

interface Props {
  current: TripPoint | null;
  pointsCount: number;
  onSwitch?: () => void;
}

const VEHICLES_LIST = [
  { id: "24H-04650", name: "Xe tải 24H-04650", tag: "fulltt (Tổng hợp)", plate: "24H-04650" },
  { id: "29C-92841", name: "Xe cao tốc 29C-92841", tag: "Cao tốc Ninh Bình", plate: "29C-92841" },
  { id: "29H-77123", name: "Xe leo dốc 29H-77123", tag: "TEST DO DOC", plate: "29H-77123" },
];

export default function LiveStreamControl({ current, pointsCount, onSwitch }: Props) {
  const [loadingCar, setLoadingCar] = useState<string | null>(null);
  const vehicleId = current?.vehicleId || "24H-04650";
  const aiState = current?.aiState || "STABLE_JITTER";

  const handleSwitchCar = async (carId: string) => {
    setLoadingCar(carId);
    try {
      await fetch("http://localhost:8000/api/switch-vehicle", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ vehicle_id: carId }),
      });
      onSwitch?.();
    } catch (err) {
      console.error("Lỗi khi chuyển xe:", err);
    } finally {
      setTimeout(() => setLoadingCar(null), 300);
    }
  };

  return (
    <aside className="w-80 shrink-0 bg-cockpit-850 border border-cockpit-700 rounded-2xl p-4 flex flex-col justify-between shadow-2xl">
      <div className="space-y-4">
        {/* Header */}
        <div className="flex items-center justify-between border-b border-cockpit-700 pb-3">
          <div className="flex items-center gap-2">
            <div className="w-8 h-8 rounded-lg bg-red-500/10 border border-red-500/30 flex items-center justify-center">
              <Wifi size={16} className="text-red-400 animate-pulse" />
            </div>
            <div>
              <h2 className="text-xs font-bold uppercase tracking-wider text-cockpit-100">Live Telemetry</h2>
              <p className="text-[10px] text-cockpit-400">Luồng API Realtime</p>
            </div>
          </div>
          <span className="flex items-center gap-1.5 px-2 py-0.5 rounded-full bg-emerald-500/20 text-emerald-400 text-[10px] font-mono font-semibold border border-emerald-500/30">
            <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-ping" />
            Online
          </span>
        </div>

        {/* Thông tin xe đang stream */}
        <div className="bg-cockpit-900 border border-cockpit-750 rounded-xl p-3 space-y-2">
          <div className="text-[10px] uppercase tracking-widest text-cockpit-400 font-semibold">Xe đang nhận diện</div>
          <div className="flex items-center justify-between">
            <span className="text-base font-mono font-bold text-white tracking-wide">{vehicleId}</span>
            <span className="text-[10px] px-2 py-0.5 rounded bg-blue-500/20 text-blue-400 border border-blue-500/30 font-medium">
              Cảm biến 500L
            </span>
          </div>
          <div className="text-[11px] text-cockpit-400 flex items-center gap-2 pt-1 border-t border-cockpit-800">
            <Activity size={12} className="text-fuel-filter" />
            <span>Đã nhận: <b className="text-cockpit-200 font-mono">{pointsCount}</b> điểm đo</span>
          </div>
        </div>

        {/* Trạng thái xử lý của AI Model */}
        <div className="bg-cockpit-900 border border-cockpit-750 rounded-xl p-3 space-y-2">
          <div className="flex items-center justify-between">
            <span className="text-[10px] uppercase tracking-widest text-cockpit-400 font-semibold flex items-center gap-1.5">
              <Cpu size={12} className="text-purple-400" />
              Mô hình Causal AI v3
            </span>
            <span className="text-[10px] text-purple-300 font-mono font-semibold">~75ms/điểm</span>
          </div>

          <div className="p-2.5 rounded-lg bg-cockpit-950 border border-cockpit-800 space-y-1.5">
            <div className="text-[10px] text-cockpit-400">Nhãn phân loại tức thời:</div>
            <div className="text-xs font-mono font-bold text-purple-400 tracking-wide">
              {aiState}
            </div>
            <div className="flex items-center justify-between text-[10px] text-cockpit-400 pt-1">
              <span>Độ tin cậy:</span>
              <span className="text-emerald-400 font-mono font-semibold">97.8%</span>
            </div>
          </div>
        </div>

        {/* Danh sách các xe có thể bấm chọn */}
        <div className="space-y-2">
          <div className="flex items-center justify-between text-[10px] uppercase tracking-widest text-cockpit-400 font-semibold">
            <span>Bấm để đổi xe phát</span>
            <span className="text-[9px] text-fuel-filter lowercase font-mono">1-click switch</span>
          </div>
          <div className="space-y-2">
            {VEHICLES_LIST.map((car) => {
              const isSelected = vehicleId.includes(car.id);
              const isLoadingThis = loadingCar === car.id;
              return (
                <button
                  key={car.id}
                  onClick={() => handleSwitchCar(car.id)}
                  disabled={loadingCar !== null}
                  className={`w-full text-left flex items-center justify-between p-2.5 rounded-xl border text-xs transition-all cursor-pointer shadow-sm ${
                    isSelected
                      ? "bg-blue-500/20 border-blue-500 text-blue-300 shadow-md ring-1 ring-blue-500/40"
                      : "bg-cockpit-900 border-cockpit-800 text-cockpit-400 hover:border-cockpit-600 hover:bg-cockpit-800/80 hover:text-white"
                  }`}
                >
                  <div className="flex items-center gap-2.5">
                    <span className="text-base">🚗</span>
                    <div>
                      <div className={`font-medium ${isSelected ? "text-white font-bold" : ""}`}>{car.name}</div>
                      <div className="text-[9px] text-cockpit-500">{car.tag}</div>
                    </div>
                  </div>
                  {isLoadingThis ? (
                    <Loader2 size={14} className="animate-spin text-blue-400" />
                  ) : isSelected ? (
                    <span className="text-[10px] px-2 py-0.5 rounded-full bg-blue-500/30 text-blue-300 font-mono font-semibold border border-blue-400/40">
                      Đang xem
                    </span>
                  ) : (
                    <span className="text-[10px] text-cockpit-500 hover:text-cockpit-300 font-mono">
                      Xem ngay &rarr;
                    </span>
                  )}
                </button>
              );
            })}
          </div>
        </div>
      </div>

      {/* Footer hướng dẫn */}
      <div className="mt-4 pt-3 border-t border-cockpit-700 text-[10px] text-cockpit-400 space-y-1">
        <div className="text-cockpit-300 font-semibold">Lệnh phát xe khác trên Terminal:</div>
        <code className="block bg-cockpit-950 p-1.5 rounded border border-cockpit-800 text-[10px] font-mono text-fuel-filter">
          python simulate_live_car.py --car 2
        </code>
      </div>
    </aside>
  );
}
