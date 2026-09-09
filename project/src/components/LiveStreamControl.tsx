import { useState, useEffect } from "react";
import { Activity, Wifi, Radio, Cpu, CheckCircle2, ShieldCheck, MapPin, Gauge, Loader2, ListOrdered } from "lucide-react";
import type { TripPoint } from "@/simulation/types";

interface Props {
  current: TripPoint | null;
  pointsCount: number;
  onSwitch?: () => void;
  selectedVehicleId?: string;
  onSelectVehicle?: (vid: string) => void;
}

const DEFAULT_VEHICLES = [
  { id: "21H-03221", name: "Xe bồn 21H-03221", tag: "Lào Cai (850L)", plate: "21H-03221" },
  { id: "24H-04650", name: "Xe tải 24H-04650", tag: "TienXuLy (500L)", plate: "24H-04650" },
  { id: "29E-45520", name: "Xe tải 29E-45520", tag: "TienXuLy (200L)", plate: "29E-45520" },
  { id: "90H-03494", name: "Xe tải 90H-03494", tag: "TienXuLy (250L)", plate: "90H-03494" },
];

export default function LiveStreamControl({ current, pointsCount, onSwitch, selectedVehicleId, onSelectVehicle }: Props) {
  const [loadingCar, setLoadingCar] = useState<string | null>(null);
  const [fleetStatus, setFleetStatus] = useState<Record<string, any>>({});

  const vehicleId = selectedVehicleId || current?.vehicleId || "21H-03221";
  const aiState = current?.aiState || "STABLE_JITTER";

  // Polling trạng thái hàng đợi toàn đội xe
  useEffect(() => {
    const fetchFleet = async () => {
      try {
        const res = await fetch("http://localhost:8000/api/fleet/status");
        if (res.ok) {
          const data = await res.json();
          const map: Record<string, any> = {};
          data.forEach((item: any) => {
            map[item.vehicle_id] = item;
          });
          setFleetStatus(map);
        }
      } catch (e) {
        // server offline
      }
    };
    fetchFleet();
    const interval = setInterval(fetchFleet, 2000);
    return () => clearInterval(interval);
  }, []);

  const handleSwitchCar = async (carId: string) => {
    setLoadingCar(carId);
    onSelectVehicle?.(carId);
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

  const activeStat = fleetStatus[vehicleId];
  const queueSize = activeStat?.queue_size ?? 0;
  const processedTotal = activeStat?.total_processed ?? pointsCount;

  return (
    <aside className="w-84 shrink-0 bg-cockpit-850 border border-cockpit-700 rounded-2xl p-4 flex flex-col justify-between shadow-2xl">
      <div className="space-y-4">
        {/* Header */}
        <div className="flex items-center justify-between border-b border-cockpit-700 pb-3">
          <div className="flex items-center gap-2">
            <div className="w-8 h-8 rounded-lg bg-red-500/10 border border-red-500/30 flex items-center justify-center">
              <Wifi size={16} className="text-red-400 animate-pulse" />
            </div>
            <div>
              <h2 className="text-xs font-bold uppercase tracking-wider text-cockpit-100">Live Telemetry</h2>
              <p className="text-[10px] text-cockpit-400">Hàng chờ & Xử lý Đa xe</p>
            </div>
          </div>
          <span className="flex items-center gap-1.5 px-2 py-0.5 rounded-full bg-emerald-500/20 text-emerald-400 text-[10px] font-mono font-semibold border border-emerald-500/30">
            <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-ping" />
            Queue Ingest
          </span>
        </div>

        {/* Thông tin xe đang stream & Hàng chờ */}
        <div className="bg-cockpit-900 border border-cockpit-750 rounded-xl p-3 space-y-2">
          <div className="flex items-center justify-between">
            <span className="text-[10px] uppercase tracking-widest text-cockpit-400 font-semibold">Xe đang chọn</span>
            <span className="text-[10px] px-2 py-0.5 rounded bg-blue-500/20 text-blue-400 border border-blue-500/30 font-medium">
              Độc lập luồng
            </span>
          </div>
          <div className="flex items-center justify-between">
            <span className="text-base font-mono font-bold text-cockpit-100 tracking-wide">{vehicleId}</span>
            <div className="text-right">
              <span className="text-[11px] font-mono font-bold text-amber-400 bg-amber-500/10 px-2 py-0.5 rounded border border-amber-500/30">
                Queue: {queueSize}
              </span>
            </div>
          </div>
          <div className="text-[11px] text-cockpit-400 flex items-center justify-between pt-1 border-t border-cockpit-800">
            <span className="flex items-center gap-1.5">
              <Activity size={12} className="text-fuel-filter" />
              Đã xử lý:
            </span>
            <span className="text-cockpit-100 font-mono font-bold">{processedTotal} điểm đo</span>
          </div>
        </div>

        {/* Trạng thái xử lý của AI Model */}
        <div className="bg-cockpit-900 border border-cockpit-750 rounded-xl p-3 space-y-2">
          <div className="flex items-center justify-between">
            <span className="text-[10px] uppercase tracking-widest text-cockpit-400 font-semibold flex items-center gap-1.5">
              <Cpu size={12} className="text-[#ab63fa]" />
              AI Smooth-Tracking (Màu tím)
            </span>
            <span className="text-[10px] text-[#ab63fa] font-mono font-semibold">Realtime Flow</span>
          </div>

          <div className="p-2.5 rounded-lg bg-cockpit-950 border border-cockpit-800 space-y-1.5">
            <div className="text-[10px] text-cockpit-400">Trạng thái tín hiệu tức thời:</div>
            <div className="text-xs font-mono font-bold text-[#ab63fa] tracking-wide">
              {aiState}
            </div>
            <div className="flex items-center justify-between text-[10px] text-cockpit-400 pt-1">
              <span>Độ trễ xử lý:</span>
              <span className="text-emerald-400 font-mono font-semibold">&lt; 1ms / điểm</span>
            </div>
          </div>
        </div>

        {/* Danh sách các xe có thể bấm chọn */}
        <div className="space-y-2">
          <div className="flex items-center justify-between text-[10px] uppercase tracking-widest text-cockpit-400 font-semibold">
            <span>Chọn xe theo dõi</span>
            <span className="text-[9px] text-fuel-filter lowercase font-mono">1-click switch</span>
          </div>
          <div className="space-y-2">
            {DEFAULT_VEHICLES.map((car) => {
              const isSelected = vehicleId.includes(car.id);
              const isLoadingThis = loadingCar === car.id;
              const carStat = fleetStatus[car.id];
              const carQueue = carStat?.queue_size ?? 0;
              const carProcessed = carStat?.total_processed;

              return (
                <button
                  key={car.id}
                  onClick={() => handleSwitchCar(car.id)}
                  disabled={loadingCar !== null}
                  className={`w-full text-left flex items-center justify-between p-2.5 rounded-xl border text-xs transition-all cursor-pointer shadow-sm ${
                    isSelected
                      ? "bg-blue-500/20 border-blue-500 text-blue-300 shadow-md ring-1 ring-blue-500/40"
                      : "bg-cockpit-900 border-cockpit-800 text-cockpit-400 hover:border-cockpit-600 hover:bg-cockpit-800/80 hover:text-cockpit-100"
                  }`}
                >
                  <div className="flex items-center gap-2.5">
                    <span className="text-base">🚗</span>
                    <div>
                      <div className={`font-medium ${isSelected ? "text-cockpit-100 dark:text-white font-bold" : ""}`}>{car.name}</div>
                      <div className="text-[9px] text-cockpit-500 flex items-center gap-1.5">
                        <span>{car.tag}</span>
                        {carProcessed !== undefined && (
                          <span className="text-emerald-400 font-mono">• {carProcessed} pts</span>
                        )}
                      </div>
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
                      {carQueue > 0 ? `Q:${carQueue}` : "Xem →"}
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
        <div className="text-cockpit-300 font-semibold">Lệnh phát xe 21H-03221 trên Terminal:</div>
        <code className="block bg-cockpit-950 p-1.5 rounded border border-cockpit-800 text-[10px] font-mono text-fuel-filter">
          python simulate_live_car.py --car 1
        </code>
      </div>
    </aside>
  );
}
