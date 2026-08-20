import { useEffect, useState } from "react";
import { Moon, Radio, Sun, Activity, History, Wifi, Database } from "lucide-react";
import RemoteControl from "@/components/RemoteControl";
import Dashboard from "@/components/Dashboard";
import RollingChart from "@/components/RollingChart";
import EventLog from "@/components/EventLog";
import HistoryView from "@/components/HistoryView";
import { findTrip, DEFAULT_TRIP_ID } from "@/simulation/trips";
import { useSimulation } from "@/simulation/useSimulation";
import { useRealtimeData } from "@/simulation/useRealtimeData";
import type { Trip } from "@/simulation/types";

export default function App() {
  const [tripId, setTripId] = useState(DEFAULT_TRIP_ID);
  const [darkMode, setDarkMode] = useState(false);
  const [view, setView] = useState<"dashboard" | "history">("dashboard");
  const [dataSource, setDataSource] = useState<"simulation" | "realtime">("realtime");
  
  const trip = findTrip(tripId) ?? null;

  const simLocal = useSimulation(trip);
  const simLive = useRealtimeData();
  
  const sim = dataSource === "simulation" ? simLocal : simLive;

  const handleTrip = (t: Trip) => setTripId(t.id);

  useEffect(() => {
    document.documentElement.classList.toggle("dark", darkMode);
  }, [darkMode]);

  useEffect(() => {
    if (dataSource === "simulation") {
      simLocal.seek(0);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [tripId, dataSource]);

  return (
    <div className="min-h-screen flex flex-col bg-cockpit-950 text-cockpit-100 transition-colors duration-300">
      <header className="flex items-center justify-between px-6 py-3 border-b border-cockpit-800 bg-cockpit-900/80 backdrop-blur transition-colors duration-300">
        <div className="flex items-center gap-6">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-xl bg-cockpit-800 border border-cockpit-700 flex items-center justify-center">
              <Radio size={20} className="text-fuel-filter" />
            </div>
            <div>
              <h1 className="text-base font-semibold text-cockpit-100 leading-tight">
                Hệ Thống Giám Sát Nhiên Liệu
              </h1>
              <p className="text-[11px] text-cockpit-400">Trình diễn thuật toán lọc nhiễu cảm biến xăng</p>
            </div>
          </div>
          
          {/* Main Navigation Menu */}
          <nav className="hidden md:flex bg-cockpit-950 rounded-lg p-1 border border-cockpit-800">
            <button 
              onClick={() => setView("dashboard")}
              className={`flex items-center gap-2 px-4 py-1.5 rounded-md text-sm font-medium transition-colors ${view === "dashboard" ? "bg-cockpit-800 text-fuel-filter" : "text-cockpit-400 hover:text-cockpit-100 hover:bg-cockpit-800/50"}`}
            >
              <Activity size={16} /> Dashboard
            </button>
            <button 
              onClick={() => setView("history")}
              className={`flex items-center gap-2 px-4 py-1.5 rounded-md text-sm font-medium transition-colors ${view === "history" ? "bg-cockpit-800 text-fuel-filter" : "text-cockpit-400 hover:text-cockpit-100 hover:bg-cockpit-800/50"}`}
            >
              <History size={16} /> Lịch sử & Báo cáo
            </button>
          </nav>
        </div>

        <div className="flex items-center gap-4">
          {view === "dashboard" && (
            <div className="flex bg-cockpit-950 rounded-lg p-1 border border-cockpit-800 mr-2">
              <button 
                onClick={() => setDataSource("realtime")}
                className={`flex items-center gap-1.5 px-3 py-1 rounded text-xs font-medium transition-colors ${dataSource === "realtime" ? "bg-fuel-filter/20 text-fuel-filter" : "text-cockpit-400 hover:text-cockpit-100"}`}
              >
                <Wifi size={14} /> Live Stream
              </button>
              <button 
                onClick={() => setDataSource("simulation")}
                className={`flex items-center gap-1.5 px-3 py-1 rounded text-xs font-medium transition-colors ${dataSource === "simulation" ? "bg-cockpit-800 text-white" : "text-cockpit-400 hover:text-cockpit-100"}`}
              >
                <Database size={14} /> Dữ liệu mẫu
              </button>
            </div>
          )}

          <div className="hidden sm:flex items-center gap-2 text-[11px] text-cockpit-400">
            <span className={`w-2 h-2 rounded-full animate-pulse ${dataSource === 'realtime' ? 'bg-red-500' : 'bg-fuel-filter'}`} />
            <span>{dataSource === 'realtime' ? 'Đang nhận dữ liệu' : 'Sẵn sàng trình diễn'}</span>
          </div>
          
          <button
            type="button"
            onClick={() => setDarkMode((value) => !value)}
            aria-label={darkMode ? "Chuyển sang giao diện sáng" : "Chuyển sang giao diện tối"}
            className="w-9 h-9 rounded-lg border border-cockpit-700 bg-cockpit-800 text-cockpit-300 hover:text-cockpit-100 hover:bg-cockpit-700 transition-colors flex items-center justify-center"
          >
            {darkMode ? <Sun size={17} /> : <Moon size={17} />}
          </button>
        </div>
      </header>

      {view === "dashboard" ? (
        <>
          <div className="px-4 pt-4">
            <Dashboard current={sim.current} playing={sim.playing} />
          </div>

          <div className="flex gap-4 px-4 pb-4">
            <RollingChart
              trip={trip!}
              points={sim.window}
              cursor={sim.cursor}
              windowSec={sim.windowSec}
              playing={sim.playing}
            />
            {dataSource === "simulation" && (
              <RemoteControl
                trip={trip!}
                playing={sim.playing}
                speed={sim.speed}
                cursor={sim.cursor}
                duration={sim.duration}
                onToggle={sim.toggle}
                onSpeed={sim.setSpeed}
                onSeek={sim.seek}
                onReset={sim.reset}
                onTrip={handleTrip}
              />
            )}
          </div>

          <div className="px-4 pb-4">
            <EventLog points={sim.events} trip={trip} />
          </div>
        </>
      ) : (
        <div className="p-4 flex-1">
          <HistoryView />
        </div>
      )}
    </div>
  );
}
