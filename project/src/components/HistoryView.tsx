import { useState, useEffect } from "react";
import { Download, Search, MapPin, Gauge } from "lucide-react";
import { AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, ReferenceLine } from "recharts";
import MapView from "./MapView";
import { Map, BarChart2 } from "lucide-react";
interface HistoryRecord {
  timestamp: string;
  raw_fuel: number;
  kalman: number;
  adaptive: number;
  ai_enhanced: number;
  speed?: number;
  lat?: number;
  lng?: number;
  address?: string;
}

interface VehicleItem {
  id: string;
  name: string;
  start_date?: string;
  end_date?: string;
  total_points?: number;
}

const getClampedEndDate = (startStr: string, endStr: string): string => {
  if (!startStr || !endStr) return endStr;
  const s = new Date(startStr);
  const e = new Date(endStr);
  const diffDays = (e.getTime() - s.getTime()) / (1000 * 3600 * 24);
  if (diffDays > 10) {
    const clamped = new Date(s.getTime() + 10 * 24 * 3600 * 1000);
    return clamped.toISOString().split("T")[0];
  }
  return endStr;
};

export default function HistoryView() {
  const [selectedVehicle, setSelectedVehicle] = useState("21H-03221");
  const [availableVehicles, setAvailableVehicles] = useState<VehicleItem[]>([
    { id: "21H-03221", name: "Xe 21H-03221", start_date: "2026-08-10", end_date: "2026-08-18" },
    { id: "24H-04650", name: "Xe 24H-04650", start_date: "2026-08-12", end_date: "2026-08-18" },
    { id: "29E-45520", name: "Xe 29E-45520", start_date: "2026-08-10", end_date: "2026-08-18" },
    { id: "90H-03494", name: "Xe 90H-03494", start_date: "2026-08-10", end_date: "2026-08-18" },
  ]);
  const [activeStreamingVehicle, setActiveStreamingVehicle] = useState<string | null>(null);
  const [startDate, setStartDate] = useState("2026-08-10");
  const [endDate, setEndDate] = useState("2026-08-18");
  const [data, setData] = useState<HistoryRecord[]>([]);
  const [hasSearched, setHasSearched] = useState(false);
  const [showOnlySpikes, setShowOnlySpikes] = useState(false);
  const [viewMode, setViewMode] = useState<'chart' | 'map'>('chart');
  const [hoveredTime, setHoveredTime] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  // 1. Tự động kiểm tra xe đang nhận telemetry từ server và danh sách xe kèm khoảng ngày
  useEffect(() => {
    fetch("http://localhost:8000/api/vehicles")
      .then((r) => r.json())
      .then((res: VehicleItem[]) => {
        if (Array.isArray(res) && res.length > 0) {
          setAvailableVehicles(res);
          // Tìm xe đang chọn hoặc xe 21H-03221
          const curr = res.find((v) => v.id === selectedVehicle) || res.find((v) => v.id === "21H-03221") || res[0];
          if (curr && curr.start_date && curr.end_date) {
            setStartDate(curr.start_date);
            setEndDate(getClampedEndDate(curr.start_date, curr.end_date));
            setSelectedVehicle(curr.id);
            // Không tự động fetch dữ liệu ở đây, chỉ hiển thị khi người dùng bấm Tra cứu
          }
        }
      })
      .catch(() => {});

    fetch("http://localhost:8000/api/current-vehicle")
      .then((r) => r.json())
      .then((res) => {
        if (res && res.vehicle_id) {
          setActiveStreamingVehicle(res.vehicle_id);
        }
      })
      .catch(() => {});
  }, []);

  const handleSearch = async (targetCar?: string, customStart?: string, customEnd?: string) => {
    const carToSearch = targetCar || selectedVehicle;
    const st = customStart !== undefined ? customStart : startDate;
    const et = customEnd !== undefined ? customEnd : endDate;

    if (!st || !et) {
      setError("Vui lòng chọn ngày bắt đầu và kết thúc.");
      return;
    }

    const dStart = new Date(st);
    const dEnd = new Date(et);
    const diffDays = (dEnd.getTime() - dStart.getTime()) / (1000 * 3600 * 24);

    if (diffDays < 0) {
      setError("Ngày bắt đầu không được lớn hơn ngày kết thúc.");
      return;
    }

    if (diffDays > 10) {
      setError(`Khoảng thời gian tra cứu tối đa là 10 ngày (bạn đang chọn ${Math.ceil(diffDays)} ngày). Vui lòng chọn lại.`);
      return;
    }

    setLoading(true);
    setError("");
    setHasSearched(true);
    try {
      const response = await fetch(`http://localhost:8000/api/history?vehicle_id=${carToSearch}&start_date=${st}&end_date=${et}&limit=10000`);
      if (!response.ok) throw new Error("Failed to fetch data");
      const result = await response.json();
      
      const normalized = (result || []).map((d: any) => ({
        ...d,
        timestamp: d.timestamp || d.time || "",
        time: d.time || d.timestamp || "",
        raw_fuel: Number(d.raw_fuel !== undefined ? d.raw_fuel : (d.raw !== undefined ? d.raw : 0.0)),
        raw: Number(d.raw !== undefined ? d.raw : (d.raw_fuel !== undefined ? d.raw_fuel : 0.0)),
        ai_enhanced: Number(d.ai_enhanced !== undefined ? d.ai_enhanced : (d.clean !== undefined ? d.clean : 0.0)),
        clean: Number(d.clean !== undefined ? d.clean : (d.ai_enhanced !== undefined ? d.ai_enhanced : 0.0)),
        speed: Number(d.speed || 0),
        lat: Number(d.lat || 0),
        lng: Number(d.lng || 0),
        address: d.address || ""
      }));
      setData(normalized);
    } catch (err: any) {
      setError(err.message || "Lỗi khi kết nối tới máy chủ.");
    } finally {
      setLoading(false);
    }
  };

  const handleSelectCar = (carId: string) => {
    setSelectedVehicle(carId);
    setData([]); // Xóa sạch dữ liệu, không hiện cho tới khi bấm Tra cứu
    setHasSearched(false);
    setError("");
    const vInfo = availableVehicles.find((v) => v.id === carId);
    if (vInfo && vInfo.start_date && vInfo.end_date) {
      setStartDate(vInfo.start_date);
      setEndDate(getClampedEndDate(vInfo.start_date, vInfo.end_date));
    }
  };

  const handleExport = () => {
    if (data.length === 0) {
      setError("Vui lòng bấm 'Tra cứu' dữ liệu trước khi xuất file Excel.");
      return;
    }
    window.location.href = `http://localhost:8000/api/export_history?vehicle_id=${selectedVehicle}&start_date=${startDate}&end_date=${endDate}`;
  };

  return (
    <div className="bg-cockpit-900 rounded-2xl border border-cockpit-800 p-6 shadow-xl flex flex-col h-[calc(100vh-100px)]">
      <div className="flex items-center justify-between mb-6">
        <div>
          <h2 className="text-xl font-bold text-cockpit-100">Tra cứu Lịch sử & Báo cáo</h2>
          <p className="text-sm text-cockpit-400 mt-1">Xem trước dữ liệu đo đạc và xuất ra file Excel</p>
        </div>
      </div>

      {/* Filter Bar */}
      <div className="flex flex-wrap items-center gap-4 mb-6 bg-cockpit-950 p-4 rounded-xl border border-cockpit-800">
        <div className="flex flex-col gap-1 min-w-[280px]">
          <div className="flex items-center justify-between">
            <label className="text-xs font-medium text-cockpit-400 uppercase tracking-wider">Chọn xe</label>
            {activeStreamingVehicle && selectedVehicle !== activeStreamingVehicle && (
              <button
                onClick={() => handleSelectCar(activeStreamingVehicle)}
                className="text-[10px] text-blue-400 hover:text-blue-300 font-semibold flex items-center gap-1"
                title="Bấm để chuyển nhanh sang xe đang nhận dữ liệu"
              >
                ⚡ Xem xe đang nhận
              </button>
            )}
          </div>
          <select
            value={selectedVehicle}
            onChange={(e) => handleSelectCar(e.target.value)}
            className="bg-cockpit-800 border border-cockpit-700 text-cockpit-100 rounded-lg px-3 py-2 text-sm focus:outline-none focus:border-fuel-filter transition-colors font-semibold"
          >
            {availableVehicles.map((v) => (
              <option key={v.id} value={v.id}>
                🚗 {v.name} {v.start_date ? `(${v.start_date.slice(5)} ➔ ${v.end_date?.slice(5)})` : ""} {v.id === activeStreamingVehicle ? "🟢 (Đang nhận)" : ""}
              </option>
            ))}
          </select>
        </div>
        <div className="flex flex-col gap-1">
          <label className="text-xs font-medium text-cockpit-400 uppercase tracking-wider">Từ ngày</label>
          <input
            type="date"
            value={startDate}
            onChange={(e) => setStartDate(e.target.value)}
            className="bg-cockpit-800 border border-cockpit-700 text-cockpit-100 rounded-lg px-3 py-2 text-sm focus:outline-none focus:border-fuel-filter transition-colors"
          />
        </div>
        <div className="flex flex-col gap-1">
          <div className="flex items-center justify-between">
            <label className="text-xs font-medium text-cockpit-400 uppercase tracking-wider">Đến ngày</label>
            <span className="text-[10px] text-amber-400 font-medium ml-2">⚠️ Tối đa 10 ngày</span>
          </div>
          <input
            type="date"
            value={endDate}
            onChange={(e) => setEndDate(e.target.value)}
            className="bg-cockpit-800 border border-cockpit-700 text-cockpit-100 rounded-lg px-3 py-2 text-sm focus:outline-none focus:border-fuel-filter transition-colors"
          />
        </div>
        
        <div className="flex items-end h-full pt-5">
          <button
            onClick={() => handleSearch()}
            disabled={loading}
            className="flex items-center gap-2 bg-fuel-filter text-black px-5 py-2.5 rounded-lg text-sm font-semibold hover:bg-fuel-filter/90 transition-colors disabled:opacity-50 shadow-md"
          >
            <Search size={16} />
            {loading ? "Đang tải..." : "Tra cứu"}
          </button>
        </div>
        
        <div className="flex-1" />

        <label className="flex items-center gap-2 cursor-pointer mt-5 text-sm text-cockpit-300 hover:text-cockpit-100 transition-colors">
          <input 
            type="checkbox" 
            checked={showOnlySpikes}
            onChange={(e) => setShowOnlySpikes(e.target.checked)}
            className="accent-fuel-raw rounded cursor-pointer w-4 h-4"
          />
          Chỉ hiện Nhiễu/Bất thường (Spikes)
        </label>
        
        <div className="flex items-end h-full pt-5">
          <button
            onClick={handleExport}
            className="flex items-center gap-2 bg-cockpit-800 border border-cockpit-700 text-cockpit-100 px-5 py-2.5 rounded-lg text-sm font-semibold hover:bg-cockpit-700 transition-colors"
          >
            <Download size={16} />
            Xuất Excel
          </button>
        </div>
      </div>

      {/* Error Alert Banner */}
      {error && (
        <div className="mb-4 px-4 py-3 bg-red-500/10 border border-red-500/40 rounded-xl text-red-400 text-xs flex items-center gap-2 font-medium animate-fadeIn">
          <span className="text-sm">⚠️</span>
          <span>{error}</span>
        </div>
      )}

      {/* Data Table */}
      <div className="flex-[0.4] overflow-auto rounded-xl border border-cockpit-800 bg-cockpit-950 mb-4 min-h-[200px]">
        <table className="w-full text-left text-sm text-cockpit-300">
          <thead className="text-xs uppercase bg-cockpit-900 text-cockpit-400 sticky top-0 border-b border-cockpit-800 shadow-md">
            <tr>
              <th className="px-4 py-3 font-semibold">Thời gian</th>
              <th className="px-4 py-3 font-semibold text-right">Thô (L)</th>
              <th className="px-4 py-3 font-semibold text-right">AI Sạch (L)</th>
              <th className="px-4 py-3 font-semibold text-center"><Gauge size={14} className="inline mr-1"/>Km/h</th>
              <th className="px-4 py-3 font-semibold"><MapPin size={14} className="inline mr-1"/>Tọa độ GPS / Địa điểm</th>
            </tr>
          </thead>
          <tbody>
            {error ? (
              <tr>
                <td colSpan={5} className="px-6 py-8 text-center text-red-400 bg-red-400/10 font-medium">
                  {error}
                </td>
              </tr>
            ) : data.length === 0 ? (
              <tr>
                <td colSpan={5} className="px-6 py-12 text-center text-cockpit-500 italic">
                  {hasSearched ? "Không tìm thấy bản ghi nào trong khoảng ngày đã chọn." : "Dữ liệu chưa tải. Vui lòng bấm 'Tra cứu' để xem bảng dữ liệu."}
                </td>
              </tr>
            ) : (
              data.filter(row => showOnlySpikes ? Math.abs((row.raw_fuel ?? (row as any).raw ?? 0) - (row.ai_enhanced ?? (row as any).clean ?? 0)) > 4 : true).map((row: any, idx) => {
                const timeStr = row.timestamp || row.time || "N/A";
                const rawVal = row.raw_fuel !== undefined ? row.raw_fuel : (row.raw !== undefined ? row.raw : 0.0);
                const cleanVal = row.ai_enhanced !== undefined ? row.ai_enhanced : (row.clean !== undefined ? row.clean : 0.0);
                const speedVal = row.speed !== undefined ? row.speed : 0;
                const latVal = row.lat ? Number(row.lat).toFixed(4) : null;
                const lngVal = row.lng ? Number(row.lng).toFixed(4) : null;
                const coordStr = latVal && lngVal ? `(${latVal}, ${lngVal})` : "";
                const isHovered = Boolean(hoveredTime && (timeStr === hoveredTime || timeStr.startsWith(hoveredTime) || hoveredTime.startsWith(timeStr)));
                
                return (
                  <tr 
                    key={idx} 
                    onMouseEnter={() => setHoveredTime(timeStr)}
                    onMouseLeave={() => setHoveredTime(null)}
                    className={`border-b border-cockpit-800/50 transition-all duration-150 whitespace-nowrap cursor-pointer ${
                      isHovered 
                        ? 'bg-blue-600/40 text-white shadow-md ring-1 ring-blue-400 font-semibold' 
                        : 'hover:bg-cockpit-800/80 hover:text-cockpit-100'
                    }`}
                  >
                    <td className="px-4 py-2 font-medium text-cockpit-100 font-mono">{timeStr}</td>
                    <td className="px-4 py-2 text-right text-red-400 font-mono">{Number(rawVal).toFixed(1)}</td>
                    <td className="px-4 py-2 text-right text-purple-400 font-mono font-semibold">{Number(cleanVal).toFixed(1)}</td>
                    <td className="px-4 py-2 text-center text-blue-400 font-mono font-bold">{speedVal}</td>
                    <td className="px-4 py-2 text-cockpit-300 truncate max-w-[280px]" title={`${coordStr} ${row.address || ''}`}>
                      {coordStr && <span className="text-emerald-400 font-mono text-xs mr-2">{coordStr}</span>}
                      <span>{row.address || "N/A"}</span>
                    </td>
                  </tr>
                );
              })
            )}
          </tbody>
        </table>
      </div>
      
      {/* Chart / Map Bottom Half */}
      <div className="flex-[0.6] bg-cockpit-950 rounded-xl border border-cockpit-800 shadow-inner p-3 min-h-[300px] flex flex-col relative">
        <div className="flex items-center justify-between px-2 mb-2">
          <div className="flex items-center gap-4">
            <h3 className="text-cockpit-100 font-bold text-xs uppercase tracking-wider flex items-center gap-2">
              <BarChart2 size={15} className="text-fuel-filter" />
              {viewMode === 'chart' ? 'Biểu đồ nhiên liệu (Lít)' : 'Bản đồ hành trình'}
            </h3>
            {viewMode === 'chart' && (
              <div className="flex items-center gap-3 text-xs">
                <span className="flex items-center gap-1.5"><span className="w-3 h-0.5 bg-fuel-raw" /><span className="text-cockpit-400">Xăng gốc</span></span>
                <span className="flex items-center gap-1.5"><span className="w-3 h-0.5 bg-purple-400" /><span className="text-purple-400 font-semibold">AI-Kalman</span></span>
              </div>
            )}
          </div>
          <div className="flex bg-cockpit-900 rounded-lg p-1 border border-cockpit-800">
            <button
              onClick={() => setViewMode('chart')}
              className={`flex items-center gap-1.5 px-3 py-1 text-xs font-semibold rounded-md transition-colors ${viewMode === 'chart' ? 'bg-cockpit-800 text-fuel-filter shadow-sm' : 'text-cockpit-400 hover:text-cockpit-200'}`}
            >
              <BarChart2 size={14} /> Biểu đồ
            </button>
            <button
              onClick={() => setViewMode('map')}
              className={`flex items-center gap-1.5 px-3 py-1 text-xs font-semibold rounded-md transition-colors ${viewMode === 'map' ? 'bg-cockpit-800 text-fuel-filter shadow-sm' : 'text-cockpit-400 hover:text-cockpit-200'}`}
            >
              <Map size={14} /> Bản đồ
            </button>
          </div>
        </div>
        
        <div className="flex-1 w-full h-full">
          {data.length > 0 ? (
            viewMode === 'chart' ? (() => {
              const filteredData = data.filter(row => showOnlySpikes ? Math.abs(row.raw_fuel - row.ai_enhanced) > 4 : true);
              const fuelValues = filteredData.flatMap(d => [d.raw_fuel, d.ai_enhanced]).filter(v => v > 0);
              const minVal = fuelValues.length > 0 ? Math.min(...fuelValues) : 0;
              const maxVal = fuelValues.length > 0 ? Math.max(...fuelValues) : 100;
              const spread = maxVal - minVal;
              // Nếu dải biến thiên nhỏ hơn 10L thì mở rộng dải để biểu đồ không bị phóng to dị dạng
              const pad = spread < 10 ? (10 - spread) / 2 : Math.max(1, spread * 0.05);
              const yMin = Math.max(0, Math.floor(minVal - pad));
              const yMax = Math.ceil(maxVal + pad);
              const yDomain = fuelValues.length > 0 ? [yMin, yMax] : ['auto', 'auto'];

              const CustomTooltip = ({ active, payload }: any) => {
                if (active && payload && payload.length) {
                  const pt = payload[0].payload;
                  const timeStr = pt.timestamp || pt.time || "";
                  const speed = pt.speed !== undefined ? Number(pt.speed).toFixed(0) : 0;
                  const raw = Number(pt.raw_fuel !== undefined ? pt.raw_fuel : pt.raw || 0).toFixed(1);
                  const clean = Number(pt.ai_enhanced !== undefined ? pt.ai_enhanced : pt.clean || 0).toFixed(1);
                  const address = pt.address || "";

                  return (
                    <div className="bg-cockpit-900/95 backdrop-blur-md border border-cockpit-700 shadow-2xl rounded-xl p-3 text-xs text-cockpit-200 min-w-[210px]">
                      <div className="font-semibold text-cockpit-100 border-b border-cockpit-700 pb-1 mb-2 font-mono flex items-center justify-between">
                        <span>🕒 {timeStr}</span>
                      </div>
                      <div className="flex justify-between items-center py-1 bg-blue-950/40 px-2 rounded mb-1 border border-blue-800/30">
                        <span className="text-blue-400 font-semibold flex items-center gap-1.5">
                          <Gauge size={13} className="text-blue-400" /> Vận tốc:
                        </span>
                        <span className="font-mono font-bold text-blue-300 text-sm">{speed} km/h</span>
                      </div>
                      <div className="flex justify-between items-center py-0.5 px-1">
                        <span className="text-purple-400 font-medium">🟣 AI-Kalman:</span>
                        <span className="font-mono font-bold text-purple-300">{clean} L</span>
                      </div>
                      <div className="flex justify-between items-center py-0.5 px-1">
                        <span className="text-red-400 font-medium">🔴 Xăng gốc:</span>
                        <span className="font-mono font-bold text-red-300">{raw} L</span>
                      </div>
                      {address && (
                        <div className="mt-1.5 pt-1 border-t border-cockpit-800 text-[10px] text-cockpit-400 truncate max-w-[220px]">
                          📍 {address}
                        </div>
                      )}
                    </div>
                  );
                }
                return null;
              };

              return (
                <ResponsiveContainer width="100%" height="100%">
                  <AreaChart 
                    data={filteredData} 
                    margin={{ top: 10, right: 15, left: -10, bottom: 0 }}
                    onMouseMove={(e: any) => {
                      if (e && e.activePayload && e.activePayload[0]) {
                        const pt = e.activePayload[0].payload;
                        setHoveredTime(pt.timestamp || pt.time || null);
                      }
                    }}
                    onMouseLeave={() => setHoveredTime(null)}
                  >
                    <defs>
                      <linearGradient id="colorAI" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="5%" stopColor="#c084fc" stopOpacity={0.4}/>
                        <stop offset="95%" stopColor="#c084fc" stopOpacity={0.0}/>
                      </linearGradient>
                    </defs>
                    <CartesianGrid strokeDasharray="3 3" stroke="#2a303c" vertical={true} />
                    <XAxis 
                      dataKey="timestamp" 
                      tickFormatter={(val) => {
                        if (!val) return "";
                        const d = new Date(val);
                        if (isNaN(d.getTime())) return String(val).slice(-8);
                        return `${d.getHours().toString().padStart(2, '0')}:${d.getMinutes().toString().padStart(2, '0')}`;
                      }}
                      tick={{ fontSize: 10, fill: '#9ca3af' }}
                      stroke="#374151"
                      tickMargin={5}
                    />
                    <YAxis 
                      domain={yDomain} 
                      tick={{ fontSize: 10, fill: '#9ca3af' }}
                      stroke="#374151"
                      tickFormatter={(val) => `${Number(val).toFixed(1)} L`}
                    />
                    <Tooltip content={<CustomTooltip />} />
                    
                    {/* Đường mức xăng gốc bị nhiễu */}
                    <Area 
                      type="monotone" 
                      dataKey="raw_fuel" 
                      name="raw_fuel"
                      stroke="#ef4444" 
                      strokeWidth={1.5}
                      fillOpacity={0} 
                      dot={false}
                    />
                    {/* Đường mức xăng sạch sau lọc AI */}
                    <Area 
                      type="monotone" 
                      dataKey="ai_enhanced" 
                      name="ai_enhanced"
                      stroke="#c084fc" 
                      strokeWidth={2.5}
                      fillOpacity={1} 
                      fill="url(#colorAI)" 
                      dot={false}
                      activeDot={{ r: 4, fill: "#c084fc", stroke: "#fff", strokeWidth: 2 }}
                    />
                  </AreaChart>
                </ResponsiveContainer>
              );
            })() : (
              <MapView data={data} />
            )
          ) : (
            <div className="flex flex-col items-center justify-center h-full text-cockpit-500 text-sm italic">
              <BarChart2 size={32} className="mb-2 opacity-30 text-cockpit-400" />
              <span>{hasSearched ? "Không có dữ liệu biểu đồ trong khoảng ngày đã chọn." : "Bấm 'Tra cứu' để hiển thị biểu đồ và bản đồ chi tiết."}</span>
            </div>
          )}
        </div>
      </div>

      <div className="mt-3 flex justify-between items-center text-xs text-cockpit-500">
        <div>Dữ liệu xuất chuẩn hệ thống GPS/Tracking</div>
        <div>Hiển thị {data.filter(row => showOnlySpikes ? Math.abs(row.raw_fuel - row.ai_enhanced) > 5 : true).length} bản ghi</div>
      </div>
    </div>
  );
}
