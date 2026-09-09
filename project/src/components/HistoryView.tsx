import { useState, useEffect, useRef } from "react";
import { Download, Search, MapPin, Gauge, Map, BarChart2, Maximize2, Minimize2, RotateCcw, Zap, Trash2, Radio, RefreshCw } from "lucide-react";
import { AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Brush } from "recharts";
import MapView from "./MapView";

interface HistoryRecord {
  timestamp: string;
  time?: string;
  raw_fuel: number;
  raw?: number;
  clean?: number;
  kalman?: number;
  adaptive?: number;
  ai_enhanced?: number;
  ai_smooth_tracking?: number;
  smooth_tracking?: number;
  speed?: number;
  lat?: number;
  lng?: number;
  address?: string;
  state?: string;
  ai_state?: string;
}

interface VehicleItem {
  id: string;
  name: string;
  start_date?: string;
  end_date?: string;
  total_points?: number;
  is_streaming?: boolean;
}

export default function HistoryView() {
  const [selectedVehicle, setSelectedVehicle] = useState<string>("");
  const [availableVehicles, setAvailableVehicles] = useState<VehicleItem[]>([]);
  const [activeStreamingVehicle, setActiveStreamingVehicle] = useState<string | null>(null);
  const [startDate, setStartDate] = useState("");
  const [endDate, setEndDate] = useState("");
  const [data, setData] = useState<HistoryRecord[]>([]);
  const [displayLimit, setDisplayLimit] = useState(100);
  const [showOnlySpikes, setShowOnlySpikes] = useState(false);
  const [newestFirst, setNewestFirst] = useState(true);
  const [isLiveSync, setIsLiveSync] = useState(true);
  const [viewMode, setViewMode] = useState<'chart' | 'map'>('chart');
  const [hoveredTime, setHoveredTime] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [isExpanded, setIsExpanded] = useState(false);
  const [brushRange, setBrushRange] = useState<{ startIndex?: number; endIndex?: number }>({});
  
  const lastPointsCountRef = useRef(0);

  const zoomToRefuel = (records: HistoryRecord[]) => {
    if (records.length < 2) return;
    let maxJumpIdx = -1;
    let maxJump = 0;
    for (let i = 1; i < records.length; i++) {
      const jump = (records[i].raw_fuel ?? 0) - (records[i - 1].raw_fuel ?? 0);
      if (jump > maxJump) {
        maxJump = jump;
        maxJumpIdx = i;
      }
    }
    if (maxJumpIdx !== -1) {
      const start = Math.max(0, maxJumpIdx - 30);
      const end = Math.min(records.length - 1, maxJumpIdx + 50);
      setBrushRange({ startIndex: start, endIndex: end });
    }
  };

  const resetZoom = () => {
    setBrushRange({});
  };

  // 1. Định kỳ quét danh sách xe thực tế đang có luồng dữ liệu bắn trên máy chủ (mỗi 1.5 giây)
  useEffect(() => {
    const checkVehicles = async () => {
      try {
        const res = await fetch("http://localhost:8000/api/vehicles");
        if (!res.ok) return;
        const vList: VehicleItem[] = await res.json();
        if (Array.isArray(vList)) {
          setAvailableVehicles(vList);
          
          // Tự động chọn xe đang bắn nếu chưa chọn xe nào
          if (vList.length > 0) {
            setSelectedVehicle((curr) => {
              if (!curr || !vList.some((v) => v.id === curr)) {
                return vList[0].id;
              }
              return curr;
            });
          }
        }
      } catch (e) {
        // server offline
      }
    };

    checkVehicles();
    const timer = setInterval(checkVehicles, 1500);
    return () => clearInterval(timer);
  }, []);

  // 2. Theo dõi xe đang nhận tín hiệu thời gian thực
  useEffect(() => {
    const fetchCurrent = async () => {
      try {
        const r = await fetch("http://localhost:8000/api/current-vehicle");
        if (r.ok) {
          const res = await r.json();
          if (res && res.vehicle_id) {
            setActiveStreamingVehicle(res.vehicle_id);
          }
        }
      } catch (e) {}
    };
    fetchCurrent();
    const t = setInterval(fetchCurrent, 2000);
    return () => clearInterval(t);
  }, []);

  // Hàm tải dữ liệu của xe được chọn từ buffer server
  const fetchVehicleData = async (carId: string, isManual = false) => {
    if (!carId) {
      setData([]);
      return;
    }
    if (isManual) setLoading(true);
    try {
      let url = `http://localhost:8000/api/history?vehicle_id=${encodeURIComponent(carId)}`;
      if (!isLiveSync && startDate && endDate) {
        url += `&start_date=${startDate}&end_date=${endDate}`;
      }
      const response = await fetch(url);
      if (!response.ok) throw new Error("Không thể kết nối máy chủ");
      const result = await response.json();

      const normalized: HistoryRecord[] = (result || []).map((d: any) => ({
        ...d,
        timestamp: d.timestamp || d.time || "",
        time: d.time || d.timestamp || "",
        raw_fuel: Number(d.raw_fuel !== undefined ? d.raw_fuel : (d.raw !== undefined ? d.raw : 0.0)),
        raw: Number(d.raw !== undefined ? d.raw : (d.raw_fuel !== undefined ? d.raw_fuel : 0.0)),
        ai_enhanced: Number(d.clean !== undefined ? d.clean : (d.ai_smooth_tracking !== undefined ? d.ai_smooth_tracking : (d.ai_enhanced !== undefined ? d.ai_enhanced : 0.0))),
        clean: Number(d.clean !== undefined ? d.clean : (d.ai_smooth_tracking !== undefined ? d.ai_smooth_tracking : (d.ai_enhanced !== undefined ? d.ai_enhanced : 0.0))),
        ai_smooth_tracking: Number(d.ai_smooth_tracking !== undefined ? d.ai_smooth_tracking : (d.clean !== undefined ? d.clean : 0.0)),
        speed: Number(d.speed || 0),
        lat: Number(d.lat || 0),
        lng: Number(d.lng || 0),
        address: d.address || "",
        state: d.state || d.ai_state || "NORMAL",
      }));

      setData(normalized);
      lastPointsCountRef.current = normalized.length;
      setError("");
    } catch (err: any) {
      if (isManual) setError(err.message || "Lỗi khi tải dữ liệu.");
    } finally {
      if (isManual) setLoading(false);
    }
  };

  // 3. Realtime Live Sync: Khi bật tự động cập nhật, poll mỗi 1000ms để bắn điểm nào hiện điểm đó
  useEffect(() => {
    if (!selectedVehicle) {
      setData([]);
      return;
    }

    fetchVehicleData(selectedVehicle, false);

    if (!isLiveSync) return;

    const interval = setInterval(() => {
      fetchVehicleData(selectedVehicle, false);
    }, 1000);

    return () => clearInterval(interval);
  }, [selectedVehicle, isLiveSync]);

  const handleSelectCar = (carId: string) => {
    setSelectedVehicle(carId);
    setError("");
    setBrushRange({});
  };

  const handleResetVehicleStream = async () => {
    if (!selectedVehicle) return;
    try {
      await fetch(`http://localhost:8000/api/v1/vehicles/${selectedVehicle}/reset-state`, {
        method: "POST",
      });
      setData([]);
      setBrushRange({});
    } catch (e) {
      setError("Không thể reset xe.");
    }
  };

  const handleExport = () => {
    if (data.length === 0) {
      setError("Chưa có dữ liệu để xuất file Excel.");
      return;
    }
    window.location.href = `http://localhost:8000/api/export_history?vehicle_id=${selectedVehicle}&start_date=${startDate}&end_date=${endDate}`;
  };

  return (
    <div className="bg-cockpit-900 rounded-2xl border border-cockpit-800 p-6 shadow-xl flex flex-col h-[calc(100vh-100px)]">
      <div className="flex items-center justify-between mb-4">
        <div>
          <div className="flex items-center gap-3">
            <h2 className="text-xl font-bold text-cockpit-100">Tra cứu Lịch sử & Dữ liệu Realtime</h2>
            {data.length > 0 && isLiveSync && (
              <span className="flex items-center gap-1.5 px-2.5 py-1 rounded-full bg-emerald-500/15 border border-emerald-500/30 text-emerald-400 text-xs font-mono font-semibold animate-pulse">
                <span className="w-2 h-2 rounded-full bg-emerald-400" />
                Live: {data.length} điểm đã lọc
              </span>
            )}
            {data.length > 0 && !isLiveSync && (
              <span className="px-2.5 py-1 rounded-full bg-cockpit-800 border border-cockpit-700 text-cockpit-300 text-xs font-mono">
                Tạm dừng đồng bộ ({data.length} điểm)
              </span>
            )}
          </div>
          <p className="text-xs text-cockpit-400 mt-1">
            Hiển thị điểm đo thời gian thực: Xe bắn được điểm nào hệ thống sẽ lọc và cập nhật ngay điểm đó.
          </p>
        </div>

        {/* Nút thao tác nhanh */}
        <div className="flex items-center gap-2">
          {selectedVehicle && (
            <button
              onClick={handleResetVehicleStream}
              className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-red-500/10 border border-red-500/30 text-red-400 hover:bg-red-500/20 text-xs font-medium transition-colors"
              title="Xóa buffer của xe này trên máy chủ để bắt đầu lượt bắn mới"
            >
              <Trash2 size={14} /> Xóa luồng xe này
            </button>
          )}
          <button
            onClick={handleExport}
            disabled={data.length === 0}
            className="flex items-center gap-1.5 bg-cockpit-800 border border-cockpit-700 text-cockpit-100 px-3.5 py-1.5 rounded-lg text-xs font-semibold hover:bg-cockpit-700 transition-colors disabled:opacity-40 disabled:cursor-not-allowed"
          >
            <Download size={14} /> Xuất Excel ({data.length})
          </button>
        </div>
      </div>

      {/* Filter Bar */}
      <div className="flex flex-wrap items-center gap-4 mb-4 bg-cockpit-950 p-4 rounded-xl border border-cockpit-800">
        {/* Chọn xe */}
        <div className="flex flex-col gap-1 min-w-[260px]">
          <div className="flex items-center justify-between">
            <label className="text-xs font-medium text-cockpit-400 uppercase tracking-wider">Xe đang bắn dữ liệu</label>
            {activeStreamingVehicle && selectedVehicle !== activeStreamingVehicle && (
              <button
                onClick={() => handleSelectCar(activeStreamingVehicle)}
                className="text-[10px] text-blue-400 hover:text-blue-300 font-semibold flex items-center gap-1"
                title="Bấm để chuyển nhanh sang xe đang nhận dữ liệu"
              >
                ⚡ Xem xe đang bắn
              </button>
            )}
          </div>
          <select
            value={selectedVehicle}
            onChange={(e) => handleSelectCar(e.target.value)}
            disabled={availableVehicles.length === 0}
            className="bg-cockpit-800 border border-cockpit-700 text-cockpit-100 rounded-lg px-3 py-2 text-sm focus:outline-none focus:border-fuel-filter transition-colors font-semibold disabled:opacity-50"
          >
            {availableVehicles.length === 0 ? (
              <option value="">-- Chưa có xe nào phát dữ liệu --</option>
            ) : (
              availableVehicles.map((v) => (
                <option key={v.id} value={v.id}>
                  🚗 {v.name} ({v.total_points ?? 0} điểm đã nhận) {v.id === activeStreamingVehicle ? "🟢" : ""}
                </option>
              ))
            )}
          </select>
        </div>

        {/* Chế độ Realtime Sync */}
        <div className="flex items-center gap-2 pt-5">
          <label className="flex items-center gap-2 cursor-pointer text-xs font-medium bg-cockpit-900 border border-cockpit-750 px-3 py-2 rounded-lg text-cockpit-200 hover:text-white transition-colors">
            <input
              type="checkbox"
              checked={isLiveSync}
              onChange={(e) => setIsLiveSync(e.target.checked)}
              className="accent-emerald-500 rounded cursor-pointer w-4 h-4"
            />
            <span>🟢 Nhận Realtime tự động</span>
          </label>
        </div>

        {/* Lọc khoảng ngày (chỉ khi tắt Live Sync) */}
        {!isLiveSync && (
          <>
            <div className="flex flex-col gap-1">
              <label className="text-xs font-medium text-cockpit-400 uppercase tracking-wider">Từ ngày</label>
              <input
                type="date"
                value={startDate}
                onChange={(e) => setStartDate(e.target.value)}
                className="bg-cockpit-800 border border-cockpit-700 text-cockpit-100 rounded-lg px-2.5 py-1.5 text-xs focus:outline-none focus:border-fuel-filter transition-colors"
              />
            </div>
            <div className="flex flex-col gap-1">
              <label className="text-xs font-medium text-cockpit-400 uppercase tracking-wider">Đến ngày</label>
              <input
                type="date"
                value={endDate}
                onChange={(e) => setEndDate(e.target.value)}
                className="bg-cockpit-800 border border-cockpit-700 text-cockpit-100 rounded-lg px-2.5 py-1.5 text-xs focus:outline-none focus:border-fuel-filter transition-colors"
              />
            </div>
            <div className="flex items-end h-full pt-5">
              <button
                onClick={() => fetchVehicleData(selectedVehicle, true)}
                disabled={loading || !selectedVehicle}
                className="flex items-center gap-1.5 bg-fuel-filter text-black px-4 py-2 rounded-lg text-xs font-semibold hover:bg-fuel-filter/90 transition-colors disabled:opacity-50 shadow-md"
              >
                <Search size={14} />
                {loading ? "Đang tải..." : "Tra cứu"}
              </button>
            </div>
          </>
        )}

        <div className="flex-1" />

        {/* Tùy chọn bảng */}
        <div className="flex items-center gap-4 pt-5">
          <label className="flex items-center gap-2 cursor-pointer text-xs text-cockpit-300 hover:text-cockpit-100 transition-colors">
            <input
              type="checkbox"
              checked={newestFirst}
              onChange={(e) => setNewestFirst(e.target.checked)}
              className="accent-blue-500 rounded cursor-pointer w-4 h-4"
            />
            <span>Mới nhất lên đầu</span>
          </label>

          <label className="flex items-center gap-2 cursor-pointer text-xs text-cockpit-300 hover:text-cockpit-100 transition-colors">
            <input
              type="checkbox"
              checked={showOnlySpikes}
              onChange={(e) => setShowOnlySpikes(e.target.checked)}
              className="accent-fuel-raw rounded cursor-pointer w-4 h-4"
            />
            <span>Chỉ hiện Nhiễu (Spikes)</span>
          </label>
        </div>
      </div>

      {/* Error Alert Banner */}
      {error && (
        <div className="mb-4 px-4 py-3 bg-red-500/10 border border-red-500/40 rounded-xl text-red-400 text-xs flex items-center gap-2 font-medium animate-fadeIn">
          <span className="text-sm">⚠️</span>
          <span>{error}</span>
        </div>
      )}

      {/* Empty State: Khi chưa có xe nào phát dữ liệu */}
      {availableVehicles.length === 0 ? (
        <div className="flex-1 flex flex-col items-center justify-center p-8 bg-cockpit-950 rounded-xl border border-dashed border-cockpit-800 text-center my-2">
          <div className="w-16 h-16 rounded-2xl bg-blue-500/10 border border-blue-500/30 flex items-center justify-center text-blue-400 mb-4 animate-pulse shadow-lg">
            <Radio size={32} />
          </div>
          <h3 className="text-base font-bold text-cockpit-100 mb-2">Chưa có luồng dữ liệu xe nào đang bắn</h3>
          <p className="text-xs text-cockpit-400 max-w-lg mb-5 leading-relaxed">
            Hệ thống hiển thị thuần túy dữ liệu thời gian thực (Realtime). Xe nào được bắn qua script mô phỏng thì máy chủ sẽ nhận, lọc AI và biểu diễn điểm đó ngay lập tức, không nạp trước dữ liệu tĩnh.
          </p>
          <div className="flex items-center gap-3 bg-cockpit-900 border border-cockpit-750 px-4 py-2.5 rounded-xl text-xs font-mono text-emerald-400 shadow-inner">
            <span className="text-cockpit-500">$</span>
            <span className="font-semibold">python simulate_live_car.py --car 1</span>
          </div>
          <p className="text-[11px] text-cockpit-500 mt-3">
            Hỗ trợ: <code className="text-cockpit-300">--car 1</code> (24H-04650), <code className="text-cockpit-300">--car 2</code> (29E-45520), <code className="text-cockpit-300">--car 3</code> (90H-03494)
          </p>
        </div>
      ) : (
        <>
          {/* Data Table */}
          <div className="flex-[0.4] overflow-auto rounded-xl border border-cockpit-800 bg-cockpit-950 mb-4 min-h-[190px]">
            <table className="w-full text-left text-sm text-cockpit-300">
              <thead className="text-xs uppercase bg-cockpit-900 text-cockpit-400 sticky top-0 border-b border-cockpit-800 shadow-md">
                <tr>
                  <th className="px-4 py-3 font-semibold">Thời gian</th>
                  <th className="px-4 py-3 font-semibold text-right">Thô (L)</th>
                  <th className="px-4 py-3 font-semibold text-right">Lọc AI (L)</th>
                  <th className="px-4 py-3 font-semibold text-center"><Gauge size={14} className="inline mr-1"/>Km/h</th>
                  <th className="px-4 py-3 font-semibold"><MapPin size={14} className="inline mr-1"/>Tọa độ GPS / Địa điểm</th>
                </tr>
              </thead>
              <tbody>
                {data.length === 0 ? (
                  <tr>
                    <td colSpan={5} className="px-6 py-12 text-center text-cockpit-500 italic">
                      Đang chờ nhận điểm telemetry từ xe {selectedVehicle}...
                    </td>
                  </tr>
                ) : (() => {
                  let processedRows = [...data];
                  if (showOnlySpikes) {
                    processedRows = processedRows.filter((r) => Math.abs((r.raw_fuel ?? 0) - (r.clean ?? 0)) > 4.0);
                  }
                  if (newestFirst) {
                    processedRows.reverse();
                  }

                  const visibleRows = processedRows.slice(0, displayLimit);

                  return (
                    <>
                      {visibleRows.map((row: any, idx) => {
                        const timeStr = row.timestamp || row.time || "N/A";
                        const rawVal = row.raw_fuel !== undefined ? row.raw_fuel : (row.raw !== undefined ? row.raw : 0.0);
                        const cleanVal = row.clean !== undefined ? row.clean : (row.ai_smooth_tracking !== undefined ? row.ai_smooth_tracking : (row.ai_enhanced !== undefined ? row.ai_enhanced : 0.0));
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
                      })}
                      {processedRows.length > displayLimit && (
                        <tr>
                          <td colSpan={5} className="px-4 py-3 text-center bg-cockpit-900/80 border-t border-cockpit-800">
                            <button
                              onClick={() => setDisplayLimit((prev) => prev + 200)}
                              className="px-4 py-1.5 rounded-lg bg-cockpit-800 hover:bg-cockpit-700 text-fuel-filter text-xs font-semibold transition-colors mr-3 border border-cockpit-700"
                            >
                              ➕ Tải thêm 200 dòng (Đang xem {displayLimit} / {processedRows.length})
                            </button>
                            <button
                              onClick={() => setDisplayLimit(processedRows.length)}
                              className="px-3 py-1.5 rounded-lg bg-cockpit-800 hover:bg-cockpit-700 text-cockpit-400 hover:text-white text-xs transition-colors border border-cockpit-700"
                            >
                              Xem tất cả
                            </button>
                          </td>
                        </tr>
                      )}
                    </>
                  );
                })()}
              </tbody>
            </table>
          </div>
          
          {/* Chart / Map Bottom Half */}
          <div className={`transition-all duration-300 ${
            isExpanded 
              ? "fixed inset-4 z-50 bg-cockpit-950/98 backdrop-blur-xl rounded-2xl border-2 border-[#ab63fa]/50 shadow-2xl p-5 flex flex-col"
              : "flex-[0.6] bg-cockpit-950 rounded-xl border border-cockpit-800 shadow-inner p-3 min-h-[350px] flex flex-col relative"
          }`}>
            <div className="flex items-center justify-between px-2 mb-2 flex-wrap gap-2">
              <div className="flex items-center gap-4">
                <h3 className="text-cockpit-100 font-bold text-xs uppercase tracking-wider flex items-center gap-2">
                  <BarChart2 size={15} className="text-fuel-filter" />
                  {viewMode === 'chart' ? 'Biểu đồ nhiên liệu theo thời gian thực (Lít)' : 'Bản đồ hành trình'}
                </h3>
                {viewMode === 'chart' && (
                  <div className="flex items-center gap-4 text-xs">
                    <span className="flex items-center gap-1.5"><span className="w-3.5 h-1 bg-fuel-raw rounded-full" /><span className="text-cockpit-300 font-medium">Xăng gốc</span></span>
                    <span className="flex items-center gap-1.5"><span className="w-3.5 h-1 bg-[#ab63fa] rounded-full shadow-[0_0_6px_#ab63fa]" /><span className="text-[#ab63fa] font-bold">Lọc AI (Màu tím)</span></span>
                  </div>
                )}
              </div>
              <div className="flex items-center gap-2">
                {viewMode === 'chart' && data.length > 0 && (
                  <div className="flex items-center gap-1.5 mr-1">
                    <button
                      onClick={() => zoomToRefuel(data)}
                      className="flex items-center gap-1 px-2.5 py-1 text-[11px] font-medium rounded-md bg-amber-500/15 border border-amber-500/30 text-amber-300 hover:bg-amber-500/25 transition-colors cursor-pointer"
                      title="Tự động phóng to vào mốc xe đổ xăng lớn nhất"
                    >
                      <Zap size={12} className="text-amber-400" /> Soi đoạn đổ xăng
                    </button>
                    <button
                      onClick={resetZoom}
                      className="flex items-center gap-1 px-2 py-1 text-[11px] font-medium rounded-md bg-cockpit-900 border border-cockpit-750 text-cockpit-300 hover:bg-cockpit-800 transition-colors cursor-pointer"
                      title="Đặt lại xem toàn bộ chuỗi thời gian"
                    >
                      <RotateCcw size={12} /> Đặt lại
                    </button>
                  </div>
                )}
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
                {/* Nút phóng to toàn màn hình */}
                <button
                  onClick={() => setIsExpanded(prev => !prev)}
                  className={`p-1.5 rounded-lg border transition-colors cursor-pointer ${
                    isExpanded 
                      ? "bg-[#ab63fa]/20 border-[#ab63fa] text-[#ab63fa]" 
                      : "bg-cockpit-900 border-cockpit-800 text-cockpit-400 hover:text-cockpit-200 hover:bg-cockpit-800"
                  }`}
                  title={isExpanded ? "Thu nhỏ lại kích thước chuẩn" : "Phóng to toàn màn hình"}
                >
                  {isExpanded ? <Minimize2 size={15} /> : <Maximize2 size={15} />}
                </button>
              </div>
            </div>
            
            <div className="flex-1 w-full h-full">
              {data.length > 0 ? (
                viewMode === 'chart' ? (() => {
                  const filteredData = data.filter(row => showOnlySpikes ? Math.abs((row.raw_fuel ?? 0) - (row.clean ?? 0)) > 4 : true);
                  
                  const chartData = filteredData.length <= 1500 
                    ? filteredData 
                    : filteredData.filter((_, idx) => idx % Math.ceil(filteredData.length / 1500) === 0 || idx === filteredData.length - 1);

                  const fuelValues = chartData.flatMap(d => [d.raw_fuel, d.clean ?? d.ai_smooth_tracking]).filter((v): v is number => typeof v === "number" && v > 0);
                  const minVal = fuelValues.length > 0 ? Math.min(...fuelValues) : 0;
                  const maxVal = fuelValues.length > 0 ? Math.max(...fuelValues) : 100;
                  const yMin = Math.max(0, Math.floor(minVal - (maxVal - minVal) * 0.1));
                  const yMax = Math.ceil(maxVal + (maxVal - minVal) * 0.1);

                  return (
                    <ResponsiveContainer width="100%" height="100%">
                      <AreaChart 
                        data={chartData} 
                        margin={{ top: 10, right: 20, left: 0, bottom: 20 }}
                        onMouseMove={(e: any) => {
                          if (e && e.activePayload && e.activePayload.length > 0) {
                            const pTime = e.activePayload[0].payload.timestamp || e.activePayload[0].payload.time;
                            setHoveredTime(pTime || null);
                          }
                        }}
                        onMouseLeave={() => setHoveredTime(null)}
                      >
                        <defs>
                          <linearGradient id="colorRawHist" x1="0" y1="0" x2="0" y2="1">
                            <stop offset="5%" stopColor="#ef4444" stopOpacity={0.15}/>
                            <stop offset="95%" stopColor="#ef4444" stopOpacity={0}/>
                          </linearGradient>
                          <linearGradient id="colorSmoothHist" x1="0" y1="0" x2="0" y2="1">
                            <stop offset="5%" stopColor="#ab63fa" stopOpacity={0.25}/>
                            <stop offset="95%" stopColor="#ab63fa" stopOpacity={0.02}/>
                          </linearGradient>
                        </defs>
                        <CartesianGrid strokeDasharray="3 3" stroke="#2a3342" vertical={true} />
                        <XAxis 
                          dataKey="timestamp" 
                          stroke="#64748b" 
                          fontSize={10} 
                          tickFormatter={(t) => {
                            if (!t) return "";
                            const parts = t.split(" ");
                            return parts.length > 1 ? parts[1].slice(0, 5) : t.slice(11, 16);
                          }}
                          minTickGap={40}
                        />
                        <YAxis 
                          domain={[yMin, yMax]} 
                          stroke="#64748b" 
                          fontSize={10} 
                          unit="L" 
                          width={45} 
                          tickFormatter={(v) => Number(v).toFixed(0)} 
                        />
                        <Tooltip 
                          contentStyle={{ backgroundColor: "#0b1329", borderColor: "#334155", borderRadius: "8px", fontSize: "11px", boxShadow: "0 10px 15px -3px rgba(0, 0, 0, 0.7)" }}
                          formatter={(value: any, name: any) => {
                            const valNum = Number(value).toFixed(1);
                            if (name === "raw_fuel") return [`${valNum} L`, "Xăng gốc (Thô)"];
                            if (name === "clean") return [`${valNum} L`, "Lọc AI"];
                            return [valNum, name];
                          }}
                          labelFormatter={(label) => `Thời gian: ${label}`}
                        />
                        <Area 
                          type="monotone" 
                          dataKey="raw_fuel" 
                          stroke="#ef4444" 
                          strokeWidth={1.2} 
                          fillOpacity={1} 
                          fill="url(#colorRawHist)" 
                          name="raw_fuel" 
                          isAnimationActive={false} 
                        />
                        <Area 
                          type="monotone" 
                          dataKey="clean" 
                          stroke="#ab63fa" 
                          strokeWidth={2.2} 
                          fillOpacity={1} 
                          fill="url(#colorSmoothHist)" 
                          name="clean" 
                          isAnimationActive={false} 
                        />
                        <Brush 
                          dataKey="timestamp" 
                          height={22} 
                          stroke="#ab63fa" 
                          fill="#0b1329" 
                          startIndex={brushRange.startIndex}
                          endIndex={brushRange.endIndex}
                          onChange={(range: any) => setBrushRange(range)}
                          tickFormatter={(t) => {
                            if (!t) return "";
                            const parts = t.split(" ");
                            return parts.length > 1 ? parts[1].slice(0, 5) : t.slice(11, 16);
                          }}
                        />
                      </AreaChart>
                    </ResponsiveContainer>
                  );
                })() : (
                  <MapView data={data} />
                )
              ) : (
                <div className="flex flex-col items-center justify-center h-full text-cockpit-500 text-xs italic">
                  <BarChart2 size={28} className="mb-2 opacity-30 text-cockpit-400" />
                  <span>Chưa có dữ liệu biểu đồ. Hãy khởi động phát luồng dữ liệu xe.</span>
                </div>
              )}
            </div>
          </div>

          <div className="mt-2.5 flex justify-between items-center text-xs text-cockpit-500">
            <div>Dữ liệu luồng trực tiếp chuẩn hệ thống GPS/Tracking Vcomsat</div>
            <div className="font-mono text-cockpit-400">
              Hiển thị <span className="text-emerald-400 font-bold">{data.length}</span> điểm đo thời gian thực
            </div>
          </div>
        </>
      )}
    </div>
  );
}
