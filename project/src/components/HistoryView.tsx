import { useState } from "react";
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

export default function HistoryView() {
  const [selectedVehicle, setSelectedVehicle] = useState("24H-04650");
  const [startDate, setStartDate] = useState(new Date().toISOString().split("T")[0]);
  const [endDate, setEndDate] = useState(new Date().toISOString().split("T")[0]);
  const [data, setData] = useState<HistoryRecord[]>([]);
  const [showOnlySpikes, setShowOnlySpikes] = useState(false);
  const [viewMode, setViewMode] = useState<'chart' | 'map'>('chart');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const handleSearch = async () => {
    setLoading(true);
    setError("");
    try {
      const response = await fetch(`http://localhost:8000/api/history?vehicle_id=${selectedVehicle}&start_date=${startDate}&end_date=${endDate}`);
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

  const handleExport = () => {
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
        <div className="flex flex-col gap-1">
          <label className="text-xs font-medium text-cockpit-400 uppercase tracking-wider">Chọn xe</label>
          <select
            value={selectedVehicle}
            onChange={(e) => setSelectedVehicle(e.target.value)}
            className="bg-cockpit-800 border border-cockpit-700 text-cockpit-100 rounded-lg px-3 py-2 text-sm focus:outline-none focus:border-fuel-filter transition-colors font-semibold"
          >
            <option value="24H-04650">🚗 24H-04650 (Xe tải fulltt)</option>
            <option value="29C-92841">🚗 29C-92841 (Cao tốc Ninh Bình)</option>
            <option value="29H-77123">🚗 29H-77123 (Thử nghiệm leo dốc)</option>
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
          <label className="text-xs font-medium text-cockpit-400 uppercase tracking-wider">Đến ngày</label>
          <input
            type="date"
            value={endDate}
            onChange={(e) => setEndDate(e.target.value)}
            className="bg-cockpit-800 border border-cockpit-700 text-cockpit-100 rounded-lg px-3 py-2 text-sm focus:outline-none focus:border-fuel-filter transition-colors"
          />
        </div>
        
        <div className="flex items-end h-full pt-5">
          <button
            onClick={handleSearch}
            disabled={loading}
            className="flex items-center gap-2 bg-fuel-filter text-black px-5 py-2.5 rounded-lg text-sm font-semibold hover:bg-fuel-filter/90 transition-colors disabled:opacity-50"
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
                <td colSpan={5} className="px-6 py-8 text-center text-red-400 bg-red-400/10">
                  {error}
                </td>
              </tr>
            ) : data.length === 0 ? (
              <tr>
                <td colSpan={5} className="px-6 py-12 text-center text-cockpit-500">
                  Chưa có dữ liệu nào. Hãy bấm Tra cứu.
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
                
                return (
                  <tr key={idx} className="border-b border-cockpit-800/50 hover:bg-cockpit-800/30 transition-colors whitespace-nowrap">
                    <td className="px-4 py-2 font-medium text-cockpit-100 font-mono">{timeStr}</td>
                    <td className="px-4 py-2 text-right text-red-400 font-mono">{Number(rawVal).toFixed(1)}</td>
                    <td className="px-4 py-2 text-right text-purple-400 font-mono font-semibold">{Number(cleanVal).toFixed(1)}</td>
                    <td className="px-4 py-2 text-center text-blue-400 font-mono">{speedVal}</td>
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
            viewMode === 'chart' ? (
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart data={data.filter(row => showOnlySpikes ? Math.abs(row.raw_fuel - row.ai_enhanced) > 4 : true)} margin={{ top: 10, right: 15, left: -10, bottom: 0 }}>
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
                  domain={['auto', 'auto']} 
                  tick={{ fontSize: 10, fill: '#9ca3af' }}
                  stroke="#374151"
                  tickFormatter={(val) => `${Number(val).toFixed(0)} L`}
                />
                <Tooltip 
                  formatter={(value: any, name: any) => [
                    `${Number(value).toFixed(1)} L`,
                    name === 'raw_fuel' ? 'Xăng gốc' : 'AI-Kalman'
                  ]}
                  contentStyle={{ backgroundColor: '#111827', borderColor: '#374151', color: '#f3f4f6', borderRadius: '8px', fontSize: '12px' }}
                  itemStyle={{ color: '#c084fc', fontWeight: 'bold' }}
                  labelStyle={{ color: '#9ca3af', marginBottom: '4px' }}
                />
                
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
            ) : (
              <MapView data={data} />
            )
          ) : (
            <div className="flex items-center justify-center h-full text-cockpit-500 text-sm italic">
              Tra cứu dữ liệu để xem {viewMode === 'chart' ? 'biểu đồ' : 'bản đồ'}
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
