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
  // Default date to today
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
      const response = await fetch(`http://localhost:8000/api/history?start_date=${startDate}&end_date=${endDate}`);
      if (!response.ok) throw new Error("Failed to fetch data");
      const result = await response.json();
      setData(result);
    } catch (err: any) {
      setError(err.message || "Lỗi khi kết nối tới máy chủ.");
    } finally {
      setLoading(false);
    }
  };

  const handleExport = () => {
    window.location.href = `http://localhost:8000/api/export_history?start_date=${startDate}&end_date=${endDate}`;
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
              <th className="px-4 py-3 font-semibold"><MapPin size={14} className="inline mr-1"/>Địa điểm</th>
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
              data.filter(row => showOnlySpikes ? Math.abs(row.raw_fuel - row.ai_enhanced) > 5 : true).map((row, idx) => (
                <tr key={idx} className="border-b border-cockpit-800/50 hover:bg-cockpit-800/30 transition-colors whitespace-nowrap">
                  <td className="px-4 py-2 font-medium text-cockpit-100">{row.timestamp}</td>
                  <td className="px-4 py-2 text-right text-red-400 font-mono">{row.raw_fuel?.toFixed(1)}</td>
                  <td className="px-4 py-2 text-right text-purple-400 font-mono font-semibold">{row.ai_enhanced?.toFixed(1)}</td>
                  <td className="px-4 py-2 text-center text-blue-400 font-mono">{row.speed || 0}</td>
                  <td className="px-4 py-2 text-cockpit-300 truncate max-w-[200px]" title={row.address || "N/A"}>{row.address || "N/A"}</td>
                </tr>
              ))
            )}
          </tbody>
        </table>
      </div>
      
      {/* Chart / Map Bottom Half */}
      <div className="flex-[0.6] bg-white rounded-xl border border-gray-300 shadow-inner p-2 min-h-[300px] flex flex-col relative">
        <div className="flex items-center justify-between px-2 mb-2">
          <h3 className="text-gray-700 font-bold text-sm uppercase tracking-wide">
            {viewMode === 'chart' ? 'Biểu đồ nhiên liệu (Lít)' : 'Bản đồ hành trình'}
          </h3>
          <div className="flex bg-gray-100 rounded-lg p-1 border border-gray-200">
            <button
              onClick={() => setViewMode('chart')}
              className={`flex items-center gap-1.5 px-3 py-1 text-xs font-semibold rounded-md transition-colors ${viewMode === 'chart' ? 'bg-white text-blue-600 shadow-sm' : 'text-gray-500 hover:text-gray-700'}`}
            >
              <BarChart2 size={14} /> Biểu đồ
            </button>
            <button
              onClick={() => setViewMode('map')}
              className={`flex items-center gap-1.5 px-3 py-1 text-xs font-semibold rounded-md transition-colors ${viewMode === 'map' ? 'bg-white text-blue-600 shadow-sm' : 'text-gray-500 hover:text-gray-700'}`}
            >
              <Map size={14} /> Bản đồ
            </button>
          </div>
        </div>
        
        <div className="flex-1 w-full h-full text-black">
          {data.length > 0 ? (
            viewMode === 'chart' ? (
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart data={data.filter(row => showOnlySpikes ? Math.abs(row.raw_fuel - row.ai_enhanced) > 5 : true)} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
                <defs>
                  <linearGradient id="colorFuel" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#84cc16" stopOpacity={0.8}/>
                    <stop offset="95%" stopColor="#84cc16" stopOpacity={0.1}/>
                  </linearGradient>
                </defs>
                <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" vertical={true} />
                <XAxis 
                  dataKey="timestamp" 
                  tickFormatter={(val) => {
                    const d = new Date(val);
                    return `${d.getHours()}:${d.getMinutes().toString().padStart(2, '0')}`;
                  }}
                  tick={{ fontSize: 10, fill: '#6b7280' }}
                  tickMargin={5}
                />
                <YAxis 
                  domain={['dataMin - 20', 'dataMax + 20']} 
                  tick={{ fontSize: 10, fill: '#6b7280' }}
                  tickFormatter={(val) => `${val.toFixed(0)} L`}
                />
                <Tooltip 
                  formatter={(value: number) => [value.toFixed(1), undefined]}
                  contentStyle={{ backgroundColor: '#1f2937', borderColor: '#374151', color: '#f3f4f6', borderRadius: '8px', fontSize: '12px' }}
                  itemStyle={{ color: '#a78bfa', fontWeight: 'bold' }}
                  labelStyle={{ color: '#9ca3af', marginBottom: '4px' }}
                />
                
                {/* Reference line for spikes if filtering */}
                {showOnlySpikes && <ReferenceLine y={0} stroke="red" strokeDasharray="3 3" />}
                
                <Area 
                  type="stepAfter" 
                  dataKey="ai_enhanced" 
                  name="Nhiên liệu AI"
                  stroke="#ec4899" 
                  strokeWidth={2}
                  fillOpacity={1} 
                  fill="url(#colorFuel)" 
                  activeDot={{ r: 4, fill: "#ec4899", stroke: "#fff", strokeWidth: 2 }}
                />
              </AreaChart>
            </ResponsiveContainer>
            ) : (
              <MapView data={data} />
            )
          ) : (
            <div className="flex items-center justify-center h-full text-gray-400 text-sm italic">
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
