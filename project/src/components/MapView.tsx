import { useEffect } from "react";
import { MapContainer, TileLayer, Polyline, Marker, Popup, useMap } from "react-leaflet";
import "leaflet/dist/leaflet.css";
import L from "leaflet";

// Fix default marker icon issue in leaflet with bundlers
import iconUrl from "leaflet/dist/images/marker-icon.png";
import iconRetinaUrl from "leaflet/dist/images/marker-icon-2x.png";
import shadowUrl from "leaflet/dist/images/marker-shadow.png";

const DefaultIcon = L.icon({
  iconUrl,
  iconRetinaUrl,
  shadowUrl,
  iconSize: [25, 41],
  iconAnchor: [12, 41],
  popupAnchor: [1, -34],
  tooltipAnchor: [16, -28],
  shadowSize: [41, 41]
});
L.Marker.prototype.options.icon = DefaultIcon;

function FitBounds({ positions }: { positions: [number, number][] }) {
  const map = useMap();
  useEffect(() => {
    if (positions && positions.length > 1) {
      try {
        map.fitBounds(positions, { padding: [40, 40] });
      } catch {
        // ignore if map not ready
      }
    }
  }, [positions, map]);
  return null;
}

interface MapViewProps {
  data: any[];
}

export default function MapView({ data }: MapViewProps) {
  // Filter out invalid coordinates and fix swapped lat/lng
  const validPoints = data
    .map((row) => {
      let lat = Number(row.lat);
      let lng = Number(row.lng);
      if (isNaN(lat) || isNaN(lng) || (lat === 0 && lng === 0)) return null;

      // Nếu bị ngược (Lat > 90 là Kinh độ Lng, Lng < 90 là Vĩ độ Lat)
      if (lat > 50 && lng < 50) {
        const tmp = lat;
        lat = lng;
        lng = tmp;
      }
      return { ...row, lat, lng };
    })
    .filter((row): row is NonNullable<typeof row> => row !== null && row.lat > 5 && row.lat < 30 && row.lng > 90 && row.lng < 120);

  if (validPoints.length === 0) {
    return (
      <div className="flex items-center justify-center h-full w-full text-cockpit-500 text-sm italic bg-cockpit-900 rounded-xl border border-cockpit-800">
        Không có dữ liệu tọa độ hợp lệ để vẽ bản đồ
      </div>
    );
  }

  const positions: [number, number][] = validPoints.map((row) => [
    row.lat,
    row.lng,
  ]);

  const startPoint = validPoints[0];
  const endPoint = validPoints[validPoints.length - 1];

  return (
    <div className="w-full h-full rounded-xl overflow-hidden border border-cockpit-800 shadow-inner z-0 relative isolate">
      <MapContainer
        center={[startPoint.lat, startPoint.lng]}
        zoom={14}
        className="w-full h-full"
        style={{ background: "#1a1d24" }}
      >
        <TileLayer
          attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
          url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
        />
        <FitBounds positions={positions} />
        
        {positions.length > 0 && (
          <Polyline 
            positions={positions} 
            color="#ec4899" 
            weight={4} 
            opacity={0.8}
          />
        )}

        {startPoint && (
          <Marker position={[startPoint.lat, startPoint.lng]}>
            <Popup className="text-black min-w-[150px]">
              <div className="font-bold text-sm mb-1 text-blue-600">Điểm bắt đầu</div>
              <div className="text-xs mt-1"><b>Thời gian:</b> {startPoint.timestamp}</div>
              <div className="text-xs mt-1"><b>Vận tốc:</b> {startPoint.speed || 0} km/h</div>
              <div className="text-xs mt-1"><b>Nhiên liệu:</b> {startPoint.ai_enhanced?.toFixed(1)} L</div>
            </Popup>
          </Marker>
        )}

        {endPoint && endPoint !== startPoint && (
          <Marker position={[endPoint.lat, endPoint.lng]}>
            <Popup className="text-black min-w-[150px]">
              <div className="font-bold text-sm mb-1 text-red-600">Điểm kết thúc</div>
              <div className="text-xs mt-1"><b>Thời gian:</b> {endPoint.timestamp}</div>
              <div className="text-xs mt-1"><b>Vận tốc:</b> {endPoint.speed || 0} km/h</div>
              <div className="text-xs mt-1"><b>Nhiên liệu:</b> {endPoint.ai_enhanced?.toFixed(1)} L</div>
            </Popup>
          </Marker>
        )}
      </MapContainer>
    </div>
  );
}
