import { MapContainer, TileLayer, Polyline, Marker } from 'react-leaflet'
import L from 'leaflet'

delete L.Icon.Default.prototype._getIconUrl
L.Icon.Default.mergeOptions({
  iconUrl: 'https://unpkg.com/leaflet@1.9.4/dist/images/marker-icon.png',
  iconRetinaUrl: 'https://unpkg.com/leaflet@1.9.4/dist/images/marker-icon-2x.png',
  shadowUrl: 'https://unpkg.com/leaflet@1.9.4/dist/images/marker-shadow.png',
})

export default function RouteMap({ route, profile }) {
  if (!route) return null

  const { start_lat, start_lon, end_lat, end_lon } = route
  const center = [(start_lat + end_lat) / 2, (start_lon + end_lon) / 2]
  const line = [[start_lat, start_lon], [end_lat, end_lon]]

  return (
    <div className="space-y-4">
      <div className="h-64 rounded-xl overflow-hidden border border-slate-700">
        <MapContainer center={center} zoom={10} style={{ height: '100%', width: '100%' }}>
          <TileLayer
            attribution='&copy; OpenStreetMap'
            url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
          />
          <Polyline positions={line} color="#10b981" weight={4} />
          <Marker position={[start_lat, start_lon]} />
          <Marker position={[end_lat, end_lon]} />
        </MapContainer>
      </div>

      {profile?.length > 0 && (
        <div className="bg-slate-800/60 rounded-xl p-4 border border-slate-700">
          <h4 className="text-sm font-medium text-slate-300 mb-2">Elevation Profile</h4>
          <svg viewBox={`0 0 ${profile.length * 10} 80`} className="w-full h-20">
            {profile.map((p, i) => {
              const h = Math.min(p.elevation_m / 10, 70)
              return (
                <rect
                  key={i}
                  x={i * 10}
                  y={80 - h}
                  width={8}
                  height={h}
                  fill="#10b981"
                  opacity={0.7}
                />
              )
            })}
          </svg>
        </div>
      )}
    </div>
  )
}
