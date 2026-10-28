/**
 * RouteMap — Phase 7
 *
 * Shows a Leaflet map with the route waypoints from the recommendation response.
 * Falls back to a straight line between start/end if only coords are available.
 *
 * Props:
 *   waypoints  array of {lat, lon} from route_info
 *   profile    array of {distance_km, elevation_m, gradient} for elevation chart
 */
import { MapContainer, TileLayer, Polyline, Marker } from 'react-leaflet'
import L from 'leaflet'

// Fix Leaflet default icon paths when bundled with Vite
delete L.Icon.Default.prototype._getIconUrl
L.Icon.Default.mergeOptions({
  iconUrl:       'https://unpkg.com/leaflet@1.9.4/dist/images/marker-icon.png',
  iconRetinaUrl: 'https://unpkg.com/leaflet@1.9.4/dist/images/marker-icon-2x.png',
  shadowUrl:     'https://unpkg.com/leaflet@1.9.4/dist/images/marker-shadow.png',
})

export default function RouteMap({ waypoints, profile }) {
  if (!waypoints?.length) return null

  const line   = waypoints.map(w => [w.lat, w.lon])
  const start  = line[0]
  const end    = line[line.length - 1]
  const center = [(start[0] + end[0]) / 2, (start[1] + end[1]) / 2]

  // Elevation profile SVG
  let elevSvg = null
  if (profile?.length > 1) {
    const elevs  = profile.map(p => p.elevation_m ?? 0)
    const eMin   = Math.min(...elevs)
    const eMax   = Math.max(...elevs) || eMin + 1
    const W = profile.length
    const points = elevs.map((e, i) => `${i},${60 - Math.round(((e - eMin) / (eMax - eMin)) * 55)}`).join(' ')
    elevSvg = (
      <div className="bg-slate-800/60 rounded-xl p-4 border border-slate-700">
        <p className="text-xs text-slate-400 mb-2">Elevation profile ({eMin.toFixed(0)}–{eMax.toFixed(0)} m)</p>
        <svg viewBox={`0 0 ${W} 60`} className="w-full h-14" preserveAspectRatio="none">
          <polyline points={points} fill="none" stroke="#10b981" strokeWidth="1.5" />
        </svg>
      </div>
    )
  }

  return (
    <div className="space-y-4">
      <div className="h-56 rounded-xl overflow-hidden border border-slate-700">
        <MapContainer center={center} zoom={10} style={{ height: '100%', width: '100%' }}>
          <TileLayer
            attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>'
            url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
          />
          <Polyline positions={line} color="#6366f1" weight={3} opacity={0.8} />
          <Marker position={start} />
          <Marker position={end} />
        </MapContainer>
      </div>
      {elevSvg}
    </div>
  )
}
