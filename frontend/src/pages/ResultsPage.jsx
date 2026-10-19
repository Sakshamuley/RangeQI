import { useEffect, useState } from 'react'
import { Link } from 'react-router-dom'
import RangeBand from '../components/RangeBand'
import EnergyChart from '../components/EnergyChart'
import SuitabilityCard from '../components/SuitabilityCard'
import CarbonCallout from '../components/CarbonCallout'
import RouteMap from '../components/RouteMap'

export default function ResultsPage() {
  const [data, setData] = useState(null)
  const [route, setRoute] = useState(null)

  useEffect(() => {
    const raw = sessionStorage.getItem('rangeiq_result')
    if (raw) {
      const parsed = JSON.parse(raw)
      setData(parsed.result)
      setRoute(parsed.route)
      if (parsed.result?.scan_id) {
        const ids = JSON.parse(localStorage.getItem('rangeiq_scan_ids') || '[]')
        if (!ids.includes(parsed.result.scan_id)) {
          localStorage.setItem(
            'rangeiq_scan_ids',
            JSON.stringify([parsed.result.scan_id, ...ids].slice(0, 20)),
          )
        }
      }
    }
  }, [])

  if (!data) {
    return (
      <div className="text-center py-16">
        <p className="text-slate-400 mb-4">No results yet.</p>
        <Link to="/" className="text-rangeiq-500 hover:underline">Run a simulation</Link>
      </div>
    )
  }

  const { predictions, trip_distance_km, temperature_c, api_warnings, scan_id } = data

  return (
    <div className="space-y-8">
      <div className="flex flex-wrap justify-between items-start gap-4">
        <div>
          <h2 className="text-3xl font-bold">Your Results</h2>
          <p className="text-slate-400 text-sm mt-1">
            Trip: {trip_distance_km} km · {temperature_c}°C · Scan #{scan_id}
          </p>
        </div>
        <Link to="/" className="text-sm text-rangeiq-500 hover:underline">← New simulation</Link>
      </div>

      {api_warnings?.length > 0 && (
        <div className="bg-amber-950/40 border border-amber-600/40 rounded-lg p-3 text-sm text-amber-200">
          {api_warnings.map((w, i) => <p key={i}>{w}</p>)}
        </div>
      )}

      {route && <RouteMap route={route} profile={data.route_profile} />}

      <div className="grid gap-6">
        {predictions.map((pred, idx) => (
          <div key={pred.ev_model_id} className="space-y-4">
            <RangeBand
              claimed={pred.claimed_range_km}
              predicted={pred.predicted_range}
              modelName={pred.model_name}
            />
            <div className="grid md:grid-cols-2 gap-4">
              <EnergyChart breakdown={pred.energy_breakdown} />
              <div className="space-y-4">
                <SuitabilityCard suitability={pred.suitability} modelName={pred.model_name} />
                <CarbonCallout carbon={pred.carbon} />
              </div>
            </div>
            {idx < predictions.length - 1 && <hr className="border-slate-700" />}
          </div>
        ))}
      </div>

      <div className="bg-slate-800/40 rounded-lg p-4 text-xs text-slate-500">
        Range shown as p10–p90 uncertainty band. Median (p50) is the best single estimate.
        Manufacturer claimed range shown for contrast — real-world is typically lower.
      </div>
    </div>
  )
}
