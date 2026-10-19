import { useEffect, useState } from 'react'
import { Link } from 'react-router-dom'
import { getHistory } from '../api/client'

export default function HistoryPage() {
  const [scans, setScans] = useState([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')

  useEffect(() => {
    getHistory(20)
      .then(setScans)
      .catch(() => setError('Could not load history — is the backend running?'))
      .finally(() => setLoading(false))
  }, [])

  if (loading) return <p className="text-slate-400">Loading history…</p>
  if (error) return <p className="text-red-400">{error}</p>

  if (!scans.length) {
    return (
      <div className="text-center py-16">
        <h2 className="text-2xl font-bold mb-2">Prediction History</h2>
        <p className="text-slate-400 mb-4">No saved predictions yet.</p>
        <Link to="/" className="text-rangeiq-500 hover:underline">Run your first simulation</Link>
      </div>
    )
  }

  return (
    <div className="space-y-6">
      <h2 className="text-3xl font-bold">Prediction History</h2>
      <div className="space-y-3">
        {scans.map(scan => {
          const out = scan.output
          const first = out?.predictions?.[0]
          return (
            <div key={scan.id} className="bg-slate-800/60 rounded-xl p-4 border border-slate-700 flex flex-wrap justify-between gap-3">
              <div>
                <p className="font-medium">Scan #{scan.id}</p>
                <p className="text-xs text-slate-400">{new Date(scan.created_at).toLocaleString()}</p>
                <p className="text-sm mt-1">{out?.trip_distance_km} km trip · {first?.model_name}</p>
              </div>
              <div className="text-right">
                <p className="text-rangeiq-500 font-bold">
                  {first?.predicted_range?.p10}–{first?.predicted_range?.p90} km
                </p>
                <p className="text-xs text-slate-400">median {first?.predicted_range?.p50} km</p>
                <p className="text-xs text-amber-400">claimed {first?.claimed_range_km} km</p>
              </div>
            </div>
          )
        })}
      </div>
    </div>
  )
}
