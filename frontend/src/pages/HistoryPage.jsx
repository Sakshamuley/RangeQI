/**
 * HistoryPage — Phase 7
 *
 * Lists past prediction scans from GET /api/history.
 * Adapts to both old /api/predict shape and new /api/recommend shape.
 */
import { useEffect, useState } from 'react'
import { Link } from 'react-router-dom'
import { getHistory } from '../api/client'

const VERDICT_COLOR = {
  'Suitable':     'text-emerald-400',
  'Marginal':     'text-amber-400',
  'Not suitable': 'text-red-400',
}

export default function HistoryPage() {
  const [scans,   setScans]   = useState([])
  const [loading, setLoading] = useState(true)
  const [error,   setError]   = useState('')

  useEffect(() => {
    getHistory(20)
      .then(setScans)
      .catch(() => setError('Could not load history — is the backend running?'))
      .finally(() => setLoading(false))
  }, [])

  if (loading) return <p className="text-slate-400">Loading history…</p>
  if (error)   return <p className="text-red-400">{error}</p>

  if (!scans.length) {
    return (
      <div className="text-center py-16">
        <h2 className="text-2xl font-bold mb-2">Prediction History</h2>
        <p className="text-slate-400 mb-4">No saved predictions yet.</p>
        <Link to="/" className="text-rangeiq-500 hover:underline">Run your first search</Link>
      </div>
    )
  }

  return (
    <div className="space-y-6">
      <h2 className="text-3xl font-bold">Prediction History</h2>
      <div className="space-y-3">
        {scans.map(scan => {
          const inp = scan.input  ?? {}
          const out = scan.output ?? {}

          // Recommend output shape
          const feasible   = out.feasible_count
          const topVerdict = out.top_verdict

          // Legacy predict output shape
          const firstPred  = out.predictions?.[0]

          return (
            <div key={scan.id}
              className="bg-slate-800/60 rounded-xl p-4 border border-slate-700 flex flex-wrap justify-between gap-3">
              <div>
                <p className="font-medium text-sm">Scan #{scan.id}</p>
                <p className="text-xs text-slate-500">{new Date(scan.created_at).toLocaleString()}</p>
                {inp.route && (
                  <p className="text-xs text-slate-400 mt-1">
                    ({inp.route.start_lat?.toFixed(3)}, {inp.route.start_lon?.toFixed(3)}) →
                    ({inp.route.end_lat?.toFixed(3)},   {inp.route.end_lon?.toFixed(3)})
                  </p>
                )}
                <p className="text-xs text-slate-500 mt-0.5">
                  Style: {inp.driving_style ?? '—'} · {inp.num_passengers ?? '—'} pax ·
                  AC {inp.ac_on ? 'on' : 'off'}
                  {inp.round_trip ? ' · round trip' : ''}
                </p>
              </div>

              <div className="text-right">
                {/* Recommend shape */}
                {topVerdict != null && (
                  <p className={`font-bold text-sm ${VERDICT_COLOR[topVerdict] ?? 'text-slate-300'}`}>
                    {topVerdict}
                  </p>
                )}
                {feasible != null && (
                  <p className="text-xs text-slate-400">{feasible} EV(s) feasible</p>
                )}
                {/* Legacy predict shape */}
                {firstPred?.predicted_range && (
                  <>
                    <p className="text-rangeiq-500 font-bold text-sm">
                      {firstPred.predicted_range.p10}–{firstPred.predicted_range.p90} km
                    </p>
                    <p className="text-xs text-slate-400">median {firstPred.predicted_range.p50} km</p>
                  </>
                )}
              </div>
            </div>
          )
        })}
      </div>
    </div>
  )
}
