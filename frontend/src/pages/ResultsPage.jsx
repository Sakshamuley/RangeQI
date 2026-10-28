/**
 * ResultsPage — Phase 7
 *
 * Reads sessionStorage key 'rangeiq_result' written by InputPage.
 * Expects the shape returned by POST /api/recommend (RecommendResponse).
 */
import { useEffect, useState } from 'react'
import { Link } from 'react-router-dom'
import RangeBand from '../components/RangeBand'
import EnergyChart from '../components/EnergyChart'
import CarbonCallout from '../components/CarbonCallout'
import VerdictBadge from '../components/VerdictBadge'

const VERDICT_ORDER = { Suitable: 0, Marginal: 1, 'Not suitable': 2 }

function WarningBanner({ warnings }) {
  if (!warnings?.length) return null
  return (
    <div className="bg-amber-950/40 border border-amber-600/40 rounded-lg p-4 text-sm text-amber-200 space-y-1">
      <p className="font-medium text-amber-300">⚠ Data-quality notices</p>
      {warnings.map((w, i) => <p key={i} className="text-xs">{w}</p>)}
    </div>
  )
}

function MetaBar({ data }) {
  return (
    <div className="flex flex-wrap gap-x-6 gap-y-1 text-xs text-slate-400">
      <span>🛣 Route: <strong className="text-slate-200">{data.route_distance_km} km</strong></span>
      {data.round_trip && <span>🔄 Round trip: <strong className="text-slate-200">{data.trip_distance_km} km</strong></span>}
      <span>🌡 {data.temperature_c}°C</span>
      <span>⬆ {data.elevation_gain_m} m gain</span>
      <span>⬇ {data.elevation_loss_m} m loss</span>
      <span>🎯 Required: <strong className="text-slate-200">{data.required_km} km</strong></span>
      <span className="text-slate-600">Scan #{data.scan_id}</span>
    </div>
  )
}

function EVCard({ rec, required_km }) {
  const [expanded, setExpanded] = useState(false)

  const marginColor = rec.verdict === 'Suitable'
    ? 'text-emerald-400'
    : rec.verdict === 'Marginal'
    ? 'text-amber-400'
    : 'text-red-400'

  return (
    <div className="bg-slate-800/60 border border-slate-700 rounded-xl overflow-hidden">
      {/* Header */}
      <div className="p-5">
        <div className="flex flex-wrap items-start justify-between gap-3 mb-4">
          <div>
            <div className="flex items-center gap-2 flex-wrap">
              <h3 className="font-semibold text-lg">{rec.model_name}</h3>
              {rec.variant && <span className="text-xs bg-slate-700 rounded px-2 py-0.5 text-slate-300">{rec.variant}</span>}
              <span className="text-xs bg-slate-700 rounded px-2 py-0.5 text-slate-400 capitalize">{rec.body_type}</span>
            </div>
            <p className="text-xs text-slate-500 mt-0.5">
              {rec.claimed_range_km} km claimed ({rec.claimed_cycle})
              {rec.ex_showroom_price_lakh
                ? ` · ₹${rec.ex_showroom_price_lakh}L`
                : rec.price_assumed
                ? ' · price unpublished'
                : ''}
            </p>
          </div>
          <VerdictBadge verdict={rec.verdict} />
        </div>

        <RangeBand
          claimed={rec.claimed_range_km}
          predicted={rec.predicted_range}
          required={required_km}
        />

        <div className="mt-3 flex flex-wrap gap-x-6 gap-y-1 text-xs text-slate-400">
          <span className={marginColor}>
            {rec.range_margin_km >= 0 ? '+' : ''}{rec.range_margin_km} km margin ({rec.range_margin_pct >= 0 ? '+' : ''}{rec.range_margin_pct}%)
          </span>
          <span>Trip energy: {rec.trip_energy_kwh} kWh</span>
          <span>Est. cost: ₹{rec.trip_cost_inr} <span className="text-slate-600">(illustrative)</span></span>
        </div>

        {rec.warnings?.length > 0 && (
          <div className="mt-2 text-xs text-amber-300 space-y-0.5">
            {rec.warnings.map((w, i) => <p key={i}>⚠ {w}</p>)}
          </div>
        )}
      </div>

      {/* Expandable detail */}
      <button
        type="button"
        onClick={() => setExpanded(v => !v)}
        className="w-full px-5 py-2 text-xs text-slate-400 hover:text-slate-200 border-t border-slate-700 text-left hover:bg-slate-700/30 transition"
      >
        {expanded ? '▲ Hide breakdown' : '▼ Show energy breakdown & CO₂'}
      </button>

      {expanded && (
        <div className="p-5 border-t border-slate-700 grid md:grid-cols-2 gap-4">
          <EnergyChart breakdown={rec.energy_breakdown} />
          <CarbonCallout carbon={rec.carbon} />
        </div>
      )}
    </div>
  )
}

export default function ResultsPage() {
  const [data, setData] = useState(null)

  useEffect(() => {
    const raw = sessionStorage.getItem('rangeiq_result')
    if (raw) setData(JSON.parse(raw).result)
  }, [])

  if (!data) {
    return (
      <div className="text-center py-16">
        <p className="text-slate-400 mb-4">No results yet.</p>
        <Link to="/" className="text-rangeiq-500 hover:underline">← Start a new search</Link>
      </div>
    )
  }

  const { recommendations, fallback_warnings, feasible_count, required_km } = data

  const suitable    = recommendations.filter(r => r.verdict === 'Suitable')
  const marginal    = recommendations.filter(r => r.verdict === 'Marginal')
  const notSuitable = recommendations.filter(r => r.verdict === 'Not suitable')

  return (
    <div className="space-y-8">
      {/* Page header */}
      <div className="flex flex-wrap justify-between items-start gap-4">
        <div>
          <h2 className="text-3xl font-bold">EV Suitability Results</h2>
          <p className="text-slate-400 text-sm mt-1">
            {feasible_count} of {recommendations.length} EV{recommendations.length !== 1 ? 's' : ''} can handle this trip
          </p>
        </div>
        <Link to="/" className="text-sm text-rangeiq-500 hover:underline">← New search</Link>
      </div>

      <MetaBar data={data} />
      <WarningBanner warnings={fallback_warnings} />

      {/* ML disclosure — required */}
      <div className="bg-slate-900/60 border border-slate-700 rounded-lg px-4 py-3 text-xs text-slate-400">
        <strong className="text-slate-300">How range bands are computed:</strong>{' '}
        The ML layer is a fast emulator and uncertainty estimator of the physics model, trained
        on simulated data. p10 = conservative (worst-case), p50 = expected median, p90 = optimistic.
        Accuracy against reality is judged only by the road-test validation — see the{' '}
        <Link to="/about" className="text-rangeiq-500 hover:underline">Methodology</Link> page.
      </div>

      {/* Verdict groups */}
      {suitable.length > 0 && (
        <section className="space-y-4">
          <h3 className="text-lg font-semibold text-emerald-400">✓ Suitable ({suitable.length})</h3>
          {suitable.map(r => <EVCard key={r.ev_model_id} rec={r} required_km={required_km} />)}
        </section>
      )}

      {marginal.length > 0 && (
        <section className="space-y-4">
          <h3 className="text-lg font-semibold text-amber-400">⚡ Marginal ({marginal.length})</h3>
          <p className="text-xs text-slate-500">
            Expected range covers the trip, but worst-case estimate does not.
          </p>
          {marginal.map(r => <EVCard key={r.ev_model_id} rec={r} required_km={required_km} />)}
        </section>
      )}

      {notSuitable.length > 0 && (
        <section className="space-y-4">
          <h3 className="text-lg font-semibold text-red-400">✗ Not suitable ({notSuitable.length})</h3>
          <p className="text-xs text-slate-500">
            Median range estimate does not cover the required distance.
          </p>
          {notSuitable.map(r => <EVCard key={r.ev_model_id} rec={r} required_km={required_km} />)}
        </section>
      )}

      {recommendations.length === 0 && (
        <div className="text-center py-8 text-slate-400">
          <p>No EVs returned. Check filters or fallback warnings above.</p>
        </div>
      )}

      <div className="bg-slate-800/40 rounded-lg p-4 text-xs text-slate-500 space-y-1">
        <p>Range shown as p10–p90 uncertainty band (conservative → optimistic). p50 is the best single estimate.</p>
        <p>Manufacturer claimed range shown for contrast — real-world is typically lower.</p>
        <p>Cost and CO₂ figures are illustrative. Electricity: ₹8/kWh (assumed). Grid: 0.72 kg CO₂/kWh (CEA 2022).</p>
      </div>
    </div>
  )
}
