/**
 * InputPage — Phase 7
 *
 * Primary flow: user types place names → geocoded → POST /api/recommend → ranked results.
 * No EV model pre-selection: the backend shortlists and ranks all EVs.
 */
import { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { geocode, recommend } from '../api/client'

const BODY_TYPES  = ['', 'hatchback', 'sedan', 'compact_suv', 'suv', 'mpv']
const BODY_LABELS = ['Any body type', 'Hatchback', 'Sedan', 'Compact SUV', 'SUV', 'MPV']
const STYLES      = ['calm', 'normal', 'aggressive']

function GeocodeInput({ label, value, onChange, placeholder }) {
  const [query, setQuery]   = useState(value?.place || '')
  const [status, setStatus] = useState('idle') // idle | loading | ok | error
  const [msg, setMsg]       = useState('')

  const lookup = async () => {
    if (!query.trim()) return
    setStatus('loading')
    setMsg('')
    try {
      const data = await geocode(query.trim())
      onChange({ place: query.trim(), lat: data.lat, lon: data.lon })
      setStatus('ok')
      setMsg(`${data.lat.toFixed(4)}, ${data.lon.toFixed(4)}`)
    } catch (err) {
      setStatus('error')
      setMsg(err.response?.data?.detail || 'Location not found')
    }
  }

  const handleKey = (e) => { if (e.key === 'Enter') { e.preventDefault(); lookup() } }

  return (
    <div>
      <label className="block text-xs text-slate-400 mb-1">{label}</label>
      <div className="flex gap-2">
        <input
          type="text"
          value={query}
          onChange={e => { setQuery(e.target.value); setStatus('idle'); setMsg('') }}
          onKeyDown={handleKey}
          placeholder={placeholder}
          className="flex-1 bg-slate-900 border border-slate-600 rounded-lg px-3 py-2 text-sm focus:outline-none focus:border-rangeiq-500"
        />
        <button
          type="button"
          onClick={lookup}
          disabled={status === 'loading'}
          className="px-3 py-2 bg-slate-700 hover:bg-slate-600 rounded-lg text-sm disabled:opacity-50 transition"
        >
          {status === 'loading' ? '…' : 'Locate'}
        </button>
      </div>
      {msg && (
        <p className={`mt-1 text-xs ${status === 'ok' ? 'text-emerald-400' : 'text-red-400'}`}>
          {status === 'ok' ? '✓ ' : '✗ '}{msg}
        </p>
      )}
    </div>
  )
}

export default function InputPage() {
  const navigate = useNavigate()

  // Locations
  const [origin, setOrigin] = useState(null)
  const [dest,   setDest]   = useState(null)

  // Filters
  const [bodyType,     setBodyType]     = useState('')
  const [priceMin,     setPriceMin]     = useState('')
  const [priceMax,     setPriceMax]     = useState('')

  // Driving profile
  const [drivingStyle,   setDrivingStyle]   = useState('normal')
  const [numPassengers,  setNumPassengers]  = useState(1)
  const [acOn,           setAcOn]           = useState(true)
  const [roundTrip,      setRoundTrip]      = useState(false)
  const [reservePct,     setReservePct]     = useState(10)

  const [showAdvanced, setShowAdvanced] = useState(false)
  const [loading,  setLoading]  = useState(false)
  const [error,    setError]    = useState('')

  const handleSubmit = async (e) => {
    e.preventDefault()
    if (!origin?.lat || !dest?.lat) {
      setError('Please locate both origin and destination before simulating.')
      return
    }
    setLoading(true)
    setError('')
    try {
      const payload = {
        route: {
          start_lat: origin.lat, start_lon: origin.lon,
          end_lat:   dest.lat,   end_lon:   dest.lon,
        },
        driving_style:  drivingStyle,
        num_passengers: numPassengers,
        ac_on:          acOn,
        round_trip:     roundTrip,
        reserve_pct:    reservePct,
      }
      if (bodyType)  payload.body_type      = bodyType
      if (priceMin)  payload.price_min_lakh = parseFloat(priceMin)
      if (priceMax)  payload.price_max_lakh = parseFloat(priceMax)

      const result = await recommend(payload)
      sessionStorage.setItem('rangeiq_result', JSON.stringify({ result, mode: 'recommend' }))
      navigate('/results')
    } catch (err) {
      setError(err.response?.data?.detail || err.message || 'Recommendation failed')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="max-w-2xl mx-auto">
      <div className="mb-8">
        <h2 className="text-3xl font-bold mb-2">Find the Right EV for Your Route</h2>
        <p className="text-slate-400">
          Enter your journey. We rank every EV by whether it can actually complete the trip —
          with honest uncertainty bands, not manufacturer lab figures.
        </p>
      </div>

      <form onSubmit={handleSubmit} className="space-y-5">

        {/* ── Route ── */}
        <div className="bg-slate-800/60 rounded-xl p-5 border border-slate-700 space-y-4">
          <p className="text-sm font-medium">Journey</p>
          <GeocodeInput
            label="From"
            value={origin}
            onChange={setOrigin}
            placeholder="e.g. Mumbai, Maharashtra"
          />
          <GeocodeInput
            label="To"
            value={dest}
            onChange={setDest}
            placeholder="e.g. Pune, Maharashtra"
          />
          <label className="flex items-center gap-2 text-sm cursor-pointer">
            <input type="checkbox" checked={roundTrip} onChange={e => setRoundTrip(e.target.checked)}
              className="accent-rangeiq-500" />
            <span>Round trip</span>
          </label>
        </div>

        {/* ── Filters ── */}
        <div className="bg-slate-800/60 rounded-xl p-5 border border-slate-700 space-y-4">
          <p className="text-sm font-medium">Filters <span className="text-slate-500">(optional)</span></p>

          <div>
            <label className="block text-xs text-slate-400 mb-1">Body type</label>
            <select
              value={bodyType}
              onChange={e => setBodyType(e.target.value)}
              className="w-full bg-slate-900 border border-slate-600 rounded-lg px-3 py-2 text-sm"
            >
              {BODY_TYPES.map((v, i) => <option key={v} value={v}>{BODY_LABELS[i]}</option>)}
            </select>
          </div>

          <div className="grid grid-cols-2 gap-3">
            <div>
              <label className="block text-xs text-slate-400 mb-1">Min price (₹ lakh)</label>
              <input type="number" min={0} step={0.5} placeholder="—"
                value={priceMin} onChange={e => setPriceMin(e.target.value)}
                className="w-full bg-slate-900 border border-slate-600 rounded-lg px-3 py-2 text-sm" />
            </div>
            <div>
              <label className="block text-xs text-slate-400 mb-1">Max price (₹ lakh)</label>
              <input type="number" min={0} step={0.5} placeholder="—"
                value={priceMax} onChange={e => setPriceMax(e.target.value)}
                className="w-full bg-slate-900 border border-slate-600 rounded-lg px-3 py-2 text-sm" />
            </div>
          </div>
          <p className="text-xs text-amber-400">
            Note: ex-showroom prices for all EVs are currently unpublished in our database.
            Price filters will exclude all results and show a warning.
          </p>
        </div>

        {/* ── Driving profile ── */}
        <div className="bg-slate-800/60 rounded-xl p-5 border border-slate-700 space-y-4">
          <div className="flex justify-between items-center">
            <p className="text-sm font-medium">Driving profile</p>
            <button type="button" onClick={() => setShowAdvanced(v => !v)}
              className="text-xs text-slate-400 hover:text-slate-200 transition">
              {showAdvanced ? 'Hide advanced ▲' : 'Advanced ▼'}
            </button>
          </div>

          <div>
            <label className="text-xs text-slate-400">Driving style</label>
            <div className="flex gap-2 mt-1">
              {STYLES.map(s => (
                <button key={s} type="button"
                  onClick={() => setDrivingStyle(s)}
                  className={`flex-1 py-2 rounded-lg text-sm capitalize transition ${
                    drivingStyle === s
                      ? 'bg-rangeiq-600 text-white'
                      : 'bg-slate-700 text-slate-300 hover:bg-slate-600'
                  }`}>
                  {s}
                </button>
              ))}
            </div>
          </div>

          {showAdvanced && (
            <div className="space-y-4 pt-2 border-t border-slate-700">
              <div>
                <label className="text-xs text-slate-400">
                  Passengers: <span className="text-white">{numPassengers}</span>
                </label>
                <input type="range" min={1} max={7} value={numPassengers}
                  onChange={e => setNumPassengers(Number(e.target.value))}
                  className="w-full accent-rangeiq-500 mt-1" />
              </div>
              <div>
                <label className="text-xs text-slate-400">
                  Reserve buffer: <span className="text-white">{reservePct}%</span>
                </label>
                <input type="range" min={5} max={20} step={5} value={reservePct}
                  onChange={e => setReservePct(Number(e.target.value))}
                  className="w-full accent-rangeiq-500 mt-1" />
                <p className="text-xs text-slate-500 mt-1">
                  Required range = trip distance × (1 + {reservePct}%)
                </p>
              </div>
              <label className="flex items-center gap-2 text-sm cursor-pointer">
                <input type="checkbox" checked={acOn} onChange={e => setAcOn(e.target.checked)}
                  className="accent-rangeiq-500" />
                <span>Air conditioning on</span>
              </label>
            </div>
          )}
        </div>

        {error && (
          <div className="bg-red-950/40 border border-red-600/40 rounded-lg p-3 text-sm text-red-300">
            {error}
          </div>
        )}

        <button
          type="submit"
          disabled={loading}
          className="w-full py-3 bg-rangeiq-600 hover:bg-rangeiq-700 disabled:opacity-50 rounded-xl font-semibold transition text-white"
        >
          {loading ? 'Finding best EVs…' : 'Find Suitable EVs →'}
        </button>

        <p className="text-center text-xs text-slate-500">
          Route fetched from OSRM (OpenStreetMap). Elevation from Open-Elevation.
          Weather from OpenWeatherMap. All sources may fall back to estimates.
        </p>
      </form>
    </div>
  )
}
