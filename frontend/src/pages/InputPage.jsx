import { useEffect, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { getEVModels, predict } from '../api/client'

const SEASONS = ['summer', 'monsoon', 'winter', 'spring']
const STYLES = ['calm', 'normal', 'aggressive']

export default function InputPage() {
  const navigate = useNavigate()
  const [models, setModels] = useState([])
  const [selected, setSelected] = useState([])
  const [mode, setMode] = useState('distance')
  const [dailyDistance, setDailyDistance] = useState(80)
  const [drivingStyle, setDrivingStyle] = useState('normal')
  const [season, setSeason] = useState('summer')
  const [passengers, setPassengers] = useState(1)
  const [avgSpeed, setAvgSpeed] = useState(50)
  const [route, setRoute] = useState({
    start_lat: 19.076, start_lon: 72.877,
    end_lat: 18.520, end_lon: 73.856,
  })
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')

  useEffect(() => {
    getEVModels()
      .then(data => {
        setModels(data)
        if (data.length) setSelected([data[0].id])
      })
      .catch(() => setError('Backend unavailable — start the API server first'))
  }, [])

  const toggleModel = (id) => {
    setSelected(prev =>
      prev.includes(id) ? prev.filter(x => x !== id) : [...prev, id]
    )
  }

  const handleSubmit = async (e) => {
    e.preventDefault()
    if (!selected.length) {
      setError('Select at least one EV model')
      return
    }
    setLoading(true)
    setError('')
    try {
      const payload = {
        ev_model_ids: selected,
        driving_style: drivingStyle,
        season,
        num_passengers: passengers,
        avg_speed_kmh: avgSpeed,
      }
      if (mode === 'distance') {
        payload.daily_distance_km = dailyDistance
      } else {
        payload.route = route
      }
      const result = await predict(payload)
      sessionStorage.setItem('rangeiq_result', JSON.stringify({ result, route: mode === 'route' ? route : null }))
      navigate('/results')
    } catch (err) {
      setError(err.response?.data?.detail || err.message || 'Prediction failed')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="max-w-2xl mx-auto">
      <div className="mb-8">
        <h2 className="text-3xl font-bold mb-2">Simulate Your Commute</h2>
        <p className="text-slate-400">
          Get honest range estimates with uncertainty bands — not manufacturer lab figures.
        </p>
      </div>

      <form onSubmit={handleSubmit} className="space-y-6">
        {/* Trip mode */}
        <div className="bg-slate-800/60 rounded-xl p-5 border border-slate-700">
          <label className="block text-sm font-medium mb-3">Trip Input</label>
          <div className="flex gap-2 mb-4">
            {['distance', 'route'].map(m => (
              <button
                key={m}
                type="button"
                onClick={() => setMode(m)}
                className={`px-4 py-2 rounded-lg text-sm capitalize ${
                  mode === m ? 'bg-rangeiq-600 text-white' : 'bg-slate-700 text-slate-300'
                }`}
              >
                {m === 'distance' ? 'Daily Distance' : 'Route Coords'}
              </button>
            ))}
          </div>

          {mode === 'distance' ? (
            <div>
              <label className="text-xs text-slate-400">Daily commute (km)</label>
              <input
                type="range" min={10} max={400} value={dailyDistance}
                onChange={e => setDailyDistance(Number(e.target.value))}
                className="w-full accent-rangeiq-500"
              />
              <p className="text-center text-rangeiq-500 font-bold">{dailyDistance} km</p>
            </div>
          ) : (
            <div className="grid grid-cols-2 gap-3 text-sm">
              {['start_lat','start_lon','end_lat','end_lon'].map(key => (
                <div key={key}>
                  <label className="text-xs text-slate-400">{key.replace('_', ' ')}</label>
                  <input
                    type="number" step="any"
                    value={route[key]}
                    onChange={e => setRoute(r => ({ ...r, [key]: parseFloat(e.target.value) }))}
                    className="w-full bg-slate-900 border border-slate-600 rounded px-2 py-1"
                  />
                </div>
              ))}
              <p className="col-span-2 text-xs text-slate-500">Default: Mumbai → Pune</p>
            </div>
          )}
        </div>

        {/* Driving style */}
        <div className="bg-slate-800/60 rounded-xl p-5 border border-slate-700">
          <label className="block text-sm font-medium mb-3">Driving Style</label>
          <input
            type="range" min={0} max={2} step={1}
            value={STYLES.indexOf(drivingStyle)}
            onChange={e => setDrivingStyle(STYLES[Number(e.target.value)])}
            className="w-full accent-rangeiq-500"
          />
          <p className="text-center capitalize text-rangeiq-500 font-medium">{drivingStyle}</p>
        </div>

        {/* Season & passengers */}
        <div className="grid grid-cols-2 gap-4">
          <div className="bg-slate-800/60 rounded-xl p-4 border border-slate-700">
            <label className="text-sm font-medium">Season</label>
            <select
              value={season}
              onChange={e => setSeason(e.target.value)}
              className="w-full mt-2 bg-slate-900 border border-slate-600 rounded px-2 py-2"
            >
              {SEASONS.map(s => <option key={s} value={s}>{s}</option>)}
            </select>
          </div>
          <div className="bg-slate-800/60 rounded-xl p-4 border border-slate-700">
            <label className="text-sm font-medium">Avg Speed (km/h)</label>
            <input
              type="number" min={20} max={120} value={avgSpeed}
              onChange={e => setAvgSpeed(Number(e.target.value))}
              className="w-full mt-2 bg-slate-900 border border-slate-600 rounded px-2 py-2"
            />
          </div>
        </div>

        {/* EV selection */}
        <div className="bg-slate-800/60 rounded-xl p-5 border border-slate-700">
          <label className="block text-sm font-medium mb-3">Compare EV Models</label>
          <div className="max-h-48 overflow-y-auto space-y-2">
            {models.map(m => (
              <label key={m.id} className="flex items-center gap-3 p-2 rounded hover:bg-slate-700/50 cursor-pointer">
                <input
                  type="checkbox"
                  checked={selected.includes(m.id)}
                  onChange={() => toggleModel(m.id)}
                  className="accent-rangeiq-500"
                />
                <span className="flex-1 text-sm">{m.model_name}</span>
                <span className="text-xs text-amber-400">{m.claimed_range_km} km claimed</span>
              </label>
            ))}
          </div>
        </div>

        {error && <p className="text-red-400 text-sm">{error}</p>}

        <button
          type="submit"
          disabled={loading}
          className="w-full py-3 bg-rangeiq-600 hover:bg-rangeiq-700 disabled:opacity-50 rounded-xl font-semibold transition"
        >
          {loading ? 'Simulating…' : 'Predict Real-World Range'}
        </button>
      </form>
    </div>
  )
}
