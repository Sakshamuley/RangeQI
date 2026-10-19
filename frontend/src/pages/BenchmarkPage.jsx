import { useEffect, useState } from 'react'
import { Bar } from 'react-chartjs-2'
import {
  Chart as ChartJS, CategoryScale, LinearScale, BarElement, Title, Tooltip, Legend,
} from 'chart.js'
import { getBenchmark, getValidation } from '../api/client'

ChartJS.register(CategoryScale, LinearScale, BarElement, Title, Tooltip, Legend)

export default function BenchmarkPage() {
  const [benchmark, setBenchmark] = useState(null)
  const [validation, setValidation] = useState(null)
  const [error, setError] = useState('')

  useEffect(() => {
    Promise.all([getBenchmark(), getValidation()])
      .then(([b, v]) => { setBenchmark(b); setValidation(v) })
      .catch(() => setError('Benchmark data unavailable — ensure backend has trained models'))
  }, [])

  if (error) return <p className="text-red-400">{error}</p>
  if (!benchmark) return <p className="text-slate-400">Loading benchmark data…</p>

  const models = ['linear_regression', 'random_forest', 'xgboost']
  const chartData = {
    labels: models.map(m => m.replace('_', ' ')),
    datasets: [
      {
        label: 'MAE (km)',
        data: models.map(m => benchmark[m]?.mae),
        backgroundColor: 'rgba(16, 185, 129, 0.7)',
      },
      {
        label: 'RMSE (km)',
        data: models.map(m => benchmark[m]?.rmse),
        backgroundColor: 'rgba(59, 130, 246, 0.7)',
      },
    ],
  }

  return (
    <div className="space-y-8">
      <div>
        <h2 className="text-3xl font-bold mb-2">Model Benchmark</h2>
        <p className="text-slate-400">
          Trained on {benchmark.training_samples?.toLocaleString()} physics-simulated trips.
          Best model: <span className="text-rangeiq-500 font-medium">{benchmark.best_model}</span>
        </p>
      </div>

      <div className="bg-slate-800/60 rounded-xl p-5 border border-slate-700">
        <Bar
          data={chartData}
          options={{
            responsive: true,
            plugins: { legend: { labels: { color: '#94a3b8' } }, title: { display: true, text: 'Model Comparison (lower is better)', color: '#94a3b8' } },
            scales: {
              x: { ticks: { color: '#94a3b8' }, grid: { color: '#334155' } },
              y: { ticks: { color: '#94a3b8' }, grid: { color: '#334155' }, title: { display: true, text: 'Error (km)', color: '#64748b' } },
            },
          }}
        />
      </div>

      <div className="overflow-x-auto">
        <table className="w-full text-sm">
          <thead>
            <tr className="text-left text-slate-400 border-b border-slate-700">
              <th className="py-2 pr-4">Model</th>
              <th className="py-2 pr-4">MAE</th>
              <th className="py-2 pr-4">RMSE</th>
              <th className="py-2 pr-4">Train (s)</th>
              <th className="py-2">Inference (ms)</th>
            </tr>
          </thead>
          <tbody>
            {models.map(m => (
              <tr key={m} className="border-b border-slate-800">
                <td className="py-3 pr-4 capitalize">{m.replace('_', ' ')}</td>
                <td className="py-3 pr-4">{benchmark[m]?.mae}</td>
                <td className="py-3 pr-4">{benchmark[m]?.rmse}</td>
                <td className="py-3 pr-4">{benchmark[m]?.train_time_sec}</td>
                <td className="py-3">{benchmark[m]?.inference_time_ms}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {benchmark.quantile_models && (
        <div className="bg-slate-800/60 rounded-xl p-5 border border-slate-700">
          <h3 className="font-semibold mb-3">Quantile Models (Uncertainty)</h3>
          <div className="grid grid-cols-3 gap-4 text-sm">
            {Object.entries(benchmark.quantile_models).map(([k, v]) => (
              <div key={k} className="bg-slate-900 rounded p-3">
                <p className="text-rangeiq-500 font-bold uppercase">{k}</p>
                <p>MAE: {v.mae} km</p>
                <p className="text-slate-500">Train: {v.train_time_sec}s</p>
              </div>
            ))}
          </div>
        </div>
      )}

      {validation && (
        <div className="bg-slate-800/60 rounded-xl p-5 border border-slate-700">
          <h3 className="font-semibold mb-2">Real-World Validation</h3>
          <p className="text-sm text-slate-400 mb-4">
            {validation.cases} journalist test-drive cases · Simulator MAE: {validation.simulator_mae_km} km
            {validation.ml_mae_km && ` · ML MAE: ${validation.ml_mae_km} km`}
          </p>
          <div className="overflow-x-auto max-h-64 overflow-y-auto">
            <table className="w-full text-xs">
              <thead>
                <tr className="text-slate-400 border-b border-slate-700">
                  <th className="py-1 text-left">Model</th>
                  <th className="py-1">Claimed</th>
                  <th className="py-1">Reported</th>
                  <th className="py-1">Predicted</th>
                  <th className="py-1">Error</th>
                </tr>
              </thead>
              <tbody>
                {validation.results?.map((r, i) => (
                  <tr key={i} className="border-b border-slate-800">
                    <td className="py-2">{r.model}</td>
                    <td className="text-center">{r.claimed_km}</td>
                    <td className="text-center text-amber-400">{r.reported_km}</td>
                    <td className="text-center text-rangeiq-500">{r.ml_predicted_km ?? r.simulator_predicted_km}</td>
                    <td className="text-center">{r.ml_error_km ?? r.simulator_error_km}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  )
}
