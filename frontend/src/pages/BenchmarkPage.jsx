/**
 * BenchmarkPage — Phase 7
 *
 * Shows ML benchmark results (from /api/benchmark) and validation results
 * (from /api/validation). Adapts to the actual Phase 5 benchmark_results.json shape.
 */
import { useEffect, useState } from 'react'
import { Bar } from 'react-chartjs-2'
import {
  Chart as ChartJS, CategoryScale, LinearScale, BarElement, Title, Tooltip, Legend,
} from 'chart.js'
import { getBenchmark, getValidation } from '../api/client'

ChartJS.register(CategoryScale, LinearScale, BarElement, Title, Tooltip, Legend)

const CHART_OPTIONS = {
  responsive: true,
  plugins: {
    legend: { labels: { color: '#94a3b8', font: { size: 11 } } },
    title: { display: false },
  },
  scales: {
    x: { ticks: { color: '#94a3b8', font: { size: 10 } }, grid: { color: '#1e293b' } },
    y: {
      ticks: { color: '#94a3b8', font: { size: 10 } },
      grid:  { color: '#1e293b' },
      title: { display: true, text: 'MAE (Wh/km)', color: '#64748b', font: { size: 10 } },
    },
  },
}

export default function BenchmarkPage() {
  const [benchmark,  setBenchmark]  = useState(null)
  const [validation, setValidation] = useState(null)
  const [error,      setError]      = useState('')

  useEffect(() => {
    Promise.allSettled([getBenchmark(), getValidation()]).then(([b, v]) => {
      if (b.status === 'fulfilled') setBenchmark(b.value)
      else setError('Benchmark data unavailable — ensure backend has trained models and run validation.')
      if (v.status === 'fulfilled') setValidation(v.value)
    })
  }, [])

  // Build chart from quantile_models if present
  const quantileModels = benchmark?.quantile_models ?? {}
  const qKeys   = Object.keys(quantileModels)
  const qLabels = qKeys.map(k => `q${k}`)
  const qMAEs   = qKeys.map(k => quantileModels[k]?.cv_mae_wh_per_km ?? quantileModels[k]?.mae ?? null)

  const chartData = {
    labels: qLabels.length ? qLabels : ['p10', 'p50', 'p90'],
    datasets: [{
      label: 'CV MAE (Wh/km)',
      data: qMAEs.length ? qMAEs : [],
      backgroundColor: ['rgba(99,102,241,0.7)', 'rgba(16,185,129,0.7)', 'rgba(245,158,11,0.7)'],
      borderRadius: 4,
    }],
  }

  if (error && !benchmark) return (
    <div className="text-amber-400 text-sm">{error}</div>
  )

  if (!benchmark) return (
    <p className="text-slate-400">Loading benchmark data…</p>
  )

  return (
    <div className="space-y-8">
      <div>
        <h2 className="text-3xl font-bold mb-2">ML Benchmark</h2>
        <p className="text-slate-400 text-sm">
          The ML quantile models are trained on{' '}
          <strong>{(benchmark.training_rows ?? benchmark.training_samples)?.toLocaleString() ?? '—'}</strong>{' '}
          physics-simulated rows.{' '}
          <span className="text-amber-300">All training data is synthetic.</span>
        </p>
      </div>

      {/* Quantile model MAE chart */}
      {qMAEs.some(v => v != null) && (
        <div className="bg-slate-800/60 rounded-xl p-5 border border-slate-700">
          <h3 className="text-sm font-medium mb-4 text-slate-300">
            Quantile model cross-validated MAE (Wh/km) — lower is better
          </h3>
          <Bar data={chartData} options={CHART_OPTIONS} />
        </div>
      )}

      {/* Quantile model details table */}
      {qKeys.length > 0 && (
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="text-left text-slate-400 border-b border-slate-700">
                <th className="py-2 pr-4">Quantile</th>
                <th className="py-2 pr-4">CV MAE (Wh/km)</th>
                <th className="py-2 pr-4">Training rows</th>
                <th className="py-2">Note</th>
              </tr>
            </thead>
            <tbody>
              {qKeys.map(k => {
                const m = quantileModels[k]
                return (
                  <tr key={k} className="border-b border-slate-800">
                    <td className="py-3 pr-4 font-mono text-rangeiq-400">q{k}</td>
                    <td className="py-3 pr-4">{(m?.cv_mae_wh_per_km ?? m?.mae)?.toFixed(2) ?? '—'}</td>
                    <td className="py-3 pr-4">{m?.training_rows?.toLocaleString() ?? (benchmark.training_rows ?? '—')}</td>
                    <td className="py-3 text-slate-500 text-xs">GroupKFold(4) by EV model</td>
                  </tr>
                )
              })}
            </tbody>
          </table>
        </div>
      )}

      {/* Data generation summary */}
      <div className="bg-slate-800/60 rounded-xl p-5 border border-slate-700 text-sm space-y-2">
        <h3 className="font-medium text-slate-300">Training data generation</h3>
        <ul className="text-slate-400 space-y-1 list-disc pl-4 text-xs">
          <li>16 EV models × 300 trip conditions × 40 Monte Carlo parameter samples = 192,000 physics calls</li>
          <li>Collapsed to {(benchmark.training_rows ?? benchmark.training_samples)?.toLocaleString() ?? '4,800'} rows (mean over MC samples per trip)</li>
          <li>Calibrated centres: p_aux = 1.0 kW, η_regen = 0.30</li>
          <li>GroupKFold(n_splits=4) split by EV model to prevent leakage</li>
        </ul>
      </div>

      {/* Validation table */}
      {validation && (
        <div className="bg-slate-800/60 rounded-xl p-5 border border-slate-700 space-y-4">
          <div>
            <h3 className="font-semibold mb-1">Real-World Validation</h3>
            <p className="text-sm text-slate-400">
              {validation.n_cases ?? validation.cases} Autocar India journalist road-test cases.{' '}
              Calibrated MAPE:{' '}
              <strong className="text-rangeiq-400">
                {(validation.calibrated_mape_pct ?? validation.calib_mape_pct)?.toFixed(1) ?? '—'}%
              </strong>{' '}
              · Test MAPE:{' '}
              <strong className="text-rangeiq-400">
                {validation.test_mape_pct?.toFixed(1) ?? '—'}%
              </strong>
            </p>
          </div>
          {validation.results?.length > 0 && (
            <div className="overflow-x-auto max-h-72 overflow-y-auto">
              <table className="w-full text-xs">
                <thead>
                  <tr className="text-slate-400 border-b border-slate-700 sticky top-0 bg-slate-800">
                    <th className="py-1.5 text-left pr-4">Model</th>
                    <th className="py-1.5 text-right pr-3">Claimed (km)</th>
                    <th className="py-1.5 text-right pr-3">Road-test (km)</th>
                    <th className="py-1.5 text-right pr-3">Predicted (km)</th>
                    <th className="py-1.5 text-right">Error (%)</th>
                  </tr>
                </thead>
                <tbody>
                  {validation.results.map((r, i) => (
                    <tr key={i} className="border-b border-slate-800">
                      <td className="py-2 pr-4">{r.model ?? r.ev_name}</td>
                      <td className="py-2 pr-3 text-right text-amber-400">{r.claimed_km ?? r.claimed_range_km}</td>
                      <td className="py-2 pr-3 text-right text-slate-300">{r.reported_km ?? r.actual_km}</td>
                      <td className="py-2 pr-3 text-right text-rangeiq-400">{r.predicted_km?.toFixed(0) ?? '—'}</td>
                      <td className="py-2 text-right">
                        <span className={Math.abs(r.error_pct) > 15 ? 'text-amber-400' : 'text-slate-400'}>
                          {r.error_pct != null ? `${r.error_pct > 0 ? '+' : ''}${r.error_pct.toFixed(1)}%` : '—'}
                        </span>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
          <p className="text-xs text-slate-600">
            Source: Autocar India 28 Apr 2026 road-test edition. Cross-checked vs Gaadify.
          </p>
        </div>
      )}

      <div className="bg-slate-800/40 rounded-lg p-4 text-xs text-slate-500">
        All benchmark figures are on simulated data. Real-world accuracy is shown in the validation table above.
      </div>
    </div>
  )
}
