/**
 * AboutPage — Phase 7
 *
 * Shows methodology, validation numbers (fetched live from /api/validation),
 * and the ML disclosure. No "validated" claim — shows computed error numbers.
 */
import { useEffect, useState } from 'react'
import { getValidation } from '../api/client'

function Stat({ label, value, sub }) {
  return (
    <div className="bg-slate-900/60 rounded-lg p-4 text-center">
      <p className="text-2xl font-bold text-rangeiq-400">{value}</p>
      <p className="text-xs text-slate-300 mt-1">{label}</p>
      {sub && <p className="text-xs text-slate-500 mt-0.5">{sub}</p>}
    </div>
  )
}

export default function AboutPage() {
  const [val, setVal]   = useState(null)
  const [err, setErr]   = useState(false)

  useEffect(() => {
    getValidation()
      .then(setVal)
      .catch(() => setErr(true))
  }, [])

  return (
    <div className="max-w-3xl space-y-8">
      <div>
        <h2 className="text-3xl font-bold mb-2">Methodology</h2>
        <p className="text-slate-400">
          RangeIQ uses a segment-by-segment physics model calibrated against journalist road tests,
          plus an ML uncertainty estimator trained on simulated data. Here is exactly what we do
          and what we don't claim.
        </p>
      </div>

      {/* Validation numbers — the only place we quote accuracy */}
      <section className="bg-slate-800/60 rounded-xl p-6 border border-slate-700 space-y-4">
        <h3 className="text-xl font-semibold text-rangeiq-500">Road-Test Validation</h3>
        <p className="text-sm text-slate-300">
          The physics simulator was compared against Autocar India journalist test-drive results
          (28 Apr 2026 road-test edition, cross-checked via Gaadify). These are the only
          accuracy numbers we publish — measured, not claimed.
        </p>

        {val ? (
          <>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
              <Stat
                label="Simulator MAPE"
                value={`${val.simulator_mape_pct?.toFixed(1) ?? '—'}%`}
                sub="mean abs % error"
              />
              <Stat
                label="Calibrated MAPE"
                value={`${val.calibrated_mape_pct?.toFixed(1) ?? val.calib_mape_pct?.toFixed(1) ?? '—'}%`}
                sub="after calibration"
              />
              <Stat
                label="Test MAPE"
                value={`${val.test_mape_pct?.toFixed(1) ?? '—'}%`}
                sub="held-out set"
              />
              <Stat
                label="Spearman ρ"
                value={val.spearman_rho?.toFixed(3) ?? '—'}
                sub={val.spearman_p != null ? `p=${val.spearman_p.toFixed(3)}` : ''}
              />
            </div>
            <div className="text-xs text-slate-500 space-y-1">
              <p>Validation scenario: 50% city (35 km/h) + 50% highway (80 km/h), flat, 28°C, 2 passengers, AC on.</p>
              <p>Calibrated parameters: p_aux_kw = {val.calibrated_p_aux_kw ?? 1.0} kW, η_regen = {val.calibrated_eta_regen ?? 0.30}.</p>
              {val.n_cases != null && <p>Cases: {val.n_cases} Autocar India road tests.</p>}
              {val.source && <p>Source: {val.source}</p>}
            </div>
          </>
        ) : err ? (
          <p className="text-xs text-amber-400">Validation data unavailable — ensure backend has run validation.</p>
        ) : (
          <p className="text-xs text-slate-400">Loading validation data…</p>
        )}
      </section>

      {/* Physics model */}
      <section className="bg-slate-800/60 rounded-xl p-6 border border-slate-700 space-y-4">
        <h3 className="text-xl font-semibold text-rangeiq-500">Physics Simulator</h3>
        <p className="text-sm text-slate-300">
          Segment-by-segment force balance (F = ma) over OSRM road geometry. Energy is decomposed into:
        </p>
        <ul className="text-sm text-slate-400 space-y-1.5 list-disc pl-5">
          <li><strong>Aerodynamic drag</strong> — ½ρC<sub>d</sub>Av², ISA air density at route altitude</li>
          <li><strong>Rolling resistance</strong> — C<sub>rr</sub>·m·g (Gillespie, Vehicle Dynamics)</li>
          <li><strong>Gradient</strong> — m·g·sin θ over OSRM elevation profile</li>
          <li><strong>Inertia</strong> — accelerating from stop per stop-and-go model (Ahn &amp; Rakha 2016)</li>
          <li><strong>Regenerative braking</strong> — partial energy recovery on descents (η_regen = 0.30, calibrated)</li>
          <li><strong>Auxiliary loads</strong> — p_aux = 1.0 kW (calibrated), includes lights, infotainment</li>
          <li><strong>HVAC</strong> — thermal model per Yuksel &amp; Michalek 2015; AC only when ac_on = True</li>
        </ul>
        <p className="text-xs text-slate-500">
          References: Gillespie "Fundamentals of Vehicle Dynamics"; Ahn &amp; Rakha, Appl. Energy 168 (2016);
          Fiori et al. (2016); Dollinger &amp; Fischerauer, Energies 14(18) 5804 (2021);
          Yuksel &amp; Michalek 2015; ARAI AIS-040 (MIDC).
        </p>
      </section>

      {/* ML layer */}
      <section className="bg-slate-800/60 rounded-xl p-6 border border-slate-700 space-y-4">
        <h3 className="text-xl font-semibold text-rangeiq-500">ML Uncertainty Estimator</h3>
        <p className="text-sm text-slate-300">
          The ML layer is a fast emulator and uncertainty estimator of the physics model.
          It is <strong>trained on simulated data</strong> — not real telematics.
          Accuracy against reality is judged only by the road-test validation above.
        </p>
        <ul className="text-sm text-slate-400 space-y-1 list-disc pl-5">
          <li>16 EVs × 300 synthetic trips × 40 Monte Carlo parameter samples = 192,000 physics calls → 4,800 training rows</li>
          <li>HistGradientBoostingRegressor with quantile loss (p10, p50, p90)</li>
          <li>GroupKFold(4) by EV model — prevents data leakage across vehicles</li>
          <li>Cross-validated MAE (p50): ~7.7 Wh/km on simulated data</li>
        </ul>
        <p className="text-xs text-slate-500">
          p90 Wh/km → conservative (short) range · p10 Wh/km → optimistic (long) range.
          Physics fallback (±15% band) used when models are absent.
        </p>
      </section>

      {/* Data sources */}
      <section className="bg-slate-800/60 rounded-xl p-6 border border-slate-700 space-y-4">
        <h3 className="text-xl font-semibold text-rangeiq-500">Data &amp; Honesty Policy</h3>
        <ul className="text-sm text-slate-400 space-y-1.5 list-disc pl-5">
          <li>16 EV models seeded from publicly available manufacturer spec sheets</li>
          <li>Ex-showroom prices are <strong>not published</strong> — price filter warns and excludes NULL-price EVs</li>
          <li>Drag coefficient and frontal area values marked as <em>assumed</em> when not officially published</li>
          <li>Route data: OSRM (OpenStreetMap). Elevation: Open-Elevation. Weather: OpenWeatherMap.</li>
          <li>All fallbacks (API unavailable) are surfaced as visible warnings, never silently substituted</li>
          <li>No authentication, no user tracking</li>
        </ul>
      </section>
    </div>
  )
}
