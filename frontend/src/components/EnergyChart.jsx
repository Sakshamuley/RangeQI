/**
 * EnergyChart — Phase 7
 *
 * Horizontal bar chart of per-term energy breakdown (Wh/km).
 * Uses Chart.js via react-chartjs-2.
 *
 * Props: breakdown (EnergyTermBreakdown)
 */
import { Bar } from 'react-chartjs-2'
import {
  Chart as ChartJS,
  CategoryScale, LinearScale, BarElement, Tooltip, Legend,
} from 'chart.js'

ChartJS.register(CategoryScale, LinearScale, BarElement, Tooltip, Legend)

const TERMS = [
  { key: 'aero_wh_per_km',      label: 'Aero drag',      color: 'rgba(99, 102, 241, 0.8)'  },
  { key: 'rolling_wh_per_km',   label: 'Rolling resist.', color: 'rgba(16, 185, 129, 0.8)'  },
  { key: 'gradient_wh_per_km',  label: 'Gradient',        color: 'rgba(245, 158, 11, 0.8)'  },
  { key: 'inertia_wh_per_km',   label: 'Inertia',         color: 'rgba(139, 92, 246, 0.8)'  },
  { key: 'auxiliary_wh_per_km', label: 'Auxiliary',       color: 'rgba(59, 130, 246, 0.8)'  },
  { key: 'hvac_wh_per_km',      label: 'HVAC',            color: 'rgba(236, 72, 153, 0.8)'  },
  { key: 'regen_wh_per_km',     label: 'Regen (saved)',   color: 'rgba(34, 197, 94, 0.5)'   },
]

export default function EnergyChart({ breakdown }) {
  if (!breakdown) return null

  const terms    = TERMS.filter(t => breakdown[t.key] !== undefined)
  const labels   = terms.map(t => t.label)
  const values   = terms.map(t => breakdown[t.key])
  const colors   = terms.map(t => t.color)

  const data = {
    labels,
    datasets: [{
      label: 'Wh/km',
      data: values,
      backgroundColor: colors,
      borderRadius: 4,
    }],
  }

  const options = {
    indexAxis: 'y',
    responsive: true,
    plugins: {
      legend: { display: false },
      tooltip: {
        callbacks: {
          label: ctx => ` ${ctx.raw.toFixed(1)} Wh/km`,
        },
      },
    },
    scales: {
      x: {
        ticks: { color: '#94a3b8', font: { size: 10 } },
        grid:  { color: '#1e293b' },
        title: { display: true, text: 'Wh/km', color: '#64748b', font: { size: 10 } },
      },
      y: {
        ticks: { color: '#94a3b8', font: { size: 10 } },
        grid:  { display: false },
      },
    },
  }

  return (
    <div className="bg-slate-900/60 rounded-lg p-4">
      <p className="text-xs text-slate-400 mb-3 font-medium">
        Physics breakdown — {breakdown.total_wh_per_km.toFixed(1)} Wh/km total
      </p>
      <Bar data={data} options={options} />
    </div>
  )
}
