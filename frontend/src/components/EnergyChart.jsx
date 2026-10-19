import { Bar } from 'react-chartjs-2'
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  BarElement,
  Title,
  Tooltip,
  Legend,
} from 'chart.js'

ChartJS.register(CategoryScale, LinearScale, BarElement, Title, Tooltip, Legend)

export default function EnergyChart({ breakdown }) {
  if (!breakdown?.length) return null

  const labels = breakdown.map(b => b.label)
  const values = breakdown.map(b => b.km_impact)
  const colors = values.map(v => (v >= 0 ? 'rgba(16, 185, 129, 0.7)' : 'rgba(239, 68, 68, 0.7)'))

  const data = {
    labels,
    datasets: [{
      label: 'Range impact (km)',
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
      title: { display: true, text: 'Energy Factor Breakdown (SHAP)', color: '#94a3b8', font: { size: 13 } },
    },
    scales: {
      x: {
        grid: { color: '#334155' },
        ticks: { color: '#94a3b8' },
        title: { display: true, text: 'km impact on range', color: '#64748b' },
      },
      y: {
        grid: { display: false },
        ticks: { color: '#cbd5e1', font: { size: 11 } },
      },
    },
  }

  return (
    <div className="bg-slate-800/60 rounded-xl p-4 border border-slate-700">
      <Bar data={data} options={options} />
    </div>
  )
}
