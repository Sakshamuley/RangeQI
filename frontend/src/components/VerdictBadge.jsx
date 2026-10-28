/** VerdictBadge — coloured pill showing Suitable / Marginal / Not suitable */
const CONFIG = {
  'Suitable':     { bg: 'bg-emerald-900/60', border: 'border-emerald-600/60', text: 'text-emerald-300', label: '✓ Suitable' },
  'Marginal':     { bg: 'bg-amber-900/60',   border: 'border-amber-600/60',   text: 'text-amber-300',   label: '⚡ Marginal' },
  'Not suitable': { bg: 'bg-red-900/60',     border: 'border-red-600/60',     text: 'text-red-300',     label: '✗ Not suitable' },
}

export default function VerdictBadge({ verdict }) {
  const c = CONFIG[verdict] ?? CONFIG['Not suitable']
  return (
    <span className={`inline-flex items-center px-3 py-1 rounded-full text-sm font-semibold border ${c.bg} ${c.border} ${c.text}`}>
      {c.label}
    </span>
  )
}
