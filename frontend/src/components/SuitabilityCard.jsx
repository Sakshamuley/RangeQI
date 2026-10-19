const styles = {
  suitable: 'border-emerald-500/50 bg-emerald-950/40 text-emerald-300',
  marginal: 'border-amber-500/50 bg-amber-950/40 text-amber-300',
  not_suitable: 'border-red-500/50 bg-red-950/40 text-red-300',
}

const labels = {
  suitable: 'Suitable',
  marginal: 'Marginal',
  not_suitable: 'Not Suitable',
}

export default function SuitabilityCard({ suitability, modelName }) {
  const { verdict, safe_verdict, buffer_pct, message } = suitability
  return (
    <div className={`rounded-xl p-4 border ${styles[verdict] || styles.marginal}`}>
      <div className="flex justify-between items-center mb-2">
        <span className="font-semibold">{modelName}</span>
        <span className="text-sm font-bold uppercase tracking-wide">{labels[verdict]}</span>
      </div>
      <p className="text-sm opacity-90 mb-2">{message}</p>
      <div className="flex gap-4 text-xs opacity-75">
        <span>Buffer: {buffer_pct}%</span>
        <span className={safe_verdict === 'safe' ? 'text-emerald-400' : 'text-red-400'}>
          Worst-case: {safe_verdict === 'safe' ? 'Safe' : 'Risky'}
        </span>
      </div>
    </div>
  )
}
