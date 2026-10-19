export default function RangeBand({ claimed, predicted, modelName }) {
  const { p10, p50, p90 } = predicted
  const maxScale = Math.max(claimed, p90) * 1.15
  const pct = (v) => `${Math.min((v / maxScale) * 100, 100)}%`

  return (
    <div className="bg-slate-800/60 rounded-xl p-5 border border-slate-700">
      <div className="flex justify-between items-start mb-4">
        <div>
          <h3 className="font-semibold text-lg">{modelName}</h3>
          <p className="text-sm text-slate-400">
            Claimed: <span className="text-amber-400 font-medium">{claimed} km</span>
          </p>
        </div>
        <div className="text-right">
          <p className="text-2xl font-bold text-rangeiq-500">{p50} km</p>
          <p className="text-xs text-slate-400">median estimate</p>
        </div>
      </div>

      <div className="relative h-10 bg-slate-900 rounded-full overflow-hidden mb-2">
        {/* Uncertainty band p10-p90 */}
        <div
          className="absolute top-1 bottom-1 bg-rangeiq-600/40 rounded-full border border-rangeiq-500/50"
          style={{ left: pct(p10), width: `calc(${pct(p90)} - ${pct(p10)})` }}
        />
        {/* Median marker */}
        <div
          className="absolute top-0 bottom-0 w-1 bg-rangeiq-400 z-10"
          style={{ left: pct(p50) }}
        />
        {/* Claimed marker */}
        <div
          className="absolute top-0 bottom-0 w-0.5 bg-amber-400/80 z-10"
          style={{ left: pct(claimed) }}
          title="Manufacturer claimed"
        />
      </div>

      <div className="flex justify-between text-xs text-slate-400">
        <span>Worst: {p10} km</span>
        <span>Likely: {p10}–{p90} km</span>
        <span>Best: {p90} km</span>
      </div>
    </div>
  )
}
