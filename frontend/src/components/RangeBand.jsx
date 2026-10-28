/**
 * RangeBand — Phase 7
 *
 * Visualises the p10/p50/p90 range quantiles plus the claimed range and the
 * required distance as a proportional horizontal bar.
 *
 * Props:
 *   claimed   (number) manufacturer claimed range km
 *   predicted ({ p10, p50, p90 })
 *   required  (number|undefined) required km including reserve
 */
export default function RangeBand({ claimed, predicted, required }) {
  const max = Math.max(claimed, predicted.p90, required ?? 0) * 1.1

  const pct = (v) => `${Math.min((v / max) * 100, 100).toFixed(1)}%`

  return (
    <div className="space-y-3">
      {/* Predicted band */}
      <div>
        <div className="flex justify-between text-xs text-slate-400 mb-1">
          <span>Real-world range estimate</span>
          <span className="text-rangeiq-400 font-medium">{predicted.p50} km (median)</span>
        </div>
        <div className="relative h-7 bg-slate-900 rounded-full overflow-hidden">
          {/* p10–p90 band */}
          <div
            className="absolute top-0 bottom-0 bg-rangeiq-700/50 rounded-full"
            style={{ left: pct(predicted.p10), width: `${((predicted.p90 - predicted.p10) / max) * 100}%` }}
          />
          {/* p50 line */}
          <div
            className="absolute top-1 bottom-1 w-0.5 bg-rangeiq-400 rounded"
            style={{ left: pct(predicted.p50) }}
          />
          {/* required marker */}
          {required != null && (
            <div
              className="absolute top-0 bottom-0 w-0.5 bg-amber-400"
              style={{ left: pct(required) }}
              title={`Required: ${required} km`}
            />
          )}
        </div>
        <div className="flex justify-between text-xs text-slate-500 mt-0.5">
          <span>{predicted.p10} km <span className="text-slate-600">(pessimistic)</span></span>
          <span className="text-slate-600">{predicted.p90} km (optimistic)</span>
        </div>
      </div>

      {/* Claimed bar */}
      <div>
        <div className="flex justify-between text-xs text-slate-500 mb-1">
          <span>Manufacturer claimed</span>
          <span className="text-amber-400">{claimed} km</span>
        </div>
        <div className="relative h-3 bg-slate-900 rounded-full overflow-hidden">
          <div
            className="absolute top-0 bottom-0 left-0 bg-amber-700/40 rounded-full"
            style={{ width: pct(claimed) }}
          />
        </div>
      </div>

      {required != null && (
        <p className="text-xs text-slate-500">
          <span className="inline-block w-2 h-2 bg-amber-400 rounded-sm mr-1" />
          Required distance: {required} km (trip + reserve)
        </p>
      )}
    </div>
  )
}
