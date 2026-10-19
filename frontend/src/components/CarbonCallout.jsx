export default function CarbonCallout({ carbon }) {
  if (!carbon) return null
  return (
    <div className="bg-slate-800/60 rounded-xl p-5 border border-slate-700">
      <h4 className="font-semibold text-rangeiq-500 mb-3">Carbon & Cost Comparison</h4>
      <div className="grid grid-cols-2 gap-4 text-sm">
        <div>
          <p className="text-slate-400">CO₂ saved vs petrol</p>
          <p className="text-2xl font-bold text-emerald-400">{carbon.co2_saved_kg} kg</p>
        </div>
        <div>
          <p className="text-slate-400">Cost saved (INR)</p>
          <p className="text-2xl font-bold text-emerald-400">₹{carbon.cost_saved_inr}</p>
        </div>
        <div>
          <p className="text-slate-500 text-xs">EV trip CO₂</p>
          <p>{carbon.ev_co2_kg} kg</p>
        </div>
        <div>
          <p className="text-slate-500 text-xs">Petrol equivalent</p>
          <p>{carbon.petrol_co2_kg} kg · ₹{carbon.petrol_cost_inr}</p>
        </div>
      </div>
    </div>
  )
}
