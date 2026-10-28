/**
 * CarbonCallout — Phase 7
 *
 * Shows the illustrative CO₂ + cost comparison from CarbonComparison schema.
 * Always labels figures as "illustrative".
 *
 * Props: carbon (CarbonComparison)
 */
export default function CarbonCallout({ carbon }) {
  if (!carbon) return null

  const rows = [
    { label: 'EV CO₂',      value: `${carbon.ev_co2_kg.toFixed(1)} kg`,   color: 'text-emerald-400' },
    { label: 'Petrol CO₂',  value: `${carbon.petrol_co2_kg.toFixed(1)} kg`, color: 'text-red-400'    },
    { label: 'CO₂ saved',   value: `${carbon.co2_saved_kg.toFixed(1)} kg`, color: 'text-emerald-300' },
    { label: 'EV cost',     value: `₹${carbon.ev_cost_inr.toFixed(0)}`,    color: 'text-slate-200'   },
    { label: 'Petrol cost', value: `₹${carbon.petrol_cost_inr.toFixed(0)}`, color: 'text-slate-200'  },
    { label: 'Cost saved',  value: `₹${carbon.cost_saved_inr.toFixed(0)}`, color: 'text-emerald-300' },
  ]

  return (
    <div className="bg-slate-900/60 rounded-lg p-4">
      <p className="text-xs text-slate-400 mb-3 font-medium">
        CO₂ &amp; cost comparison
        <span className="ml-1 text-slate-600">(illustrative)</span>
      </p>
      <div className="grid grid-cols-2 gap-2">
        {rows.map(r => (
          <div key={r.label} className="flex justify-between gap-2 text-xs">
            <span className="text-slate-500">{r.label}</span>
            <span className={r.color}>{r.value}</span>
          </div>
        ))}
      </div>
      <p className="text-xs text-slate-600 mt-3">
        Grid: 0.72 kg CO₂/kWh (CEA 2022). Petrol: 2.31 kg CO₂/L (IPCC).
        Electricity: ₹8/kWh (assumed). Petrol: ₹100/L at 15 km/L (assumed).
      </p>
    </div>
  )
}
