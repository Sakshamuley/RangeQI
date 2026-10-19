"""Carbon and cost comparison: EV vs petrol."""

from __future__ import annotations

# India grid emission factor (~0.82 kg CO2/kWh) — CEA/industry estimates
INDIA_GRID_CO2_KG_PER_KWH = 0.82
# Average petrol car ~120 g CO2/km
PETROL_CO2_KG_PER_KM = 0.120
# Approximate fuel and electricity costs in INR (demo values)
PETROL_PRICE_INR_PER_L = 102.0
PETROL_EFFICIENCY_KM_PER_L = 14.0
ELECTRICITY_PRICE_INR_PER_KWH = 8.5


def compute_carbon_comparison(
    trip_energy_kwh: float,
    trip_distance_km: float,
) -> dict:
    """
    Compare EV trip emissions vs equivalent petrol trip.
    Returns CO2 saved and cost comparison in INR.
    """
    if trip_distance_km <= 0:
        trip_distance_km = 1.0

    ev_co2_kg = trip_energy_kwh * INDIA_GRID_CO2_KG_PER_KWH
    petrol_co2_kg = trip_distance_km * PETROL_CO2_KG_PER_KM
    co2_saved_kg = round(petrol_co2_kg - ev_co2_kg, 3)

    ev_cost_inr = round(trip_energy_kwh * ELECTRICITY_PRICE_INR_PER_KWH, 2)
    petrol_liters = trip_distance_km / PETROL_EFFICIENCY_KM_PER_L
    petrol_cost_inr = round(petrol_liters * PETROL_PRICE_INR_PER_L, 2)
    cost_saved_inr = round(petrol_cost_inr - ev_cost_inr, 2)

    return {
        "ev_co2_kg": round(ev_co2_kg, 3),
        "petrol_co2_kg": round(petrol_co2_kg, 3),
        "co2_saved_kg": co2_saved_kg,
        "ev_cost_inr": ev_cost_inr,
        "petrol_cost_inr": petrol_cost_inr,
        "cost_saved_inr": cost_saved_inr,
        "trip_distance_km": round(trip_distance_km, 2),
        "trip_energy_kwh": round(trip_energy_kwh, 3),
        "grid_factor_kg_per_kwh": INDIA_GRID_CO2_KG_PER_KWH,
    }
