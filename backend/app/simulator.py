"""
Physics-informed EV energy consumption simulator.

Formulas follow standard automotive energy modeling literature:
- SAE J1634 / EPA UDDS-style decomposition of traction energy
- Rolling resistance: F_rr = C_rr * m * g  (typical C_rr ~ 0.01 for EV tires)
- Aerodynamic drag: F_aero = 0.5 * rho * Cd * A * v^2
- Gravitational potential: E = m * g * h
- Regenerative braking recovers a fraction of descent energy

Reference: Redelbach et al., "Energy consumption estimation for electric vehicles"
           IEEE Trans. Intelligent Transportation Systems; also standard texts
           on vehicle longitudinal dynamics (Gillespie, "Fundamentals of Vehicle Dynamics").
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Literal

import numpy as np

logger = logging.getLogger(__name__)

# Physical constants
RHO_AIR = 1.225  # kg/m³ at sea level, ~15°C
G = 9.81  # m/s²
C_ROLLING = 0.011  # dimensionless rolling resistance coefficient (EV tires)
MOTOR_EFFICIENCY = 0.92  # typical inverter + motor efficiency
REGEN_EFFICIENCY = 0.65  # fraction of descent energy recovered to battery
HVAC_BASE_KW = 1.5  # baseline HVAC draw at moderate temps
HVAC_EXTREME_KW = 3.5  # extra draw in hot/cold extremes

DrivingStyle = Literal["aggressive", "normal", "calm"]

DRIVING_STYLE_MULTIPLIER: dict[DrivingStyle, float] = {
    "aggressive": 1.18,  # hard acceleration, high regen braking variance
    "normal": 1.0,
    "calm": 0.90,  # eco driving, anticipatory braking
}


@dataclass
class EVSpecs:
    battery_capacity_kwh: float
    kerb_weight_kg: float
    drag_coefficient: float
    frontal_area_m2: float
    motor_power_kw: float
    claimed_range_km: float = 0.0


@dataclass
class TripConditions:
    distance_km: float
    avg_speed_kmh: float
    elevation_gain_m: float
    elevation_loss_m: float
    temperature_c: float
    driving_style: DrivingStyle = "normal"
    num_passengers: int = 1


@dataclass
class EnergyBreakdown:
    """Per-term energy consumption in kWh for one trip."""
    base_consumption_kwh: float
    aero_drag_kwh: float
    rolling_resistance_kwh: float
    elevation_gain_kwh: float
    regen_recovery_kwh: float  # negative = energy returned
    hvac_kwh: float
    style_multiplier: float
    noise_kwh: float
    total_energy_kwh: float


def base_consumption_term(
    motor_efficiency: float,
    weight_kg: float,
    distance_km: float,
) -> float:
    """
    Parasitic/auxiliary + drivetrain overhead scaled with mass and distance.
    Heavier vehicles need more energy for ancillary systems and inertia management.
    """
    # ~0.02 kWh/km baseline at reference mass 1400 kg, adjusted by efficiency
    ref_mass = 1400.0
    kwh_per_km = 0.02 * (weight_kg / ref_mass) / motor_efficiency
    return kwh_per_km * distance_km


def aero_drag_term(
    speed_kmh: float,
    drag_coefficient: float,
    frontal_area_m2: float,
    distance_km: float,
) -> float:
    """
    Aerodynamic drag power ~ v^3 integrated over time.
    E_aero = (0.5 * rho * Cd * A * v^2) * distance / efficiency
    Uses average speed as proxy for constant-speed cruise segment.
    """
    v_ms = max(speed_kmh, 5.0) / 3.6
    force_n = 0.5 * RHO_AIR * drag_coefficient * frontal_area_m2 * v_ms**2
    energy_j = force_n * (distance_km * 1000)
    return (energy_j / MOTOR_EFFICIENCY) / 3_600_000  # J -> kWh


def rolling_resistance_term(weight_kg: float, distance_km: float) -> float:
    """
    F_rr = C_rr * m * g; energy = F * d / efficiency.
    Dominates at city speeds; proportional to weight and distance.
    """
    force_n = C_ROLLING * weight_kg * G
    energy_j = force_n * (distance_km * 1000)
    return (energy_j / MOTOR_EFFICIENCY) / 3_600_000


def elevation_gain_term(weight_kg: float, elevation_gain_m: float) -> float:
    """
    Potential energy to climb: E = m * g * h / efficiency.
    Only positive gain consumes net energy.
    """
    if elevation_gain_m <= 0:
        return 0.0
    energy_j = weight_kg * G * elevation_gain_m
    return (energy_j / MOTOR_EFFICIENCY) / 3_600_000


def regen_recovery_term(weight_kg: float, elevation_loss_m: float) -> float:
    """
    Descent recovers energy via regen braking: E_regen = m * g * h * eta_regen.
    Returned as positive kWh subtracted from total consumption.
    """
    if elevation_loss_m <= 0:
        return 0.0
    energy_j = weight_kg * G * elevation_loss_m * REGEN_EFFICIENCY
    return energy_j / 3_600_000


def hvac_term(temperature_c: float, trip_duration_min: float) -> float:
    """
    Cabin thermal load depends on ambient temperature deviation from comfort (22°C).
    Hot Indian summers and cold hill-station mornings increase HVAC draw.
    """
    comfort_c = 22.0
    delta = abs(temperature_c - comfort_c)
    if delta < 5:
        power_kw = HVAC_BASE_KW * 0.3
    elif delta < 15:
        power_kw = HVAC_BASE_KW
    else:
        power_kw = HVAC_EXTREME_KW
    return power_kw * (trip_duration_min / 60.0)


def simulate_trip_energy(
    ev: EVSpecs,
    trip: TripConditions,
    *,
    add_noise: bool = True,
    rng: np.random.Generator | None = None,
) -> EnergyBreakdown:
    """
    Full physics-based energy model for a single trip.

    energy_kwh = base + aero + rolling + elevation_gain
                 - regen_recovery + hvac
    Then multiplied by driving_style_multiplier.
    Optional Gaussian noise captures real-world variability (traffic, wind, etc.).
    """
    rng = rng or np.random.default_rng()

    total_weight = ev.kerb_weight_kg + max(trip.num_passengers - 1, 0) * 75.0
    duration_min = (trip.distance_km / max(trip.avg_speed_kmh, 10.0)) * 60.0

    base = base_consumption_term(MOTOR_EFFICIENCY, total_weight, trip.distance_km)
    aero = aero_drag_term(
        trip.avg_speed_kmh, ev.drag_coefficient, ev.frontal_area_m2, trip.distance_km
    )
    rolling = rolling_resistance_term(total_weight, trip.distance_km)
    elev_gain = elevation_gain_term(total_weight, trip.elevation_gain_m)
    regen = regen_recovery_term(total_weight, trip.elevation_loss_m)
    hvac = hvac_term(trip.temperature_c, duration_min)

    subtotal = base + aero + rolling + elev_gain - regen + hvac
    style_mult = DRIVING_STYLE_MULTIPLIER.get(trip.driving_style, 1.0)
    energy = subtotal * style_mult

    noise = 0.0
    if add_noise:
        # ~8% coefficient of variation — models traffic, wind, tire wear, etc.
        noise = rng.normal(0.0, max(energy * 0.08, 0.05))
        energy += noise

    energy = max(energy, 0.01)

    return EnergyBreakdown(
        base_consumption_kwh=base,
        aero_drag_kwh=aero,
        rolling_resistance_kwh=rolling,
        elevation_gain_kwh=elev_gain,
        regen_recovery_kwh=regen,
        hvac_kwh=hvac,
        style_multiplier=style_mult,
        noise_kwh=noise,
        total_energy_kwh=energy,
    )


def energy_to_range_km(energy_kwh: float, battery_kwh: float, claimed_range_km: float) -> float:
    """
    Convert trip energy to achievable range on full battery.
    Uses proportional scaling: range = battery / (energy/distance).
    Caps unrealistic values near claimed range * 1.15.
    """
    if energy_kwh <= 0:
        return claimed_range_km
    consumption_per_km = energy_kwh  # normalized per-km when distance=1 in caller
    return min(battery_kwh / consumption_per_km, claimed_range_km * 1.15)


def simulate_range_km(
    ev: EVSpecs,
    trip: TripConditions,
    *,
    add_noise: bool = True,
    rng: np.random.Generator | None = None,
) -> tuple[float, EnergyBreakdown]:
    """
    Simulate real-world range (km on full charge) implied by trip conditions.
    Normalizes energy to per-km consumption then scales to battery capacity.
    """
    breakdown = simulate_trip_energy(ev, trip, add_noise=add_noise, rng=rng)
    per_km = breakdown.total_energy_kwh / max(trip.distance_km, 0.1)
    range_km = ev.battery_capacity_kwh / per_km
    # Soft cap — no EV exceeds ~115% of claimed range in real conditions
    if ev.claimed_range_km > 0:
        range_km = min(range_km, ev.claimed_range_km * 1.12)
    range_km = max(range_km, ev.claimed_range_km * 0.45 if ev.claimed_range_km else 50)
    return range_km, breakdown
