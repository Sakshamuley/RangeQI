"""
Segment-based physics model for EV energy consumption.

Theory — per segment i with length L (m), gradient θ (rad), speed v (m/s):

  F_aero    = 0.5 · ρ · Cd · A · (v + v_wind)²          [N]
  F_rolling = Crr · m · g · cos(θ)                        [N]  [FAR16]
  F_grade   = m · g · sin(θ)                              [N]  [G92]
  F_total   = F_aero + F_rolling + F_grade

  P_wheel = F_total · v                                    [W]
  P_battery = P_wheel / η_drive    if P_wheel ≥ 0  (driving)
              P_wheel · η_regen    if P_wheel < 0  (regen)  [FAR16]
  plus: P_aux (constant auxiliary) + P_hvac(T_ambient)    [YM15]

  E_segment = (P_battery + P_aux + P_hvac) · (L / v)     [J]

Air density is computed from ambient temperature + altitude via the
ISA barometric formula (ideal gas law approximation).

HVAC direction per Yuksel & Michalek 2015 [YM15]:
  heating below ~15 °C → reduces range
  cooling above ~28 °C → reduces range

Stop-and-go correction (urban cycles):
  KE_per_stop = 0.5 · m · v² · inertia_factor
  E_stop_net  = n_stops · KE_per_stop · (1 − η_regen) / η_drive

No range clamps are applied.  Range = usable_battery_kwh / (Wh/km × 10⁻³).

References:
  [FAR16] Fiori, Ahn & Rakha 2016, Appl. Energy 168
  [DF21]  Dollinger & Fischerauer 2021, Energies 14(18) 5804
  [G92]   Gillespie, "Fundamentals of Vehicle Dynamics"
  [ARAI]  ARAI AIS-040 (MIDC test cycle)
  [YM15]  Yuksel & Michalek 2015 (temperature effect on EV range)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Literal

from app.physics_config import DEFAULT_CONFIG, PhysicsConfig

G = 9.81        # m/s²
R_AIR = 287.05  # J/(kg·K) — specific gas constant for dry air

DrivingStyle = Literal["aggressive", "normal", "calm"]

# Speed factor: scales avg_speed, affecting aero (v²) and time-based loads.
# Justification: aggressive drivers choose higher speeds; calm drivers ease off.
STYLE_SPEED_FACTOR: dict[DrivingStyle, float] = {
    "aggressive": 1.10,
    "normal":     1.00,
    "calm":       0.92,
}
# Acceleration factor: scales effective acceleration for stop-and-go events.
STYLE_ACCEL_FACTOR: dict[DrivingStyle, float] = {
    "aggressive": 1.50,  # harder acceleration between stops → more inertia energy
    "normal":     1.00,
    "calm":       0.70,  # anticipatory braking, softer re-acceleration
}


# ── Data types ────────────────────────────────────────────────────────────────

@dataclass
class EVSpecs:
    """
    EV physics parameters required by the energy model.
    Use usable_battery_kwh for range calculation (not nominal battery_capacity_kwh).
    """
    battery_capacity_kwh: float        # nominal (gross) kWh
    usable_battery_kwh: float          # net usable kWh (default 0.95 × nominal)
    kerb_weight_kg: float
    drag_coefficient: float            # Cd
    frontal_area_m2: float             # A
    motor_power_kw: float

    @classmethod
    def from_ev_model(cls, ev) -> "EVSpecs":
        """Build EVSpecs from a SQLAlchemy EVModel row."""
        usable = ev.usable_battery_kwh or ev.battery_capacity_kwh * 0.95
        return cls(
            battery_capacity_kwh=ev.battery_capacity_kwh,
            usable_battery_kwh=usable,
            kerb_weight_kg=ev.kerb_weight_kg,
            drag_coefficient=ev.drag_coefficient,
            frontal_area_m2=ev.frontal_area_m2,
            motor_power_kw=ev.motor_power_kw,
        )


@dataclass
class SegmentData:
    """One road segment from OSRM geometry interpolation (Phase 2)."""
    gradient: float   # dimensionless (positive = uphill): Δelev / distance
    length_m: float   # metres


@dataclass
class EnergyBreakdown:
    """
    Per-term energy consumption in Wh/km, summed over the full trip.

    regen_wh_per_km is always ≤ 0 (energy returned to battery).
    gradient_wh_per_km is positive uphill, negative downhill (before regen).
    total_wh_per_km = aero + rolling + gradient + inertia + regen + auxiliary + hvac
    """
    aero_wh_per_km: float
    rolling_wh_per_km: float
    gradient_wh_per_km: float
    inertia_wh_per_km: float
    regen_wh_per_km: float
    auxiliary_wh_per_km: float
    hvac_wh_per_km: float
    total_wh_per_km: float
    total_trip_wh: float
    trip_distance_km: float


# ── Physical sub-models ───────────────────────────────────────────────────────

def air_density_kg_m3(temp_c: float, altitude_m: float = 0.0) -> float:
    """
    Air density from ambient temperature and altitude.
    Uses the ISA barometric pressure formula (troposphere).
    """
    T_K = temp_c + 273.15
    # ISA pressure at altitude (valid to ~11 km)
    P_Pa = 101_325.0 * (1.0 - 2.2557e-5 * altitude_m) ** 5.2559
    return P_Pa / (R_AIR * T_K)


def hvac_power_kw(
    ambient_c: float,
    cfg: PhysicsConfig = DEFAULT_CONFIG,
) -> float:
    """
    HVAC power draw as a function of ambient temperature.

    Heating regime (< hvac_heat_threshold_c):
      Resistive heater or heat pump; power increases with cold severity.
      Direction: lower T → higher load → less range  [YM15]

    Cooling regime (> hvac_cool_threshold_c):
      AC compressor; power increases with heat severity.
      Direction: higher T → higher load → less range  [YM15]

    Magnitudes are assumed (see ASSUMPTIONS.md).
    """
    if ambient_c < cfg.hvac_heat_threshold_c:
        delta = cfg.hvac_heat_threshold_c - ambient_c
        return min(
            cfg.hvac_base_kw + delta * cfg.hvac_heat_coef_kw_per_c,
            cfg.hvac_max_heat_kw,
        )
    if ambient_c > cfg.hvac_cool_threshold_c:
        delta = ambient_c - cfg.hvac_cool_threshold_c
        return min(
            cfg.hvac_base_kw + delta * cfg.hvac_cool_coef_kw_per_c,
            cfg.hvac_max_cool_kw,
        )
    return cfg.hvac_base_kw  # comfort zone — fan only


def _road_class(speed_kmh: float, cfg: PhysicsConfig) -> str:
    """Classify road type from average speed."""
    if speed_kmh < cfg.urban_speed_threshold_kmh:
        return "urban"
    if speed_kmh < cfg.highway_speed_threshold_kmh:
        return "mixed"
    return "highway"


def _stops_per_km(speed_kmh: float, cfg: PhysicsConfig) -> float:
    rc = _road_class(speed_kmh, cfg)
    return {
        "urban":   cfg.stops_per_km_urban,
        "mixed":   cfg.stops_per_km_mixed,
        "highway": cfg.stops_per_km_highway,
    }[rc]


# ── Core simulator ────────────────────────────────────────────────────────────

def simulate_energy(
    ev: EVSpecs,
    segments: list[SegmentData],
    avg_speed_kmh: float,
    temperature_c: float,
    driving_style: DrivingStyle = "normal",
    num_passengers: int = 1,
    ac_on: bool = True,
    altitude_m: float = 0.0,
    config: PhysicsConfig | None = None,
) -> EnergyBreakdown:
    """
    Segment-based energy model.  No range clamps are applied here.

    Parameters
    ----------
    segments    : road geometry from OSRM (Phase 2), each with gradient and length
    avg_speed_kmh : overall average speed (scaled by driving style)
    altitude_m  : mean altitude for air-density correction (0 = sea level)
    config      : PhysicsConfig; uses DEFAULT_CONFIG when None
    """
    cfg = config or DEFAULT_CONFIG

    speed_factor = STYLE_SPEED_FACTOR[driving_style]
    accel_factor = STYLE_ACCEL_FACTOR[driving_style]
    v_kmh = avg_speed_kmh * speed_factor
    v_mps = max(v_kmh / 3.6, 0.5)   # guard against zero

    total_mass_kg = ev.kerb_weight_kg + num_passengers * 75.0
    rho = air_density_kg_m3(temperature_c, altitude_m)
    CdA = ev.drag_coefficient * ev.frontal_area_m2

    p_hvac_kw = hvac_power_kw(temperature_c, cfg) if ac_on else 0.0
    p_aux_kw = cfg.p_aux_kw

    # Joule accumulators (positive = energy consumed from battery)
    J_aero = 0.0
    J_roll = 0.0
    J_grade = 0.0   # positive uphill, negative downhill-with-drive
    J_regen = 0.0   # always ≤ 0
    J_aux = 0.0
    J_hvac = 0.0

    for seg in segments:
        L = seg.length_m
        if L <= 0.0:
            continue
        theta = math.atan(seg.gradient)   # convert grade to angle
        t_s = L / v_mps                   # time to traverse segment (s)

        F_aero = 0.5 * rho * CdA * (v_mps + cfg.v_wind_mps) ** 2
        F_roll = cfg.crr * total_mass_kg * G * math.cos(theta)
        F_grade = total_mass_kg * G * math.sin(theta)
        F_total = F_aero + F_roll + F_grade

        if F_total >= 0.0:
            # Motor driving: attribute each force to battery draw
            J_aero  += F_aero  * L / cfg.eta_drive
            J_roll  += F_roll  * L / cfg.eta_drive
            J_grade += F_grade * L / cfg.eta_drive   # can be negative (downhill-but-driving)
        else:
            # Regen: motor acts as generator; aero+rolling act as passive brakes
            J_regen += F_total * L * cfg.eta_regen   # negative

        # Auxiliary and HVAC always draw from battery (even during regen)
        J_aux  += p_aux_kw  * 1_000.0 * t_s   # kW→W, then ×s = J
        J_hvac += p_hvac_kw * 1_000.0 * t_s

    # ── Stop-and-go inertia correction ────────────────────────────────────────
    # Kinetic energy budget for traffic stops.
    # Assumption: documented in ASSUMPTIONS.md
    total_distance_m = sum(s.length_m for s in segments)
    distance_km = max(total_distance_m / 1000.0, 1e-6)

    spk = _stops_per_km(v_kmh, cfg)
    ke_per_stop_J = (
        0.5 * total_mass_kg * v_mps ** 2 * cfg.inertia_factor * accel_factor
    )
    n_stops = spk * distance_km
    total_ke_J = n_stops * ke_per_stop_J

    # Portion recovered by regen, portion lost as heat/brake friction
    J_regen -= total_ke_J * cfg.eta_regen           # additional regen credit (negative)
    J_inertia = total_ke_J * (1.0 - cfg.eta_regen) / cfg.eta_drive  # net loss

    # ── Convert to Wh/km ─────────────────────────────────────────────────────
    def to_wh_km(joules: float) -> float:
        return joules / 3_600.0 / distance_km

    wh_aero     = to_wh_km(J_aero)
    wh_roll     = to_wh_km(J_roll)
    wh_grade    = to_wh_km(J_grade)
    wh_regen    = to_wh_km(J_regen)
    wh_inertia  = to_wh_km(J_inertia)
    wh_aux      = to_wh_km(J_aux)
    wh_hvac     = to_wh_km(J_hvac)

    total_wh_km = wh_aero + wh_roll + wh_grade + wh_regen + wh_inertia + wh_aux + wh_hvac

    return EnergyBreakdown(
        aero_wh_per_km=round(wh_aero, 3),
        rolling_wh_per_km=round(wh_roll, 3),
        gradient_wh_per_km=round(wh_grade, 3),
        inertia_wh_per_km=round(wh_inertia, 3),
        regen_wh_per_km=round(wh_regen, 3),
        auxiliary_wh_per_km=round(wh_aux, 3),
        hvac_wh_per_km=round(wh_hvac, 3),
        total_wh_per_km=round(total_wh_km, 3),
        total_trip_wh=round(total_wh_km * distance_km, 3),
        trip_distance_km=round(distance_km, 4),
    )


def range_km(ev: EVSpecs, wh_per_km: float) -> float:
    """
    Predicted range in km.
    Range = usable_battery_kwh / (wh_per_km / 1000)
    No clamps applied.
    """
    if wh_per_km <= 0.0:
        return float("inf")
    return ev.usable_battery_kwh * 1_000.0 / wh_per_km


# ── Convenience: build segments from elevation totals ─────────────────────────

def segments_from_elevation_totals(
    distance_km: float,
    elevation_gain_m: float,
    elevation_loss_m: float,
    n_segments: int = 20,
) -> list[SegmentData]:
    """
    Create a representative segment list when per-segment OSRM data is not
    available (e.g., validation scenarios and tests).

    Alternates between climbing (gain/n) and descending (loss/n) segments
    so that elevation gain and loss are both represented.
    """
    seg_len_m = distance_km * 1000.0 / n_segments
    climb_per = elevation_gain_m / max(n_segments // 2, 1)
    descend_per = elevation_loss_m / max(n_segments // 2, 1)

    segments: list[SegmentData] = []
    for i in range(n_segments):
        if i % 2 == 0:
            g = climb_per / seg_len_m if seg_len_m > 0 else 0.0
        else:
            g = -descend_per / seg_len_m if seg_len_m > 0 else 0.0
        segments.append(SegmentData(gradient=g, length_m=seg_len_m))
    return segments


def simulate_energy_simple(
    ev: EVSpecs,
    distance_km: float,
    elevation_gain_m: float,
    elevation_loss_m: float,
    avg_speed_kmh: float,
    temperature_c: float,
    driving_style: DrivingStyle = "normal",
    num_passengers: int = 1,
    ac_on: bool = True,
    altitude_m: float = 0.0,
    config: PhysicsConfig | None = None,
) -> EnergyBreakdown:
    """
    Convenience wrapper when per-segment OSRM data is unavailable.
    Internally creates a representative segment list.
    """
    segs = segments_from_elevation_totals(
        distance_km, elevation_gain_m, elevation_loss_m
    )
    return simulate_energy(
        ev, segs, avg_speed_kmh, temperature_c,
        driving_style=driving_style,
        num_passengers=num_passengers,
        ac_on=ac_on,
        altitude_m=altitude_m,
        config=config,
    )
