"""
Physics model configuration — all parameters in one place.

Every parameter has:
  nominal      : the value used in production predictions
  distribution : uncertainty form for Monte Carlo sampling (Phase 5)
  source       : peer-reviewed reference or "assumed"
  units        : SI units
  description  : one-line explanation

"assumed" means the value was chosen from engineering judgement or
literature ranges and has NOT been independently measured for the
specific EV models in this dataset.

Calibrated parameters (P_aux_kw, eta_regen) are adjusted in Phase 4
to minimise error on the five calibration cases; their calibrated values
replace the prior nominals here after Phase 4.

References accepted in this project:
  [FAR16]  Fiori, Ahn & Rakha 2016, Appl. Energy 168
  [DF21]   Dollinger & Fischerauer 2021, Energies 14(18) 5804
  [G92]    Gillespie, "Fundamentals of Vehicle Dynamics"
  [ARAI]   ARAI AIS-040 (MIDC test cycle)
  [YM15]   Yuksel & Michalek 2015 (temperature effect on EV range)
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Param:
    """One physics parameter entry."""
    nominal: float
    distribution: str    # e.g. "N(mu, sigma)" / "U(a, b)" / "constant"
    source: str
    units: str
    description: str
    param_type: str = "assumed"  # "published" | "assumed" | "calibrated"


@dataclass(frozen=True)
class PhysicsConfig:
    """Central config object passed to the simulator."""

    # ── Vehicle dynamics ──────────────────────────────────────────────────────
    crr: float = 0.011
    eta_drive: float = 0.92
    eta_regen: float = 0.65
    inertia_factor: float = 1.05
    v_wind_mps: float = 0.0

    # ── Auxiliary / HVAC ─────────────────────────────────────────────────────
    p_aux_kw: float = 0.25
    hvac_base_kw: float = 0.20
    hvac_heat_threshold_c: float = 15.0
    hvac_cool_threshold_c: float = 28.0
    hvac_heat_coef_kw_per_c: float = 0.10
    hvac_cool_coef_kw_per_c: float = 0.07
    hvac_max_heat_kw: float = 5.0
    hvac_max_cool_kw: float = 3.5

    # ── Stop-and-go (assumed; documented as such) ─────────────────────────────
    stops_per_km_urban: float = 1.5
    stops_per_km_mixed: float = 0.5
    stops_per_km_highway: float = 0.05

    # Road class speed thresholds
    urban_speed_threshold_kmh: float = 50.0
    highway_speed_threshold_kmh: float = 90.0


# ── Default config instance ───────────────────────────────────────────────────
DEFAULT_CONFIG = PhysicsConfig()


# ── Documented parameter catalogue (for ASSUMPTIONS.md) ──────────────────────
PARAM_CATALOGUE: dict[str, Param] = {
    "G": Param(
        nominal=9.81,
        distribution="constant",
        source="physical constant",
        units="m/s²",
        description="Standard acceleration due to gravity",
        param_type="published",
    ),
    "Crr": Param(
        nominal=0.011,
        distribution="N(0.011, 0.002)",
        source="[FAR16] Table 1; assumed for EV tyres in Indian conditions",
        units="dimensionless",
        description="Rolling resistance coefficient",
        param_type="assumed",
    ),
    "eta_drive": Param(
        nominal=0.92,
        distribution="N(0.92, 0.02)",
        source="[DF21] §3.1; typical BEV motor + inverter + reduction gear",
        units="fraction",
        description="Drivetrain efficiency (motor → wheel)",
        param_type="assumed",
    ),
    "eta_regen": Param(
        nominal=0.65,
        distribution="N(0.65, 0.05)",
        source="[FAR16] §4; calibrated on five cases in Phase 4",
        units="fraction",
        description="Regenerative braking efficiency (wheel → battery)",
        param_type="calibrated",
    ),
    "inertia_factor": Param(
        nominal=1.05,
        distribution="N(1.05, 0.02)",
        source="[G92] Ch. 2; assumed 5% rotational-mass penalty for BEV",
        units="dimensionless",
        description="Effective mass factor for rotational inertia (wheels + motor rotor)",
        param_type="assumed",
    ),
    "v_wind_mps": Param(
        nominal=0.0,
        distribution="N(0.0, 2.0)",
        source="assumed; no meteorological model included",
        units="m/s",
        description="Headwind component (positive = into wind)",
        param_type="assumed",
    ),
    "P_aux_kw": Param(
        nominal=0.25,
        distribution="N(0.25, 0.10)",
        source="assumed; calibrated on five cases in Phase 4",
        units="kW",
        description="Constant auxiliary power (lights, cooling pump, 12V electronics)",
        param_type="calibrated",
    ),
    "hvac_base_kw": Param(
        nominal=0.20,
        distribution="N(0.20, 0.05)",
        source="assumed (cabin fan only at comfort temperature 15–28 °C)",
        units="kW",
        description="HVAC power when ambient is in comfortable range (fan only)",
        param_type="assumed",
    ),
    "hvac_heat_threshold_c": Param(
        nominal=15.0,
        distribution="constant",
        source="[YM15] Fig. 4; heating load onset below ~15 °C",
        units="°C",
        description="Ambient temperature below which cabin heating is activated",
        param_type="assumed",
    ),
    "hvac_cool_threshold_c": Param(
        nominal=28.0,
        distribution="constant",
        source="[YM15] Fig. 4; cooling load onset above ~28 °C",
        units="°C",
        description="Ambient temperature above which AC compressor is activated",
        param_type="assumed",
    ),
    "hvac_heat_coef_kw_per_c": Param(
        nominal=0.10,
        distribution="N(0.10, 0.03)",
        source="assumed; direction per [YM15]",
        units="kW/°C",
        description="Additional heating power per °C below hvac_heat_threshold_c",
        param_type="assumed",
    ),
    "hvac_cool_coef_kw_per_c": Param(
        nominal=0.07,
        distribution="N(0.07, 0.02)",
        source="assumed; direction per [YM15]",
        units="kW/°C",
        description="Additional AC power per °C above hvac_cool_threshold_c",
        param_type="assumed",
    ),
    "hvac_max_heat_kw": Param(
        nominal=5.0,
        distribution="constant",
        source="assumed; typical BEV heat pump / resistive heater limit",
        units="kW",
        description="Maximum HVAC heating power (hard cap)",
        param_type="assumed",
    ),
    "hvac_max_cool_kw": Param(
        nominal=3.5,
        distribution="constant",
        source="assumed; typical BEV AC compressor limit",
        units="kW",
        description="Maximum HVAC cooling power (hard cap)",
        param_type="assumed",
    ),
    "stops_per_km_urban": Param(
        nominal=1.5,
        distribution="N(1.5, 0.5)",
        source="assumed; representative of Indian urban traffic (mixed signals + intersections)",
        units="stops/km",
        description="Traffic stops per km on urban road segments (avg_speed < 50 km/h)",
        param_type="assumed",
    ),
    "stops_per_km_mixed": Param(
        nominal=0.5,
        distribution="N(0.5, 0.2)",
        source="assumed; mixed arterial / state-highway conditions",
        units="stops/km",
        description="Traffic stops per km on mixed road segments (50–90 km/h)",
        param_type="assumed",
    ),
    "stops_per_km_highway": Param(
        nominal=0.05,
        distribution="N(0.05, 0.02)",
        source="assumed; expressway toll-plaza frequency",
        units="stops/km",
        description="Traffic stops per km on highway segments (> 90 km/h)",
        param_type="assumed",
    ),
}
