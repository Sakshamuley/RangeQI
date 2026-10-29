# RangeIQ Physics Model Assumptions

This document is generated from `backend/app/physics_config.py` (`PARAM_CATALOGUE`).
Every parameter used in the segment-based energy model is listed below.

> [!IMPORTANT]
> Parameters marked **assumed** were chosen from engineering judgement or literature
> ranges and have **not** been independently measured for the specific EV models in
> this dataset. Parameters marked **calibrated** were optimised in Phase 4 by
> multi-start L-BFGS-B against 12 Autocar India road-test calibration cases;
> their calibrated values are used in `PhysicsConfig` and the recommendation pipeline.

---

## Parameter Table

| Parameter | Nominal | Units | Distribution | Source | Type |
|-----------|---------|-------|--------------|--------|------|
| G | 9.81 | m/s² | constant | physical constant | published |
| Crr | 0.011 | dimensionless | N(0.011, 0.002) | [FAR16] Table 1; assumed for EV tyres in Indian conditions | assumed |
| eta_drive | 0.92 | fraction | N(0.92, 0.02) | [DF21] §3.1; typical BEV motor + inverter + reduction gear | assumed |
| eta_regen | **0.30** (calibrated; prior nominal 0.65) | fraction | N(0.30, 0.05) | [FAR16] §4; calibrated via L-BFGS-B on 12 Autocar India cases (Phase 4) | calibrated |
| inertia_factor | 1.05 | dimensionless | N(1.05, 0.02) | [G92] Ch. 2; assumed 5% rotational-mass penalty for BEV | assumed |
| v_wind_mps | 0.0 | m/s | N(0.0, 2.0) | assumed; no meteorological model included | assumed |
| P_aux_kw | **1.0** (calibrated; prior nominal 0.25) | kW | N(1.0, 0.10) | assumed; calibrated via L-BFGS-B on 12 Autocar India cases (Phase 4) | calibrated |
| hvac_base_kw | 0.20 | kW | N(0.20, 0.05) | assumed (cabin fan only at comfort temperature 15–28 °C) | assumed |
| hvac_heat_threshold_c | 15.0 | °C | constant | [YM15] Fig. 4; heating load onset below ~15 °C | assumed |
| hvac_cool_threshold_c | 28.0 | °C | constant | [YM15] Fig. 4; cooling load onset above ~28 °C | assumed |
| hvac_heat_coef_kw_per_c | 0.10 | kW/°C | N(0.10, 0.03) | assumed; direction per [YM15] | assumed |
| hvac_cool_coef_kw_per_c | 0.07 | kW/°C | N(0.07, 0.02) | assumed; direction per [YM15] | assumed |
| hvac_max_heat_kw | 5.0 | kW | constant | assumed; typical BEV heat pump / resistive heater limit | assumed |
| hvac_max_cool_kw | 3.5 | kW | constant | assumed; typical BEV AC compressor limit | assumed |
| stops_per_km_urban | 1.5 | stops/km | N(1.5, 0.5) | assumed; representative of Indian urban traffic (mixed signals + intersections) | assumed |
| stops_per_km_mixed | 0.5 | stops/km | N(0.5, 0.2) | assumed; mixed arterial / state-highway conditions | assumed |
| stops_per_km_highway | 0.05 | stops/km | N(0.05, 0.02) | assumed; expressway toll-plaza frequency | assumed |

---

## Driving Style Parameters

Driving style modifies speed and acceleration profiles, not a single multiplier on
total energy. This matters because aerodynamic drag scales with v², so speed
changes have a disproportionate effect on aero losses.

| Style | speed_factor | accel_factor | Effect |
|-------|-------------|-------------|--------|
| aggressive | 1.10 | 1.50 | Higher cruise speed (+10%) and harder acceleration (+50% kinetic energy per stop) |
| normal | 1.00 | 1.00 | Baseline |
| calm | 0.92 | 0.70 | Lower cruise speed (−8%) and anticipatory braking (−30% kinetic energy per stop) |

---

## HVAC Model

HVAC power is modelled as a function of ambient temperature, following the
direction established by Yuksel & Michalek 2015 [YM15]:

- **Heating** (ambient < 15 °C): P_hvac = hvac_base_kw + hvac_heat_coef_kw_per_c × (15 − T), capped at hvac_max_heat_kw
- **Comfort zone** (15 °C ≤ ambient ≤ 28 °C): P_hvac = hvac_base_kw (fan only)
- **Cooling** (ambient > 28 °C): P_hvac = hvac_base_kw + hvac_cool_coef_kw_per_c × (T − 28), capped at hvac_max_cool_kw

When `ac_on=False` (user-specified), P_hvac = 0 regardless of temperature.
The HVAC load is always sourced from the traction battery.

---

## Air Density

Computed from ambient temperature and altitude via the ISA barometric formula
(ideal gas approximation):

```
P = 101325 × (1 − 2.2557×10⁻⁵ × alt_m)^5.2559   [Pa]
ρ = P / (R_air × T_K)                              [kg/m³]
```

where R_air = 287.05 J/(kg·K) (specific gas constant for dry air) and
T_K = temperature_°C + 273.15.

---

## Stop-and-Go Correction

Traffic stops are classified by average speed (same thresholds as road class):

| Road class | Avg speed | stops_per_km |
|------------|-----------|-------------|
| Urban | < 50 km/h | 1.5 (assumed) |
| Mixed | 50–90 km/h | 0.5 (assumed) |
| Highway | > 90 km/h | 0.05 (assumed) |

**Assumption**: these values represent Indian traffic conditions and have
not been measured directly. They are documented here as assumed.

Kinetic energy budget per stop:

```
KE_per_stop = 0.5 × m_eff × v² × inertia_factor × accel_factor
n_stops = stops_per_km × distance_km
E_regen_recovered = n_stops × KE_per_stop × eta_regen
E_friction_loss   = n_stops × KE_per_stop × (1 − eta_regen) / eta_drive
```

The recovered portion reduces total energy draw (added to regen_wh_per_km).
The friction portion adds to inertia_wh_per_km.

---

## References

| Code | Full Citation |
|------|--------------|
| [FAR16] | Fiori, C., Ahn, K. & Rakha, H.A. (2016). Power-based electric vehicle energy consumption model: Model development and validation. *Applied Energy*, 168, 257–268. |
| [DF21] | Dollinger, C. & Fischerauer, G. (2021). Modelling and prediction of electric vehicle energy consumption. *Energies*, 14(18), 5804. |
| [G92] | Gillespie, T.D. (1992). *Fundamentals of Vehicle Dynamics*. SAE International. |
| [ARAI] | ARAI (2019). AIS-040: Method of Measuring the Range of Battery Operated Electric Vehicles. Automotive Research Association of India (MIDC test cycle). |
| [YM15] | Yuksel, T. & Michalek, J.J. (2015). Effects of regional temperature on electric vehicle efficiency, range, and emissions in the United States. *Environmental Science & Technology*, 49(6), 3974–3980. |

---

> [!NOTE]
> Last updated: Phase 8 (cleanup). Calibrated values for `eta_regen` (0.30) and
> `p_aux_kw` (1.0 kW) were set in Phase 4 via multi-start L-BFGS-B optimisation
> against 12 Autocar India road-test calibration cases.
> Validation MAPE after calibration: 9.2% (test set: 9.1%). Spearman ρ = 0.741 (p = 0.001).
