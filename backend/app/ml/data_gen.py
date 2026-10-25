"""
Monte Carlo training data generator for the RangeIQ ML emulator.

Each row represents one (EV, trip) scenario.  For each scenario, the physics
model is evaluated N_MC_SAMPLES times with randomly drawn uncertain parameters
(Crr, eta_drive, eta_regen, p_aux, HVAC scale, headwind, usable battery
fraction).  The p10/p50/p90 of the resulting Wh/km values become the targets
for the quantile-regression ML layer.

IMPORTANT: The p10–p90 band reflects assumed parameter uncertainty ONLY.
It does NOT represent measurement error or road-test validation.  Accuracy
against reality is quantified separately in validation_results.json.

The ML layer is a fast emulator and uncertainty estimator of the physics
model.  It is trained on simulated data.

References:
  [FAR16] Fiori, Ahn & Rakha 2016, Appl. Energy 168
  [DF21]  Dollinger & Fischerauer 2021, Energies 14(18) 5804
  [G92]   Gillespie, "Fundamentals of Vehicle Dynamics"
  [YM15]  Yuksel & Michalek 2015 (temperature effect on EV range)
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from app.physics_config import PhysicsConfig
from app.simulator import EVSpecs, simulate_energy_simple
from app.validation import EV_SPECS  # canonical physics specs (Phase 4)

logger = logging.getLogger(__name__)

OUTPUT_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "training_data.csv"

# ── Generation constants ──────────────────────────────────────────────────────
N_TRIPS_PER_EV = 300     # trip scenarios per EV
N_MC_SAMPLES   = 40      # physics evaluations per trip (for p10/p50/p90)
RANDOM_SEED    = 42

# ── Terrain classes (assumed; documented in ASSUMPTIONS.md) ──────────────────
# mean absolute gradient m/km: U(lo, hi)
TERRAIN_CLASSES = {
    "flat":    (0.0,  6.0),
    "rolling": (6.0, 18.0),
    "hilly":   (18.0, 45.0),
}
TERRAIN_PROBS = [0.45, 0.35, 0.20]   # assumed Indian route distribution

# ── Road class speed ranges km/h (assumed) ────────────────────────────────────
ROAD_CLASS_SPEED = {
    "urban":   (15,  50),
    "mixed":   (45,  85),
    "highway": (75, 120),
}
ROAD_CLASS_PROBS = [0.30, 0.40, 0.30]   # assumed

# ── Season temperature ranges °C (India) ─────────────────────────────────────
SEASON_TEMP_RANGE = {
    "winter":  ( 8, 22),
    "spring":  (20, 38),
    "monsoon": (22, 32),
    "autumn":  (18, 30),
}
SEASON_PROBS = [0.20, 0.30, 0.30, 0.20]

# ── Calibrated parameter centres (Phase 4) ───────────────────────────────────
# These replace the prior nominals as the sampling centres.
CALIB_P_AUX_KW  = 1.00   # calibrated; prior was 0.25
CALIB_ETA_REGEN = 0.30   # calibrated; prior was 0.65

# Usable battery fraction: U(0.90, 0.97) — assumed
USABLE_FRAC_LO = 0.90
USABLE_FRAC_HI = 0.97


def _sample_config(rng: np.random.Generator) -> PhysicsConfig:
    """
    Sample a PhysicsConfig from the uncertain-parameter distributions.
    Centres are the Phase-4 calibrated values; spreads from PARAM_CATALOGUE.
    The resulting spread is the sole source of the p10–p90 band.
    """
    crr       = float(np.clip(rng.normal(0.011, 0.002),         0.005, 0.025))
    eta_drive = float(np.clip(rng.normal(0.92,  0.02),          0.80,  0.99))
    eta_regen = float(np.clip(rng.normal(CALIB_ETA_REGEN, 0.05), 0.10, 0.60))
    p_aux_kw  = float(np.clip(rng.normal(CALIB_P_AUX_KW,  0.15), 0.30, 2.00))
    v_wind    = float(rng.normal(0.0, 2.0))
    hvac_s    = float(np.clip(rng.uniform(0.70, 1.30), 0.50, 1.80))

    return PhysicsConfig(
        crr=crr,
        eta_drive=eta_drive,
        eta_regen=eta_regen,
        p_aux_kw=p_aux_kw,
        v_wind_mps=v_wind,
        hvac_base_kw=round(0.20 * hvac_s, 4),
        hvac_heat_coef_kw_per_c=round(0.10 * hvac_s, 4),
        hvac_cool_coef_kw_per_c=round(0.07 * hvac_s, 4),
    )


def _sample_trip(rng: np.random.Generator) -> dict:
    """Sample one trip scenario (EV-independent features)."""
    # Distance: log-uniform 5–500 km
    distance_km = float(np.exp(rng.uniform(np.log(5), np.log(500))))

    # Terrain
    terrain = str(rng.choice(list(TERRAIN_CLASSES), p=TERRAIN_PROBS))
    grad_lo, grad_hi = TERRAIN_CLASSES[terrain]
    mean_abs_gradient_mpkm = float(rng.uniform(grad_lo, grad_hi))

    # Elevation: gain ≈ half distance × gradient; loss correlated with gain
    gain_per_km = mean_abs_gradient_mpkm * 0.5
    gain_m  = gain_per_km * distance_km
    loss_m  = gain_m * float(rng.uniform(0.70, 1.00))   # net elev change small

    # Road class and speed
    road_class = str(rng.choice(list(ROAD_CLASS_SPEED), p=ROAD_CLASS_PROBS))
    spd_lo, spd_hi = ROAD_CLASS_SPEED[road_class]
    avg_speed_kmh = float(rng.uniform(spd_lo, spd_hi))
    if road_class == "highway":
        highway_fraction = 1.0
    elif road_class == "urban":
        highway_fraction = 0.0
    else:
        highway_fraction = float(rng.uniform(0.10, 0.50))

    # Temperature (season + Gaussian noise)
    season = str(rng.choice(list(SEASON_TEMP_RANGE), p=SEASON_PROBS))
    t_lo, t_hi = SEASON_TEMP_RANGE[season]
    temperature_c = float(np.clip(rng.uniform(t_lo, t_hi) + rng.normal(0, 2), -5, 48))

    # Passengers 1–5
    num_passengers = int(rng.integers(1, 6))

    # AC: forced on >30 °C, forced off <10 °C, else random
    if temperature_c > 30.0:
        ac_on = 1
    elif temperature_c < 10.0:
        ac_on = 0
    else:
        ac_on = int(rng.random() > 0.35)

    # Driving style
    driving_style = str(rng.choice(
        ["aggressive", "normal", "calm"], p=[0.20, 0.55, 0.25]
    ))

    return {
        "distance_km":            round(distance_km, 3),
        "terrain":                terrain,
        "mean_gradient_mpkm":     round(mean_abs_gradient_mpkm, 3),
        "elevation_gain_per_km":  round(gain_per_km, 4),
        "elevation_loss_per_km":  round(loss_m / distance_km, 4),
        "elevation_gain_m":       round(gain_m, 2),
        "elevation_loss_m":       round(loss_m, 2),
        "avg_speed_kmh":          round(avg_speed_kmh, 2),
        "highway_fraction":       round(highway_fraction, 3),
        "road_class":             road_class,
        "temperature_c":          round(temperature_c, 2),
        "season":                 season,
        "num_passengers":         num_passengers,
        "ac_on":                  ac_on,
        "driving_style":          driving_style,
        "driving_style_code":     {"calm": 0, "normal": 1, "aggressive": 2}[driving_style],
    }


def generate(
    n_trips: int = N_TRIPS_PER_EV,
    n_mc: int = N_MC_SAMPLES,
    seed: int = RANDOM_SEED,
    output_path: Path | None = None,
) -> pd.DataFrame:
    """
    Generate the Monte Carlo training dataset.

    For each (EV, trip) pair runs n_mc physics simulations with sampled
    uncertain parameters.  Records p10/p50/p90 Wh/km as targets.
    """
    rng = np.random.default_rng(seed)
    rows: list[dict] = []

    ev_names = list(EV_SPECS.keys())
    logger.info(
        "Generating %d EVs × %d trips × %d MC samples = %d physics calls …",
        len(ev_names), n_trips, n_mc, len(ev_names) * n_trips * n_mc,
    )

    for ev_name, spec in EV_SPECS.items():
        for _ in range(n_trips):
            trip = _sample_trip(rng)

            # --- Monte Carlo over uncertain physics parameters ---
            wh_samples: list[float] = []
            for _ in range(n_mc):
                cfg = _sample_config(rng)
                usable_frac = float(rng.uniform(USABLE_FRAC_LO, USABLE_FRAC_HI))
                ev = EVSpecs(
                    battery_capacity_kwh=spec["battery_capacity_kwh"],
                    usable_battery_kwh=spec["battery_capacity_kwh"] * usable_frac,
                    kerb_weight_kg=spec["kerb_weight_kg"],
                    drag_coefficient=spec["drag_coefficient"],
                    frontal_area_m2=spec["frontal_area_m2"],
                    motor_power_kw=spec["motor_power_kw"],
                )
                res = simulate_energy_simple(
                    ev=ev,
                    distance_km=trip["distance_km"],
                    elevation_gain_m=trip["elevation_gain_m"],
                    elevation_loss_m=trip["elevation_loss_m"],
                    avg_speed_kmh=trip["avg_speed_kmh"],
                    temperature_c=trip["temperature_c"],
                    driving_style=trip["driving_style"],      # type: ignore[arg-type]
                    num_passengers=trip["num_passengers"],
                    ac_on=bool(trip["ac_on"]),
                    config=cfg,
                )
                wh_samples.append(res.total_wh_per_km)

            arr = np.array(wh_samples)
            p10, p50, p90 = np.percentile(arr, [10, 50, 90])

            rows.append({
                # EV physics features (no claimed_range_km)
                "ev_name":               ev_name,
                "battery_capacity_kwh":  spec["battery_capacity_kwh"],
                "usable_battery_kwh":    spec["usable_battery_kwh"],
                "kerb_weight_kg":        spec["kerb_weight_kg"],
                "drag_coefficient":      spec["drag_coefficient"],
                "frontal_area_m2":       spec["frontal_area_m2"],
                "motor_power_kw":        spec["motor_power_kw"],
                # Trip features
                "distance_km":           trip["distance_km"],
                "mean_gradient_mpkm":    trip["mean_gradient_mpkm"],
                "elevation_gain_per_km": trip["elevation_gain_per_km"],
                "elevation_loss_per_km": trip["elevation_loss_per_km"],
                "avg_speed_kmh":         trip["avg_speed_kmh"],
                "highway_fraction":      trip["highway_fraction"],
                "temperature_c":         trip["temperature_c"],
                "num_passengers":        trip["num_passengers"],
                "ac_on":                 trip["ac_on"],
                "driving_style_code":    trip["driving_style_code"],
                # Targets
                "p10_wh_per_km":  round(float(p10), 4),
                "p50_wh_per_km":  round(float(p50), 4),
                "p90_wh_per_km":  round(float(p90), 4),
            })

    df = pd.DataFrame(rows)
    out = output_path or OUTPUT_PATH
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    logger.info("Wrote %d rows to %s", len(df), out)
    return df
