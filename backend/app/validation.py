"""
Phase 4: Validation and calibration against Autocar India real-world range tests.

Source: https://www.autocarindia.com/auto-features/real-world-ev-range-tested-of-top-ev-cars-in-india-434105
Date: 28 Apr 2026

Approach
--------
Validate in Wh/km, not in range km, because Autocar's range = battery_kwh × efficiency.
Measured Wh/km = 1000 / km_per_kWh (published by Autocar).

Fixed validation scenario (defined in advance, documented here):
  50% city (avg 35 km/h, urban stops) + 50% highway (avg 80 km/h, highway stops)
  Flat road (elevation gain = 0, loss = 0)
  28 °C, two occupants, AC on

Calibration
-----------
Calibrate at most two parameters: P_aux_kw and eta_regen.
Fixed calibration set (5 EVs): Comet, Tiago 24, Nexon EV 45, Curvv EV, Harrier EV AWD.
Test set (11 EVs): all other Autocar cases.
Calibration minimises RMSE on the 5-case set using scipy.optimize.minimize (L-BFGS-B).

Sensitivity runs
----------------
In addition to the 50/50 baseline, report 30/70 and 70/30 city/highway mixes.

Output
------
validation_results.json (fields):
  scenario, calibration_params, calibration_set, test_set,
  calibration_mae, calibration_mape, test_mae, test_mape,
  claimed_baseline_mae, claimed_baseline_mape,
  spearman_rho, spearman_pvalue,
  sensitivity, cross_check_gaadify, results (per EV)
"""

from __future__ import annotations

import csv
import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import scipy.optimize
import scipy.stats

from app.physics_config import PhysicsConfig
from app.simulator import EVSpecs, simulate_energy_simple

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
VALIDATION_CSV = DATA_DIR / "validation_cases.csv"
VALIDATION_JSON = DATA_DIR / "validation_results.json"

# ── EV physics specs (from seed CSV; assumed flags documented in ASSUMPTIONS.md) ─

EV_SPECS: dict[str, dict] = {
    "MG Comet EV": dict(battery_capacity_kwh=17.3, usable_battery_kwh=16.44,
                        kerb_weight_kg=1045, drag_coefficient=0.35,
                        frontal_area_m2=2.05, motor_power_kw=42),
    "Tata Tiago EV 24 kWh": dict(battery_capacity_kwh=24.0, usable_battery_kwh=22.80,
                                  kerb_weight_kg=1235, drag_coefficient=0.33,
                                  frontal_area_m2=2.20, motor_power_kw=55),
    "Tata Tigor EV": dict(battery_capacity_kwh=26.0, usable_battery_kwh=24.70,
                           kerb_weight_kg=1230, drag_coefficient=0.34,
                           frontal_area_m2=2.20, motor_power_kw=55),
    "Citroen eC3": dict(battery_capacity_kwh=29.2, usable_battery_kwh=27.74,
                        kerb_weight_kg=1275, drag_coefficient=0.34,
                        frontal_area_m2=2.25, motor_power_kw=57),
    "MG Windsor EV": dict(battery_capacity_kwh=38.0, usable_battery_kwh=36.10,
                          kerb_weight_kg=1740, drag_coefficient=0.34,
                          frontal_area_m2=2.55, motor_power_kw=134),
    "Tata Nexon EV 40.5 kWh": dict(battery_capacity_kwh=40.5, usable_battery_kwh=38.48,
                                    kerb_weight_kg=1400, drag_coefficient=0.32,
                                    frontal_area_m2=2.35, motor_power_kw=105),
    "Tata Nexon EV 45 kWh": dict(battery_capacity_kwh=45.0, usable_battery_kwh=42.75,
                                  kerb_weight_kg=1410, drag_coefficient=0.32,
                                  frontal_area_m2=2.35, motor_power_kw=130),
    "MG ZS EV": dict(battery_capacity_kwh=50.3, usable_battery_kwh=47.79,
                     kerb_weight_kg=1530, drag_coefficient=0.33,
                     frontal_area_m2=2.45, motor_power_kw=130),
    "Hyundai Creta Electric LR": dict(battery_capacity_kwh=51.4, usable_battery_kwh=48.83,
                                       kerb_weight_kg=1595, drag_coefficient=0.28,
                                       frontal_area_m2=2.40, motor_power_kw=171),
    "Kia Carens Clavis EV": dict(battery_capacity_kwh=51.4, usable_battery_kwh=48.83,
                                  kerb_weight_kg=1730, drag_coefficient=0.32,
                                  frontal_area_m2=2.55, motor_power_kw=171),
    "Tata Curvv EV": dict(battery_capacity_kwh=55.0, usable_battery_kwh=52.25,
                          kerb_weight_kg=1680, drag_coefficient=0.31,
                          frontal_area_m2=2.45, motor_power_kw=167),
    "VinFast VF7 AWD": dict(battery_capacity_kwh=70.8, usable_battery_kwh=67.26,
                             kerb_weight_kg=1930, drag_coefficient=0.33,
                             frontal_area_m2=2.55, motor_power_kw=260),
    "Tata Harrier EV AWD": dict(battery_capacity_kwh=75.0, usable_battery_kwh=71.25,
                                 kerb_weight_kg=2150, drag_coefficient=0.33,
                                 frontal_area_m2=2.65, motor_power_kw=170),
    "Mahindra BE 6": dict(battery_capacity_kwh=79.0, usable_battery_kwh=75.05,
                          kerb_weight_kg=1900, drag_coefficient=0.32,
                          frontal_area_m2=2.55, motor_power_kw=282),
    "Mahindra XEV 9e": dict(battery_capacity_kwh=79.0, usable_battery_kwh=75.05,
                             kerb_weight_kg=2100, drag_coefficient=0.31,
                             frontal_area_m2=2.70, motor_power_kw=282),
    "Mahindra XEV 9S": dict(battery_capacity_kwh=79.0, usable_battery_kwh=75.05,
                             kerb_weight_kg=2200, drag_coefficient=0.32,
                             frontal_area_m2=2.75, motor_power_kw=170),
}

# ── Fixed validation scenario ─────────────────────────────────────────────────

SCENARIO = {
    "description": (
        "50% city (avg 35 km/h, urban stops) + 50% highway (avg 80 km/h); "
        "flat road; 28 °C; 2 occupants; AC on"
    ),
    "city_speed_kmh": 35.0,
    "highway_speed_kmh": 80.0,
    "city_fraction": 0.50,
    "temperature_c": 28.0,
    "num_passengers": 2,
    "ac_on": True,
    "elevation_gain_m": 0.0,
    "elevation_loss_m": 0.0,
    "distance_per_leg_km": 50.0,   # each half simulated over 50 km
}

CALIBRATION_CASES = {
    "MG Comet EV",
    "Tata Tiago EV 24 kWh",
    "Tata Nexon EV 45 kWh",
    "Tata Curvv EV",
    "Tata Harrier EV AWD",
}


def _predict_wh_per_km(
    ev_name: str,
    config: PhysicsConfig,
    city_fraction: float = SCENARIO["city_fraction"],
) -> float:
    """
    Predict Wh/km for a single EV using the fixed scenario.

    Runs two simulate_energy_simple calls (city leg + highway leg)
    weighted by city_fraction and (1-city_fraction).
    """
    specs = EV_SPECS[ev_name]
    ev = EVSpecs(**specs)
    d = SCENARIO["distance_per_leg_km"]
    t = SCENARIO["temperature_c"]
    n = SCENARIO["num_passengers"]
    ac = SCENARIO["ac_on"]
    gain = SCENARIO["elevation_gain_m"]
    loss = SCENARIO["elevation_loss_m"]

    city = simulate_energy_simple(
        ev, d, gain, loss,
        avg_speed_kmh=SCENARIO["city_speed_kmh"],
        temperature_c=t, num_passengers=n, ac_on=ac,
        config=config,
    )
    hw = simulate_energy_simple(
        ev, d, gain, loss,
        avg_speed_kmh=SCENARIO["highway_speed_kmh"],
        temperature_c=t, num_passengers=n, ac_on=ac,
        config=config,
    )
    hw_fraction = 1.0 - city_fraction
    return city_fraction * city.total_wh_per_km + hw_fraction * hw.total_wh_per_km


def _load_validation_cases() -> list[dict]:
    """Read validation_cases.csv, skipping rows with missing measured Wh/km."""
    cases = []
    with open(VALIDATION_CSV, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            # Skip Gaadify cross-check row (handled separately)
            if "Gaadify" in row["ev_name"]:
                continue
            try:
                wh_per_km = float(row["measured_wh_per_km"])
                claimed_km = float(row["midc_claimed_km"])
                battery_kwh = float(row["battery_kwh"])
            except (ValueError, KeyError):
                continue
            cases.append({
                "ev_name": row["ev_name"],
                "battery_kwh": battery_kwh,
                "measured_wh_per_km": wh_per_km,
                "claimed_km": claimed_km,
                "tested_km": float(row["tested_km"]) if row.get("tested_km") else None,
                "km_per_kwh": float(row["km_per_kwh"]) if row.get("km_per_kwh") else None,
                "set": row["set"],
                "source_url": row["source_url"],
                "data_verified": row.get("data_verified", "false").strip().lower() == "true",
            })
    return cases


def _objective(params: list[float], calib_cases: list[dict]) -> float:
    """RMSE objective for scipy.optimize over calibration set."""
    p_aux_kw, eta_regen = params
    # Clamp to physically reasonable bounds (avoid scipy going wild)
    p_aux_kw = max(0.0, min(2.0, p_aux_kw))
    eta_regen = max(0.30, min(0.90, eta_regen))
    cfg = PhysicsConfig(p_aux_kw=p_aux_kw, eta_regen=eta_regen)
    sse = 0.0
    for c in calib_cases:
        pred = _predict_wh_per_km(c["ev_name"], cfg)
        sse += (pred - c["measured_wh_per_km"]) ** 2
    return math.sqrt(sse / len(calib_cases))


def _calibrate(calib_cases: list[dict]) -> tuple[float, float]:
    """
    Calibrate P_aux_kw and eta_regen on the fixed calibration set.
    Uses L-BFGS-B with multiple starting points to avoid local minima.
    Returns (p_aux_kw, eta_regen).
    """
    bounds = [(0.0, 2.0), (0.30, 0.90)]
    starts = [
        [0.25, 0.65],   # prior nominal
        [0.50, 0.50],
        [1.00, 0.30],
        [0.80, 0.40],
        [1.20, 0.35],
    ]
    best_result = None
    best_val = float("inf")
    for x0 in starts:
        result = scipy.optimize.minimize(
            _objective, x0, args=(calib_cases,),
            method="L-BFGS-B", bounds=bounds,
            options={"ftol": 1e-10, "gtol": 1e-8, "maxiter": 1000},
        )
        if result.fun < best_val:
            best_val = result.fun
            best_result = result
    return float(best_result.x[0]), float(best_result.x[1])


def _per_case_result(
    case: dict,
    config: PhysicsConfig,
) -> dict:
    """Compute predicted vs measured for one case."""
    pred = _predict_wh_per_km(case["ev_name"], config)
    meas = case["measured_wh_per_km"]
    pct_err = (pred - meas) / meas * 100.0
    # Claimed baseline: convert MIDC claimed range to Wh/km
    claimed_wh_per_km = case["battery_kwh"] * 1000.0 / case["claimed_km"]
    return {
        "ev_name": case["ev_name"],
        "set": case["set"],
        "source_url": case["source_url"],
        "data_verified": case["data_verified"],
        "measured_wh_per_km": round(meas, 2),
        "predicted_wh_per_km": round(pred, 2),
        "pct_error": round(pct_err, 2),
        "claimed_wh_per_km": round(claimed_wh_per_km, 2),
        "claimed_pct_error": round((claimed_wh_per_km - meas) / meas * 100.0, 2),
    }


def _metrics(results: list[dict]) -> dict:
    errors = [abs(r["pct_error"]) for r in results]
    abs_errors = [abs(r["predicted_wh_per_km"] - r["measured_wh_per_km"]) for r in results]
    return {
        "mae_wh_per_km": round(sum(abs_errors) / len(abs_errors), 3),
        "mape_pct": round(sum(errors) / len(errors), 3),
    }


def _spearman(results: list[dict]) -> dict:
    measured = [r["measured_wh_per_km"] for r in results]
    predicted = [r["predicted_wh_per_km"] for r in results]
    rho, pval = scipy.stats.spearmanr(predicted, measured)
    return {"rho": round(float(rho), 4), "p_value": round(float(pval), 6)}


def _sensitivity(config: PhysicsConfig) -> list[dict]:
    """
    Run 30/70 and 70/30 city/highway splits in addition to 50/50 baseline.
    Returns list of scenario results.
    """
    all_cases = _load_validation_cases()
    out = []
    for label, frac in [("30% city / 70% highway", 0.30),
                         ("50% city / 50% highway (baseline)", 0.50),
                         ("70% city / 30% highway", 0.70)]:
        preds = []
        for c in all_cases:
            pred = _predict_wh_per_km(c["ev_name"], config, city_fraction=frac)
            meas = c["measured_wh_per_km"]
            preds.append({"ev": c["ev_name"], "pred": pred, "meas": meas})
        errors = [abs(p["pred"] - p["meas"]) / p["meas"] * 100.0 for p in preds]
        out.append({
            "label": label,
            "city_fraction": frac,
            "mape_pct": round(sum(errors) / len(errors), 3),
            "mae_wh_per_km": round(
                sum(abs(p["pred"] - p["meas"]) for p in preds) / len(preds), 3
            ),
        })
    return out


def _gaadify_cross_check(config: PhysicsConfig) -> dict:
    """
    Second independent cross-check: Gaadify Tiago EV 24 kWh Bengaluru test.
    Source: https://gaadify.com/tata-tiago-ev-24-kwh-real-world-range-test/
    Conditions: ~185 km, 40% city + 60% highway, 2 occupants + 20 kg luggage.
    Speed and temperature not published → marked estimated.
    """
    ev_name = "Tata Tiago EV 24 kWh"
    specs = EV_SPECS[ev_name]
    ev = EVSpecs(**specs)
    d = 50.0
    # 40% city, 60% highway; 20 kg luggage added as direct mass
    city = simulate_energy_simple(
        ev, d, 0.0, 0.0,
        avg_speed_kmh=35.0,
        temperature_c=28.0,  # estimated (Bengaluru; no season published)
        num_passengers=2,
        ac_on=True,
        luggage_kg=20.0,
        config=config,
    )
    hw = simulate_energy_simple(
        ev, d, 0.0, 0.0,
        avg_speed_kmh=80.0,
        temperature_c=28.0,
        num_passengers=2,
        ac_on=True,
        luggage_kg=20.0,
        config=config,
    )
    pred_wh_per_km = 0.40 * city.total_wh_per_km + 0.60 * hw.total_wh_per_km
    # Measured: 185 km on 24 kWh battery → ~129.7 Wh/km
    measured_wh_per_km = 24_000.0 / 185.0
    pct_err = (pred_wh_per_km - measured_wh_per_km) / measured_wh_per_km * 100.0
    return {
        "source": "Gaadify Bengaluru test",
        "source_url": "https://gaadify.com/tata-tiago-ev-24-kwh-real-world-range-test/",
        "ev_name": ev_name,
        "tested_km_approx": 185,
        "measured_wh_per_km": round(measured_wh_per_km, 2),
        "predicted_wh_per_km": round(pred_wh_per_km, 2),
        "pct_error": round(pct_err, 2),
        "scenario": "40% city 35 km/h + 60% highway 80 km/h; 28°C estimated; 2 occupants + 20 kg luggage (luggage modelled as extra passenger weight ≈ 20 kg)",
        "note": "Temperature and average speeds are not published; marked estimated. 20 kg luggage added to passenger mass (2×75 + 20 = 170 kg total occupant load).",
    }


def run_validation() -> dict:
    """
    Full Phase 4 pipeline:
    1. Load validation_cases.csv
    2. Calibrate P_aux_kw and eta_regen on 5 calibration cases
    3. Evaluate all 16 cases with calibrated config
    4. Compute metrics (MAE, MAPE) for calibration set and test set separately
    5. Compute claimed-baseline metrics
    6. Compute Spearman rank correlation
    7. Run sensitivity (30/70, 50/50, 70/30 city/highway)
    8. Run Gaadify cross-check
    9. Write validation_results.json
    """
    cases = _load_validation_cases()
    calib_cases = [c for c in cases if c["ev_name"] in CALIBRATION_CASES]
    test_cases  = [c for c in cases if c["ev_name"] not in CALIBRATION_CASES]

    logger.info("Calibrating on %d cases …", len(calib_cases))
    p_aux_kw, eta_regen = _calibrate(calib_cases)
    logger.info("Calibrated: p_aux_kw=%.4f kW, eta_regen=%.4f", p_aux_kw, eta_regen)

    config = PhysicsConfig(p_aux_kw=p_aux_kw, eta_regen=eta_regen)

    calib_results = [_per_case_result(c, config) for c in calib_cases]
    test_results  = [_per_case_result(c, config) for c in test_cases]
    all_results   = calib_results + test_results

    calib_metrics = _metrics(calib_results)
    test_metrics  = _metrics(test_results)

    # Claimed-range baseline (claimed MIDC Wh/km vs measured)
    claimed_errors_abs  = [abs(r["claimed_wh_per_km"] - r["measured_wh_per_km"]) for r in all_results]
    claimed_errors_pct  = [abs(r["claimed_pct_error"]) for r in all_results]
    claimed_baseline = {
        "mae_wh_per_km": round(sum(claimed_errors_abs) / len(claimed_errors_abs), 3),
        "mape_pct": round(sum(claimed_errors_pct) / len(claimed_errors_pct), 3),
    }

    spearman = _spearman(all_results)
    sensitivity = _sensitivity(config)
    cross_check = _gaadify_cross_check(config)

    output = {
        "scenario": SCENARIO,
        "calibration_params": {
            "p_aux_kw": round(p_aux_kw, 4),
            "eta_regen": round(eta_regen, 4),
            "note": (
                "Calibrated on 5 cases to minimise RMSE; "
                "Crr, eta_drive, HVAC parameters held fixed at prior nominals."
            ),
        },
        "calibration_set": {
            "n": len(calib_results),
            "cases": sorted(CALIBRATION_CASES),
            **calib_metrics,
        },
        "test_set": {
            "n": len(test_results),
            **test_metrics,
        },
        "all_cases_combined": {
            "n": len(all_results),
            **_metrics(all_results),
        },
        "claimed_baseline": claimed_baseline,
        "spearman_rank_correlation": spearman,
        "sensitivity": sensitivity,
        "cross_check_gaadify": cross_check,
        "results": all_results,
        "notes": [
            "Validation target is Wh/km (= 1000 / km_per_kWh published by Autocar India).",
            "Test conditions for Autocar tests: Mumbai city+highway weekday loop, "
            "2 occupants, AC auto 22°C, regen set per road type, fully charged. "
            "Average speed, ambient temperature and elevation are NOT published and "
            "are marked 'estimated' in validation_cases.csv.",
            "The physics model is trained on simulated data only (Phase 5). "
            "Errors against road tests reflect assumed parameter uncertainty.",
            "Luggage in the Gaadify cross-check (20 kg) is added to occupant mass "
            "as a direct mass increase (conservative: no change in Cd or CdA).",
        ],
    }

    VALIDATION_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(VALIDATION_JSON, "w", encoding="utf-8") as fh:
        json.dump(output, fh, indent=2)

    logger.info(
        "Validation complete — calib MAPE=%.1f%%, test MAPE=%.1f%%, "
        "claimed baseline MAPE=%.1f%%, Spearman ρ=%.3f",
        calib_metrics["mape_pct"], test_metrics["mape_pct"],
        claimed_baseline["mape_pct"], spearman["rho"],
    )
    return output
