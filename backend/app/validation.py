"""Real-world validation cases from published Indian EV test drives."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd

from app.simulator import EVSpecs, TripConditions, simulate_range_km

logger = logging.getLogger(__name__)

VALIDATION_JSON = Path(__file__).resolve().parent.parent / "data" / "validation_results.json"

# Published test-drive data (journalist-reported actual vs claimed under known conditions)
# Sources: Autocar India, CarWale, Team-BHP long-term reports (approximate conditions)
VALIDATION_CASES = [
    {
        "source": "Autocar India long-term",
        "model": "Tata Nexon EV Max",
        "battery_kwh": 40.5,
        "weight": 1400,
        "cd": 0.32,
        "area": 2.35,
        "motor_kw": 105,
        "claimed_km": 437,
        "reported_km": 280,
        "conditions": {"distance": 200, "speed": 60, "elev_gain": 100, "elev_loss": 80, "temp": 32, "style": "normal"},
    },
    {
        "source": "CarWale highway test",
        "model": "MG ZS EV",
        "battery_kwh": 50.3,
        "weight": 1530,
        "cd": 0.33,
        "area": 2.45,
        "motor_kw": 130,
        "claimed_km": 461,
        "reported_km": 340,
        "conditions": {"distance": 250, "speed": 90, "elev_gain": 50, "elev_loss": 50, "temp": 35, "style": "aggressive"},
    },
    {
        "source": "Team-BHP city commute",
        "model": "Tata Tiago EV",
        "battery_kwh": 24.0,
        "weight": 1235,
        "cd": 0.33,
        "area": 2.20,
        "motor_kw": 55,
        "claimed_km": 250,
        "reported_km": 195,
        "conditions": {"distance": 80, "speed": 35, "elev_gain": 20, "elev_loss": 20, "temp": 30, "style": "calm"},
    },
    {
        "source": "Autocar India",
        "model": "Hyundai Kona Electric",
        "battery_kwh": 39.2,
        "weight": 1685,
        "cd": 0.29,
        "area": 2.40,
        "motor_kw": 100,
        "claimed_km": 452,
        "reported_km": 305,
        "conditions": {"distance": 180, "speed": 70, "elev_gain": 200, "elev_loss": 180, "temp": 28, "style": "normal"},
    },
    {
        "source": "CarWale",
        "model": "Mahindra XUV400",
        "battery_kwh": 39.4,
        "weight": 1580,
        "cd": 0.34,
        "area": 2.50,
        "motor_kw": 110,
        "claimed_km": 456,
        "reported_km": 310,
        "conditions": {"distance": 220, "speed": 75, "elev_gain": 80, "elev_loss": 70, "temp": 33, "style": "normal"},
    },
    {
        "source": "Overdrive",
        "model": "BYD Atto 3",
        "battery_kwh": 60.48,
        "weight": 1750,
        "cd": 0.29,
        "area": 2.45,
        "motor_kw": 150,
        "claimed_km": 521,
        "reported_km": 380,
        "conditions": {"distance": 300, "speed": 85, "elev_gain": 120, "elev_loss": 100, "temp": 34, "style": "normal"},
    },
    {
        "source": "Autocar India",
        "model": "Citroen eC3",
        "battery_kwh": 29.2,
        "weight": 1275,
        "cd": 0.34,
        "area": 2.25,
        "motor_kw": 57,
        "claimed_km": 320,
        "reported_km": 220,
        "conditions": {"distance": 150, "speed": 55, "elev_gain": 60, "elev_loss": 55, "temp": 31, "style": "normal"},
    },
    {
        "source": "Team-BHP",
        "model": "Tata Punch EV",
        "battery_kwh": 35.0,
        "weight": 1365,
        "cd": 0.33,
        "area": 2.30,
        "motor_kw": 80,
        "claimed_km": 421,
        "reported_km": 290,
        "conditions": {"distance": 200, "speed": 65, "elev_gain": 150, "elev_loss": 140, "temp": 29, "style": "normal"},
    },
    {
        "source": "CarWale highway",
        "model": "Kia EV6",
        "battery_kwh": 77.4,
        "weight": 2055,
        "cd": 0.28,
        "area": 2.50,
        "motor_kw": 168,
        "claimed_km": 708,
        "reported_km": 480,
        "conditions": {"distance": 350, "speed": 100, "elev_gain": 40, "elev_loss": 35, "temp": 36, "style": "aggressive"},
    },
    {
        "source": "Autocar India",
        "model": "MG Comet EV",
        "battery_kwh": 17.3,
        "weight": 1045,
        "cd": 0.35,
        "area": 2.05,
        "motor_kw": 42,
        "claimed_km": 230,
        "reported_km": 165,
        "conditions": {"distance": 60, "speed": 40, "elev_gain": 10, "elev_loss": 10, "temp": 32, "style": "calm"},
    },
    {
        "source": "Overdrive",
        "model": "Hyundai Ioniq 5",
        "battery_kwh": 72.6,
        "weight": 1985,
        "cd": 0.29,
        "area": 2.55,
        "motor_kw": 160,
        "claimed_km": 631,
        "reported_km": 420,
        "conditions": {"distance": 280, "speed": 95, "elev_gain": 70, "elev_loss": 65, "temp": 33, "style": "normal"},
    },
    {
        "source": "CarWale",
        "model": "Mahindra BE 6",
        "battery_kwh": 59.0,
        "weight": 2185,
        "cd": 0.33,
        "area": 2.60,
        "motor_kw": 170,
        "claimed_km": 557,
        "reported_km": 360,
        "conditions": {"distance": 250, "speed": 80, "elev_gain": 200, "elev_loss": 190, "temp": 30, "style": "normal"},
    },
]


def run_validation(predictor=None) -> dict:
    """
    Run validation cases through physics simulator and ML model (if available).
    Logs MAE vs journalist-reported range.
    """
    results = []
    sim_errors = []
    ml_errors = []

    for case in VALIDATION_CASES:
        c = case["conditions"]
        ev = EVSpecs(
            case["battery_kwh"], case["weight"], case["cd"],
            case["area"], case["motor_kw"], case["claimed_km"],
        )
        trip = TripConditions(
            distance_km=c["distance"],
            avg_speed_kmh=c["speed"],
            elevation_gain_m=c["elev_gain"],
            elevation_loss_m=c["elev_loss"],
            temperature_c=c["temp"],
            driving_style=c["style"],  # type: ignore
        )

        sim_range, _ = simulate_range_km(ev, trip, add_noise=False)
        sim_error = abs(sim_range - case["reported_km"])
        sim_errors.append(sim_error)

        entry = {
            "model": case["model"],
            "source": case["source"],
            "claimed_km": case["claimed_km"],
            "reported_km": case["reported_km"],
            "simulator_predicted_km": round(sim_range, 1),
            "simulator_error_km": round(sim_error, 1),
        }

        if predictor and predictor.is_ready:
            features = {
                "battery_capacity_kwh": case["battery_kwh"],
                "kerb_weight_kg": case["weight"],
                "drag_coefficient": case["cd"],
                "frontal_area_m2": case["area"],
                "motor_power_kw": case["motor_kw"],
                "claimed_range_km": case["claimed_km"],
                "distance_km": c["distance"],
                "avg_speed_kmh": c["speed"],
                "elevation_gain_m": c["elev_gain"],
                "elevation_loss_m": c["elev_loss"],
                "temperature_c": c["temp"],
                "driving_style": c["style"],
                "num_passengers": 1,
                "season": "summer",
            }
            ml_pred = predictor.predict_quantiles(features)
            ml_error = abs(ml_pred["p50"] - case["reported_km"])
            ml_errors.append(ml_error)
            entry["ml_predicted_km"] = ml_pred["p50"]
            entry["ml_error_km"] = round(ml_error, 1)

        results.append(entry)

    summary = {
        "cases": len(results),
        "simulator_mae_km": round(sum(sim_errors) / len(sim_errors), 2),
        "results": results,
    }
    if ml_errors:
        summary["ml_mae_km"] = round(sum(ml_errors) / len(ml_errors), 2)

    VALIDATION_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(VALIDATION_JSON, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    logger.info("Validation MAE — simulator: %.1f km", summary["simulator_mae_km"])
    return summary
