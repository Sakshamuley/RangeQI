"""Synthetic training dataset generation (shared by CLI and startup)."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from app.simulator import EVSpecs, TripConditions, simulate_range_km

logger = logging.getLogger(__name__)

OUTPUT_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "training_data.csv"

EV_FLEET: list[EVSpecs] = [
    EVSpecs(40.5, 1400, 0.32, 2.35, 105, 437),
    EVSpecs(30.2, 1400, 0.32, 2.35, 95, 312),
    EVSpecs(24.0, 1235, 0.33, 2.20, 55, 250),
    EVSpecs(35.0, 1365, 0.33, 2.30, 80, 421),
    EVSpecs(17.3, 1045, 0.35, 2.05, 42, 230),
    EVSpecs(50.3, 1530, 0.33, 2.45, 130, 461),
    EVSpecs(38.0, 1740, 0.34, 2.55, 134, 331),
    EVSpecs(39.2, 1685, 0.29, 2.40, 100, 452),
    EVSpecs(72.6, 1985, 0.29, 2.55, 160, 631),
    EVSpecs(39.4, 1580, 0.34, 2.50, 110, 456),
    EVSpecs(59.0, 2185, 0.33, 2.60, 170, 557),
    EVSpecs(60.48, 1750, 0.29, 2.45, 150, 521),
    EVSpecs(82.56, 2185, 0.23, 2.35, 230, 650),
    EVSpecs(29.2, 1275, 0.34, 2.25, 57, 320),
    EVSpecs(77.4, 2055, 0.28, 2.50, 168, 708),
    EVSpecs(64.7, 2085, 0.26, 2.45, 200, 531),
    EVSpecs(66.5, 2040, 0.28, 2.45, 140, 560),
    EVSpecs(78.0, 2188, 0.32, 2.50, 150, 418),
    EVSpecs(75.0, 2003, 0.23, 2.52, 220, 533),
]

STYLES = ["aggressive", "normal", "calm"]
SEASONS = {"summer": 38, "monsoon": 28, "winter": 18, "spring": 26}


def generate_training_data(n_samples: int = 8000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows: list[dict] = []

    for _ in range(n_samples):
        ev = EV_FLEET[rng.integers(0, len(EV_FLEET))]
        distance = float(rng.uniform(20, 400))
        avg_speed = float(rng.uniform(25, 110))
        elev_gain = float(rng.exponential(150))
        elev_loss = float(rng.exponential(150))
        season = rng.choice(list(SEASONS.keys()))
        temp = float(SEASONS[season] + rng.normal(0, 4))
        style = str(rng.choice(STYLES))
        passengers = int(rng.integers(1, 5))

        trip = TripConditions(
            distance_km=distance,
            avg_speed_kmh=avg_speed,
            elevation_gain_m=elev_gain,
            elevation_loss_m=elev_loss,
            temperature_c=temp,
            driving_style=style,  # type: ignore[arg-type]
            num_passengers=passengers,
        )

        range_km, bd = simulate_range_km(ev, trip, add_noise=True, rng=rng)
        per_km_energy = bd.total_energy_kwh / max(distance, 0.1)

        rows.append(
            {
                "battery_capacity_kwh": ev.battery_capacity_kwh,
                "kerb_weight_kg": ev.kerb_weight_kg,
                "drag_coefficient": ev.drag_coefficient,
                "frontal_area_m2": ev.frontal_area_m2,
                "motor_power_kw": ev.motor_power_kw,
                "claimed_range_km": ev.claimed_range_km,
                "distance_km": distance,
                "avg_speed_kmh": avg_speed,
                "elevation_gain_m": elev_gain,
                "elevation_loss_m": elev_loss,
                "temperature_c": temp,
                "driving_style": style,
                "num_passengers": passengers,
                "season": season,
                "base_consumption_kwh": bd.base_consumption_kwh,
                "aero_drag_kwh": bd.aero_drag_kwh,
                "rolling_resistance_kwh": bd.rolling_resistance_kwh,
                "elevation_gain_kwh": bd.elevation_gain_kwh,
                "regen_recovery_kwh": bd.regen_recovery_kwh,
                "hvac_kwh": bd.hvac_kwh,
                "style_multiplier": bd.style_multiplier,
                "trip_energy_kwh": bd.total_energy_kwh,
                "energy_per_km": per_km_energy,
                "real_world_range_km": range_km,
            }
        )

    return pd.DataFrame(rows)


def save_training_csv(n_samples: int = 8000) -> Path:
    df = generate_training_data(n_samples=n_samples)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)
    logger.info("Saved %d rows to %s", len(df), OUTPUT_PATH)
    return OUTPUT_PATH
