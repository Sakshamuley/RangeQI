"""
ML inference: predict p10/p50/p90 Wh/km for a given (EV, trip) scenario.

The predictor loads the three quantile models trained in train.py.
If the model files are absent, it falls back to the physics model
(simulate_energy_simple at the calibrated config) and sets a warning flag.

Note: the ML layer is a fast emulator of the physics model trained on
simulated data.  The p10–p90 band reflects assumed parameter uncertainty only
(see docs/ASSUMPTIONS.md).  Accuracy against road tests is in
validation_results.json (Phase 4).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from app.physics_config import PhysicsConfig
from app.simulator import EVSpecs, simulate_energy_simple
from app.ml.train import FEATURE_COLS, MODELS_DIR

logger = logging.getLogger(__name__)

# Calibrated config for physics fallback (Phase 4 values)
_CALIBRATED_CONFIG = PhysicsConfig(p_aux_kw=1.0, eta_regen=0.30)


@dataclass
class QuantilePrediction:
    """p10/p50/p90 Wh/km prediction for a single (EV, trip) scenario."""
    p10_wh_per_km: float
    p50_wh_per_km: float
    p90_wh_per_km: float
    source: str = "ml"                  # "ml" | "physics_fallback"
    warnings: list[str] = field(default_factory=list)


class QuantilePredictor:
    """
    Loads and wraps the three quantile models from MODELS_DIR.
    Thread-safe after __init__ (models are read-only at inference).
    """

    def __init__(self) -> None:
        self._models: Optional[dict] = None
        self._load_error: Optional[str] = None
        self._try_load()

    def _try_load(self) -> None:
        path = MODELS_DIR / "quantile_models.joblib"
        if not path.exists():
            self._load_error = f"Model file not found: {path}"
            logger.warning(self._load_error)
            return
        try:
            import joblib
            self._models = joblib.load(path)
            logger.info("Quantile models loaded from %s", path)
        except Exception as exc:
            self._load_error = f"Failed to load models: {exc}"
            logger.warning(self._load_error)

    @property
    def is_ready(self) -> bool:
        return self._models is not None

    def predict(
        self,
        ev: EVSpecs,
        distance_km: float,
        mean_gradient_mpkm: float,
        elevation_gain_per_km: float,
        elevation_loss_per_km: float,
        avg_speed_kmh: float,
        highway_fraction: float,
        temperature_c: float,
        num_passengers: int,
        ac_on: bool,
        driving_style: str = "normal",
        elevation_gain_m: float = 0.0,
        elevation_loss_m: float = 0.0,
    ) -> QuantilePrediction:
        """
        Predict p10/p50/p90 Wh/km.  Falls back to physics model if ML is unavailable.
        """
        style_code = {"calm": 0, "normal": 1, "aggressive": 2}.get(driving_style, 1)

        features = np.array([[
            ev.battery_capacity_kwh,
            ev.usable_battery_kwh,
            ev.kerb_weight_kg,
            ev.drag_coefficient,
            ev.frontal_area_m2,
            ev.motor_power_kw,
            distance_km,
            mean_gradient_mpkm,
            elevation_gain_per_km,
            elevation_loss_per_km,
            avg_speed_kmh,
            highway_fraction,
            temperature_c,
            num_passengers,
            int(ac_on),
            style_code,
        ]])

        if self.is_ready:
            p10 = float(self._models["p10"].predict(features)[0])
            p50 = float(self._models["p50"].predict(features)[0])
            p90 = float(self._models["p90"].predict(features)[0])
            # Enforce ordering (quantile crossing can occur at extremes)
            p10 = min(p10, p50)
            p90 = max(p90, p50)
            return QuantilePrediction(
                p10_wh_per_km=round(p10, 2),
                p50_wh_per_km=round(p50, 2),
                p90_wh_per_km=round(p90, 2),
                source="ml",
            )

        # Physics fallback
        warnings = [self._load_error or "ML model unavailable; using physics model fallback."]
        res = simulate_energy_simple(
            ev=ev,
            distance_km=distance_km,
            elevation_gain_m=elevation_gain_m or elevation_gain_per_km * distance_km,
            elevation_loss_m=elevation_loss_m or elevation_loss_per_km * distance_km,
            avg_speed_kmh=avg_speed_kmh,
            temperature_c=temperature_c,
            driving_style=driving_style,        # type: ignore[arg-type]
            num_passengers=num_passengers,
            ac_on=ac_on,
            config=_CALIBRATED_CONFIG,
        )
        p50 = res.total_wh_per_km
        # Approximate uncertainty band: ±15% around physics estimate
        p10 = p50 * 0.85
        p90 = p50 * 1.15
        return QuantilePrediction(
            p10_wh_per_km=round(p10, 2),
            p50_wh_per_km=round(p50, 2),
            p90_wh_per_km=round(p90, 2),
            source="physics_fallback",
            warnings=warnings,
        )
