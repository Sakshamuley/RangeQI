"""Inference, SHAP explanations, and quantile predictions."""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
logger = logging.getLogger(__name__)

try:
    import shap

    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False

MODELS_DIR = Path(__file__).resolve().parent.parent.parent / "models"

# Human-readable labels for SHAP feature contributions (km impact)
FEATURE_LABELS = {
    "elevation_gain_m": "Elevation gain",
    "elevation_loss_m": "Regen (descent)",
    "temperature_c": "HVAC / temperature",
    "avg_speed_kmh": "Speed / aero drag",
    "distance_km": "Trip distance",
    "driving_style": "Driving style",
    "num_passengers": "Passenger load",
    "kerb_weight_kg": "Vehicle weight",
    "battery_capacity_kwh": "Battery capacity",
    "claimed_range_km": "Claimed range baseline",
    "drag_coefficient": "Aerodynamics (Cd)",
    "frontal_area_m2": "Frontal area",
    "motor_power_kw": "Motor power",
    "season": "Season",
}


class Predictor:
    """Loads quantile models and provides range predictions with SHAP breakdown."""

    def __init__(self) -> None:
        self.quantile_bundle: dict | None = None
        self.best_bundle: dict | None = None
        self._explainer = None
        self._load_models()

    def _load_models(self) -> None:
        qpath = MODELS_DIR / "quantile_models.joblib"
        bpath = MODELS_DIR / "best_model.joblib"
        if qpath.exists():
            self.quantile_bundle = joblib.load(qpath)
            logger.info("Loaded quantile models from %s", qpath)
        else:
            logger.warning("Quantile models not found — run app.ml.train first")
        if bpath.exists():
            self.best_bundle = joblib.load(bpath)
            logger.info("Loaded best model (%s)", self.best_bundle.get("name"))

    @property
    def is_ready(self) -> bool:
        return self.quantile_bundle is not None

    def _build_frame(self, features: dict[str, Any]) -> pd.DataFrame:
        categorical = self.quantile_bundle["categorical"]
        feature_cols = self.quantile_bundle["feature_cols"]
        row = {c: features.get(c) for c in categorical + feature_cols}
        return pd.DataFrame([row])

    def predict_quantiles(self, features: dict[str, Any]) -> dict[str, float]:
        """Return p10, p50, p90 range predictions in km."""
        if not self.quantile_bundle:
            raise RuntimeError("Models not trained. Run training pipeline first.")

        df = self._build_frame(features)
        preprocessor = self.quantile_bundle["preprocessor"]
        X = preprocessor.transform(df)
        models = self.quantile_bundle["models"]

        t0 = time.perf_counter()
        p10 = float(models["p10"].predict(X)[0])
        p50 = float(models["p50"].predict(X)[0])
        p90 = float(models["p90"].predict(X)[0])
        infer_ms = (time.perf_counter() - t0) * 1000

        # Ensure ordering p10 <= p50 <= p90
        p10, p50, p90 = sorted([p10, p50, p90])

        return {
            "p10": round(p10, 1),
            "p50": round(p50, 1),
            "p90": round(p90, 1),
            "inference_time_ms": round(infer_ms, 3),
        }

    def shap_breakdown(self, features: dict[str, Any], claimed_range: float) -> list[dict]:
        """
        SHAP values for p50 model — converted to km impact on range.
        Positive SHAP = increases predicted range; negative = reduces it.
        """
        if not self.quantile_bundle or not HAS_SHAP:
            return self._physics_fallback_breakdown(features, claimed_range)

        df = self._build_frame(features)
        preprocessor = self.quantile_bundle["preprocessor"]
        X = preprocessor.transform(df)
        model = self.quantile_bundle["models"]["p50"]

        try:
            if self._explainer is None:
                self._explainer = shap.TreeExplainer(model)
            shap_values = self._explainer.shap_values(X)
            if isinstance(shap_values, list):
                shap_values = shap_values[0]

            # Map transformed feature names back
            cat_names = list(
                preprocessor.named_transformers_["cat"].get_feature_names_out(
                    self.quantile_bundle["categorical"]
                )
            )
            all_names = cat_names + self.quantile_bundle["feature_cols"]
            values = shap_values[0]

            # Aggregate one-hot driving_style / season back to single labels
            aggregated: dict[str, float] = {}
            for name, val in zip(all_names, values):
                base = name.split("_")[0] if name.startswith(("driving_style", "season")) else name
                if name.startswith("driving_style"):
                    key = "driving_style"
                elif name.startswith("season"):
                    key = "season"
                else:
                    key = name
                aggregated[key] = aggregated.get(key, 0.0) + float(val)

            # Sort by absolute impact, return top contributors
            breakdown = []
            for key, km_impact in sorted(aggregated.items(), key=lambda x: abs(x[1]), reverse=True):
                label = FEATURE_LABELS.get(key, key)
                breakdown.append(
                    {
                        "feature": key,
                        "label": label,
                        "km_impact": round(km_impact, 1),
                    }
                )
            return breakdown[:8]
        except Exception as exc:
            logger.warning("SHAP computation failed: %s", exc)
            # Physics-based fallback breakdown from feature magnitudes
            return self._physics_fallback_breakdown(features, claimed_range)

    def _physics_fallback_breakdown(
        self, features: dict[str, Any], claimed_range: float
    ) -> list[dict]:
        """Fallback when SHAP unavailable — approximate km impacts."""
        elev = -features.get("elevation_gain_m", 0) * 0.02
        regen = features.get("elevation_loss_m", 0) * 0.012
        temp_delta = abs(features.get("temperature_c", 25) - 22)
        hvac = -temp_delta * 0.8
        speed = features.get("avg_speed_kmh", 50)
        aero = -(speed - 60) * 0.15 if speed > 60 else (60 - speed) * 0.05
        style_map = {"aggressive": -15, "normal": 0, "calm": 8}
        style = style_map.get(features.get("driving_style", "normal"), 0)

        items = [
            ("elevation_gain_m", "Elevation gain", elev),
            ("regen", "Regen (descent)", regen),
            ("temperature_c", "HVAC / temperature", hvac),
            ("avg_speed_kmh", "Speed / aero drag", aero),
            ("driving_style", "Driving style", style),
        ]
        return [
            {"feature": f, "label": l, "km_impact": round(v, 1)}
            for f, l, v in sorted(items, key=lambda x: abs(x[2]), reverse=True)
        ]


# Singleton for API use
predictor = Predictor()
