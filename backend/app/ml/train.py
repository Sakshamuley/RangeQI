"""
Train quantile-regression models on the Monte Carlo dataset.

Architecture
------------
Three separate HistGradientBoostingRegressor models with quantile loss:
  - p10 model (alpha=0.10) — conservative estimate
  - p50 model (alpha=0.50) — expected estimate
  - p90 model (alpha=0.90) — optimistic estimate

Cross-validation
----------------
GroupKFold(n_splits=4) grouped by ev_name so that no EV appears in both
train and validation folds.  This tests generalisation to unseen EVs
(the intended production use case).

Features (no claimed_range_km)
-------------------------------
EV physics: battery_capacity_kwh, usable_battery_kwh, kerb_weight_kg,
            drag_coefficient, frontal_area_m2, motor_power_kw
Trip:        distance_km, mean_gradient_mpkm, elevation_gain_per_km,
             elevation_loss_per_km, avg_speed_kmh, highway_fraction,
             temperature_c, num_passengers, ac_on, driving_style_code

Note: the ML layer is a fast emulator of the physics model trained on
simulated data.  It does NOT incorporate road-test data.  Accuracy
against reality is reported in validation_results.json (Phase 4).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold, cross_val_score
from sklearn.metrics import mean_absolute_error

logger = logging.getLogger(__name__)

DATA_PATH   = Path(__file__).resolve().parent.parent.parent / "data" / "training_data.csv"
MODELS_DIR  = Path(__file__).resolve().parent.parent.parent.parent / "models"
RESULTS_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "benchmark_results.json"

FEATURE_COLS = [
    "battery_capacity_kwh", "usable_battery_kwh", "kerb_weight_kg",
    "drag_coefficient", "frontal_area_m2", "motor_power_kw",
    "distance_km", "mean_gradient_mpkm", "elevation_gain_per_km",
    "elevation_loss_per_km", "avg_speed_kmh", "highway_fraction",
    "temperature_c", "num_passengers", "ac_on", "driving_style_code",
]

QUANTILES = {
    "p10": 0.10,
    "p50": 0.50,
    "p90": 0.90,
}

# Shared hyperparameters — fast but sufficient for the emulator task
HGBR_PARAMS = dict(
    loss="quantile",
    max_iter=300,
    learning_rate=0.05,
    max_depth=6,
    min_samples_leaf=20,
    random_state=42,
)


def _make_model(alpha: float) -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(quantile=alpha, **HGBR_PARAMS)


def train(data_path: Path | None = None) -> dict:
    """
    Load training data, train three quantile models, evaluate with
    GroupKFold CV, save models, and write benchmark_results.json.

    Returns a dict with CV metrics.
    """
    csv = data_path or DATA_PATH
    logger.info("Loading training data from %s …", csv)
    df = pd.read_csv(csv)

    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing feature columns: {missing}")

    X = df[FEATURE_COLS].values
    groups = df["ev_name"].values
    cv = GroupKFold(n_splits=4)

    results: dict = {
        "training_rows": len(df),
        "n_evs": int(df["ev_name"].nunique()),
        "ev_names": sorted(df["ev_name"].unique().tolist()),
        "feature_cols": FEATURE_COLS,
        "cv_strategy": "GroupKFold(n_splits=4) by ev_name",
        "note": (
            "The ML layer is a fast emulator of the physics model trained on "
            "simulated data (Monte Carlo over uncertain parameters). "
            "It does NOT incorporate road-test data. "
            "Accuracy against road tests is in validation_results.json."
        ),
        "quantile_models": {},
    }

    models: dict[str, HistGradientBoostingRegressor] = {}

    for name, alpha in QUANTILES.items():
        target_col = f"{name}_wh_per_km"
        y = df[target_col].values
        logger.info("Training %s model (alpha=%.2f) …", name, alpha)

        model = _make_model(alpha)

        # GroupKFold cross-validation MAE
        maes: list[float] = []
        for train_idx, val_idx in cv.split(X, y, groups):
            model.fit(X[train_idx], y[train_idx])
            preds = model.predict(X[val_idx])
            maes.append(float(mean_absolute_error(y[val_idx], preds)))

        cv_mae = float(np.mean(maes))
        logger.info("  %s CV MAE = %.3f Wh/km", name, cv_mae)

        # Final model trained on full dataset
        model.fit(X, y)
        models[name] = model

        results["quantile_models"][name] = {
            "alpha": alpha,
            "target": target_col,
            "cv_mae_wh_per_km": round(cv_mae, 4),
            "cv_mae_folds": [round(m, 4) for m in maes],
        }

    # Save models
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(models, MODELS_DIR / "quantile_models.joblib")
    # Also save the p50 model as best_model.joblib for backward compatibility
    joblib.dump(models["p50"], MODELS_DIR / "best_model.joblib")
    logger.info("Models saved to %s", MODELS_DIR)

    # Write benchmark results
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)
    logger.info("Benchmark results written to %s", RESULTS_PATH)

    return results
