"""Train regression models and quantile models for RangeIQ."""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
logger = logging.getLogger(__name__)

try:
    import xgboost as xgb

    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    from sklearn.ensemble import HistGradientBoostingRegressor

BACKEND_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BACKEND_ROOT / "data"
MODELS_DIR = BACKEND_ROOT / "models"
TRAINING_CSV = DATA_DIR / "training_data.csv"
BENCHMARK_JSON = DATA_DIR / "benchmark_results.json"

FEATURE_COLS = [
    "battery_capacity_kwh",
    "kerb_weight_kg",
    "drag_coefficient",
    "frontal_area_m2",
    "motor_power_kw",
    "claimed_range_km",
    "distance_km",
    "avg_speed_kmh",
    "elevation_gain_m",
    "elevation_loss_m",
    "temperature_c",
    "num_passengers",
]
CATEGORICAL = ["driving_style", "season"]
TARGET = "real_world_range_km"


def _build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
        ],
        remainder="passthrough",
    )


def _get_feature_names(preprocessor: ColumnTransformer) -> list[str]:
    cat_names = list(preprocessor.named_transformers_["cat"].get_feature_names_out(CATEGORICAL))
    return cat_names + FEATURE_COLS


def load_training_data() -> pd.DataFrame:
    if not TRAINING_CSV.exists():
        raise FileNotFoundError(
            f"{TRAINING_CSV} not found. Run scripts/generate_training_data.py first."
        )
    return pd.read_csv(TRAINING_CSV)


def train_benchmark_models(X_train, X_test, y_train, y_test) -> dict:
    """Train Linear, RF, XGBoost; return metrics and best model name."""
    preprocessor = _build_preprocessor()
    col_order = CATEGORICAL + FEATURE_COLS
    X_tr = preprocessor.fit_transform(X_train[col_order])
    X_te = preprocessor.transform(X_test[col_order])

    models = {
        "linear_regression": LinearRegression(),
        "random_forest": RandomForestRegressor(
            n_estimators=100, max_depth=12, random_state=42, n_jobs=-1
        ),
        ("xgboost" if HAS_XGBOOST else "hist_gradient_boosting"): (
            xgb.XGBRegressor(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.08,
                random_state=42,
                n_jobs=-1,
            )
            if HAS_XGBOOST
            else HistGradientBoostingRegressor(
                max_iter=200,
                max_depth=6,
                learning_rate=0.08,
                random_state=42,
            )
        ),
    }

    results: dict = {}
    best_name = ""
    best_mae = float("inf")
    best_model = None

    for name, model in models.items():
        t0 = time.perf_counter()
        model.fit(X_tr, y_train)
        train_time = time.perf_counter() - t0

        t1 = time.perf_counter()
        preds = model.predict(X_te)
        infer_time = (time.perf_counter() - t1) / len(X_te)

        mae = float(mean_absolute_error(y_test, preds))
        rmse = float(np.sqrt(mean_squared_error(y_test, preds)))

        results[name] = {
            "mae": round(mae, 3),
            "rmse": round(rmse, 3),
            "train_time_sec": round(train_time, 3),
            "inference_time_ms": round(infer_time * 1000, 4),
        }
        logger.info("%s — MAE=%.2f RMSE=%.2f", name, mae, rmse)

        if mae < best_mae:
            best_mae = mae
            best_name = name
            best_model = model

    results["best_model"] = best_name
    return results, best_name, best_model, preprocessor


def train_quantile_models(X_train, X_test, y_train, y_test, preprocessor) -> dict:
    """
    Train separate GradientBoostingRegressor models for p10, p50, p90.
    Uses scikit-learn's quantile loss — key for uncertainty bands.
    """
    col_order = CATEGORICAL + FEATURE_COLS
    X_tr = preprocessor.transform(X_train[col_order])
    X_te = preprocessor.transform(X_test[col_order])

    quantiles = {0.10: "p10", 0.50: "p50", 0.90: "p90"}
    quantile_models: dict = {}
    metrics: dict = {}

    for q, label in quantiles.items():
        t0 = time.perf_counter()
        model = GradientBoostingRegressor(
            loss="quantile",
            alpha=q,
            n_estimators=150,
            max_depth=5,
            learning_rate=0.08,
            random_state=42,
        )
        model.fit(X_tr, y_train)
        train_time = time.perf_counter() - t0
        preds = model.predict(X_te)
        mae = float(mean_absolute_error(y_test, preds))
        quantile_models[label] = model
        metrics[label] = {"mae": round(mae, 3), "train_time_sec": round(train_time, 3)}
        logger.info("Quantile %s — MAE=%.2f", label, mae)

    return quantile_models, metrics


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    df = load_training_data()
    X = df[CATEGORICAL + FEATURE_COLS]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    benchmark, best_name, best_model, preprocessor = train_benchmark_models(
        X_train, X_test, y_train, y_test
    )

    quantile_models, quantile_metrics = train_quantile_models(
        X_train, X_test, y_train, y_test, preprocessor
    )

    benchmark["quantile_models"] = quantile_metrics
    benchmark["training_samples"] = len(df)
    benchmark["feature_columns"] = FEATURE_COLS + CATEGORICAL

    BENCHMARK_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(BENCHMARK_JSON, "w", encoding="utf-8") as f:
        json.dump(benchmark, f, indent=2)

    # Save point-estimate model (best benchmark)
    joblib.dump(
        {"model": best_model, "preprocessor": preprocessor, "name": best_name},
        MODELS_DIR / "best_model.joblib",
    )

    # Save quantile bundle for API uncertainty bands
    joblib.dump(
        {
            "models": quantile_models,
            "preprocessor": preprocessor,
            "feature_cols": FEATURE_COLS,
            "categorical": CATEGORICAL,
        },
        MODELS_DIR / "quantile_models.joblib",
    )

    print(f"Best model: {best_name}")
    print(f"Benchmark saved to {BENCHMARK_JSON}")
    print(f"Models saved to {MODELS_DIR}")


if __name__ == "__main__":
    main()
