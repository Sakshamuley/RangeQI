"""
Phase 5 unit tests for the ML data pipeline.

Strategy:
  - data_gen: test with n_trips=5, n_mc=4 (fast)
  - train: test with a tiny dataframe
  - predict: test the predictor with both ML-ready and fallback modes
  - No actual ML training in CI (mocked where it takes > 2 s)
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from app.ml.data_gen import (
    N_MC_SAMPLES,
    N_TRIPS_PER_EV,
    TERRAIN_CLASSES,
    TERRAIN_PROBS,
    ROAD_CLASS_SPEED,
    ROAD_CLASS_PROBS,
    SEASON_TEMP_RANGE,
    SEASON_PROBS,
    generate,
    _sample_config,
    _sample_trip,
)
from app.ml.train import FEATURE_COLS, QUANTILES, train, _make_model
from app.ml.predict import QuantilePredictor, QuantilePrediction
from app.physics_config import PhysicsConfig
from app.simulator import EVSpecs
from app.validation import EV_SPECS


RNG = np.random.default_rng(99)

TINY_EV = EVSpecs(
    battery_capacity_kwh=40.5,
    usable_battery_kwh=38.5,
    kerb_weight_kg=1400,
    drag_coefficient=0.32,
    frontal_area_m2=2.35,
    motor_power_kw=129,
)


# ── 1. _sample_config ─────────────────────────────────────────────────────────

class TestSampleConfig:
    def test_returns_physics_config(self):
        cfg = _sample_config(np.random.default_rng(1))
        assert isinstance(cfg, PhysicsConfig)

    def test_crr_in_range(self):
        for _ in range(50):
            cfg = _sample_config(np.random.default_rng(_))
            assert 0.005 <= cfg.crr <= 0.025

    def test_eta_drive_in_range(self):
        for _ in range(50):
            cfg = _sample_config(np.random.default_rng(_))
            assert 0.80 <= cfg.eta_drive <= 0.99

    def test_eta_regen_in_range(self):
        for _ in range(50):
            cfg = _sample_config(np.random.default_rng(_))
            assert 0.10 <= cfg.eta_regen <= 0.60

    def test_p_aux_in_range(self):
        for _ in range(50):
            cfg = _sample_config(np.random.default_rng(_))
            assert 0.30 <= cfg.p_aux_kw <= 2.00

    def test_configs_vary(self):
        """Two different seeds should produce different configs."""
        c1 = _sample_config(np.random.default_rng(10))
        c2 = _sample_config(np.random.default_rng(20))
        assert c1.crr != c2.crr or c1.p_aux_kw != c2.p_aux_kw


# ── 2. _sample_trip ───────────────────────────────────────────────────────────

class TestSampleTrip:
    def test_returns_dict_with_required_keys(self):
        trip = _sample_trip(np.random.default_rng(1))
        required = {
            "distance_km", "mean_gradient_mpkm", "elevation_gain_per_km",
            "elevation_loss_per_km", "elevation_gain_m", "elevation_loss_m",
            "avg_speed_kmh", "highway_fraction", "temperature_c",
            "num_passengers", "ac_on", "driving_style", "driving_style_code",
        }
        assert required.issubset(trip.keys())

    def test_distance_in_range(self):
        for i in range(100):
            t = _sample_trip(np.random.default_rng(i))
            assert 5.0 <= t["distance_km"] <= 500.0

    def test_passengers_in_range(self):
        for i in range(100):
            t = _sample_trip(np.random.default_rng(i))
            assert 1 <= t["num_passengers"] <= 5

    def test_ac_forced_on_when_hot(self):
        """At 35 °C, AC should always be on — check via monkeypatching isn't needed;
        the function forces ac_on=1 when temp>30."""
        rng = np.random.default_rng(42)
        # Sample many trips; at least some should have temp>30 and ac_on=1
        trips = [_sample_trip(rng) for _ in range(200)]
        hot = [t for t in trips if t["temperature_c"] > 30]
        assert all(t["ac_on"] == 1 for t in hot), "AC must be on when temp > 30°C"

    def test_ac_forced_off_when_cold(self):
        trips = [_sample_trip(np.random.default_rng(i)) for i in range(500)]
        cold = [t for t in trips if t["temperature_c"] < 10]
        assert all(t["ac_on"] == 0 for t in cold), "AC must be off when temp < 10°C"

    def test_driving_style_code_matches_style(self):
        code_map = {"calm": 0, "normal": 1, "aggressive": 2}
        for i in range(50):
            t = _sample_trip(np.random.default_rng(i))
            assert t["driving_style_code"] == code_map[t["driving_style"]]

    def test_highway_fraction_zero_for_urban(self):
        trips = [_sample_trip(np.random.default_rng(i)) for i in range(300)]
        urban = [t for t in trips if t["road_class"] == "urban"]
        assert all(t["highway_fraction"] == 0.0 for t in urban)

    def test_highway_fraction_one_for_highway(self):
        trips = [_sample_trip(np.random.default_rng(i)) for i in range(300)]
        hw = [t for t in trips if t["road_class"] == "highway"]
        assert all(t["highway_fraction"] == 1.0 for t in hw)

    def test_elevation_loss_le_gain_times_factor(self):
        """loss_m ≤ gain_m (correlated but always <= gain)."""
        for i in range(100):
            t = _sample_trip(np.random.default_rng(i))
            assert t["elevation_loss_m"] <= t["elevation_gain_m"] + 1e-6


# ── 3. generate (small run) ───────────────────────────────────────────────────

class TestGenerate:
    def test_returns_dataframe(self, tmp_path):
        df = generate(n_trips=3, n_mc=4, seed=0, output_path=tmp_path / "td.csv")
        assert isinstance(df, pd.DataFrame)

    def test_row_count(self, tmp_path):
        n_evs = len(EV_SPECS)
        df = generate(n_trips=4, n_mc=3, seed=1, output_path=tmp_path / "td.csv")
        assert len(df) == n_evs * 4

    def test_all_feature_cols_present(self, tmp_path):
        df = generate(n_trips=2, n_mc=3, seed=2, output_path=tmp_path / "td.csv")
        for col in FEATURE_COLS:
            assert col in df.columns, f"Missing feature column: {col}"

    def test_target_cols_present(self, tmp_path):
        df = generate(n_trips=2, n_mc=3, seed=3, output_path=tmp_path / "td.csv")
        for col in ("p10_wh_per_km", "p50_wh_per_km", "p90_wh_per_km"):
            assert col in df.columns

    def test_quantile_ordering(self, tmp_path):
        """p10 ≤ p50 ≤ p90 for every row."""
        df = generate(n_trips=5, n_mc=10, seed=4, output_path=tmp_path / "td.csv")
        assert (df["p10_wh_per_km"] <= df["p50_wh_per_km"]).all()
        assert (df["p50_wh_per_km"] <= df["p90_wh_per_km"]).all()

    def test_no_claimed_range_km_feature(self, tmp_path):
        df = generate(n_trips=2, n_mc=3, seed=5, output_path=tmp_path / "td.csv")
        assert "claimed_range_km" not in df.columns

    def test_wh_per_km_positive(self, tmp_path):
        df = generate(n_trips=3, n_mc=5, seed=6, output_path=tmp_path / "td.csv")
        assert (df["p10_wh_per_km"] > 0).all()

    def test_csv_written(self, tmp_path):
        out = tmp_path / "td.csv"
        generate(n_trips=2, n_mc=3, seed=7, output_path=out)
        assert out.exists()
        loaded = pd.read_csv(out)
        assert len(loaded) > 0

    def test_different_evs_have_different_wh(self, tmp_path):
        """The lightest and heaviest EVs should show different mean p50."""
        df = generate(n_trips=10, n_mc=5, seed=8, output_path=tmp_path / "td.csv")
        comet  = df[df["ev_name"] == "MG Comet EV"]["p50_wh_per_km"].mean()
        harrier = df[df["ev_name"] == "Tata Harrier EV AWD"]["p50_wh_per_km"].mean()
        assert harrier > comet, "Heavier EV should consume more energy on average"

    def test_reproducible_with_same_seed(self, tmp_path):
        df1 = generate(n_trips=3, n_mc=4, seed=42, output_path=tmp_path / "a.csv")
        df2 = generate(n_trips=3, n_mc=4, seed=42, output_path=tmp_path / "b.csv")
        pd.testing.assert_frame_equal(df1, df2)


# ── 4. train (tiny synthetic dataset) ────────────────────────────────────────

def _make_tiny_df(n_evs: int = 4, n_per_ev: int = 20) -> pd.DataFrame:
    """Create a tiny synthetic dataframe with the required columns."""
    rng = np.random.default_rng(0)
    ev_pool = list(EV_SPECS.keys())[:n_evs]
    rows = []
    for ev_name in ev_pool:
        spec = EV_SPECS[ev_name]
        for _ in range(n_per_ev):
            wh = float(rng.uniform(80, 200))
            rows.append({
                "ev_name": ev_name,
                "battery_capacity_kwh": spec["battery_capacity_kwh"],
                "usable_battery_kwh":   spec["usable_battery_kwh"],
                "kerb_weight_kg":       spec["kerb_weight_kg"],
                "drag_coefficient":     spec["drag_coefficient"],
                "frontal_area_m2":      spec["frontal_area_m2"],
                "motor_power_kw":       spec["motor_power_kw"],
                "distance_km":          float(rng.uniform(20, 200)),
                "mean_gradient_mpkm":   float(rng.uniform(0, 20)),
                "elevation_gain_per_km": float(rng.uniform(0, 10)),
                "elevation_loss_per_km": float(rng.uniform(0, 10)),
                "avg_speed_kmh":        float(rng.uniform(20, 100)),
                "highway_fraction":     float(rng.uniform(0, 1)),
                "temperature_c":        float(rng.uniform(10, 38)),
                "num_passengers":       int(rng.integers(1, 5)),
                "ac_on":                int(rng.integers(0, 2)),
                "driving_style_code":   int(rng.integers(0, 3)),
                "p10_wh_per_km":        wh * 0.88,
                "p50_wh_per_km":        wh,
                "p90_wh_per_km":        wh * 1.12,
            })
    return pd.DataFrame(rows)


class TestTrain:
    def test_make_model_returns_estimator(self):
        m = _make_model(0.50)
        assert hasattr(m, "fit") and hasattr(m, "predict")

    def test_feature_cols_list_correct(self):
        """All feature columns must be valid strings."""
        assert len(FEATURE_COLS) == 16
        assert "claimed_range_km" not in FEATURE_COLS

    def test_train_produces_models_and_json(self, tmp_path):
        df = _make_tiny_df()
        csv_path = tmp_path / "td.csv"
        df.to_csv(csv_path, index=False)

        results_path = tmp_path / "benchmark_results.json"
        models_dir = tmp_path / "models"

        # Monkeypatch paths
        import app.ml.train as train_mod
        orig_models = train_mod.MODELS_DIR
        orig_results = train_mod.RESULTS_PATH
        train_mod.MODELS_DIR = models_dir
        train_mod.RESULTS_PATH = results_path
        try:
            res = train(data_path=csv_path)
        finally:
            train_mod.MODELS_DIR = orig_models
            train_mod.RESULTS_PATH = orig_results

        assert (models_dir / "quantile_models.joblib").exists()
        assert results_path.exists()
        assert "quantile_models" in res
        assert set(res["quantile_models"].keys()) == {"p10", "p50", "p90"}

    def test_train_cv_mae_reasonable(self, tmp_path):
        """CV MAE should be less than 50 Wh/km on synthetic data."""
        df = _make_tiny_df(n_evs=4, n_per_ev=30)
        csv_path = tmp_path / "td.csv"
        df.to_csv(csv_path, index=False)

        import app.ml.train as train_mod
        orig_models = train_mod.MODELS_DIR
        orig_results = train_mod.RESULTS_PATH
        train_mod.MODELS_DIR = tmp_path / "models"
        train_mod.RESULTS_PATH = tmp_path / "br.json"
        try:
            res = train(data_path=csv_path)
        finally:
            train_mod.MODELS_DIR = orig_models
            train_mod.RESULTS_PATH = orig_results

        for q in ("p10", "p50", "p90"):
            mae = res["quantile_models"][q]["cv_mae_wh_per_km"]
            assert mae < 50.0, f"{q} CV MAE={mae:.2f} is unreasonably high"


# ── 5. QuantilePredictor ──────────────────────────────────────────────────────

class TestQuantilePredictor:
    """Test the predictor with and without a real model on disk."""

    def _predictor_with_model(self, tmp_path):
        """Return a predictor backed by a real (tiny) trained model."""
        df = _make_tiny_df(n_evs=4, n_per_ev=25)
        csv_path = tmp_path / "td.csv"
        df.to_csv(csv_path, index=False)

        import app.ml.train as train_mod
        import app.ml.predict as predict_mod

        orig_train_models = train_mod.MODELS_DIR
        orig_pred_models  = predict_mod.MODELS_DIR
        orig_results      = train_mod.RESULTS_PATH

        train_mod.MODELS_DIR  = tmp_path / "models"
        predict_mod.MODELS_DIR = tmp_path / "models"
        train_mod.RESULTS_PATH = tmp_path / "br.json"

        try:
            train(data_path=csv_path)
            predictor = QuantilePredictor()
        finally:
            train_mod.MODELS_DIR  = orig_train_models
            predict_mod.MODELS_DIR = orig_pred_models
            train_mod.RESULTS_PATH = orig_results

        return predictor

    def test_fallback_when_no_model(self):
        """Predictor without a model file must still return a result."""
        import app.ml.predict as predict_mod
        orig = predict_mod.MODELS_DIR
        predict_mod.MODELS_DIR = Path("/nonexistent/path")
        try:
            p = QuantilePredictor()
            assert not p.is_ready
            result = p.predict(
                ev=TINY_EV, distance_km=100, mean_gradient_mpkm=5,
                elevation_gain_per_km=2.5, elevation_loss_per_km=2.0,
                avg_speed_kmh=60, highway_fraction=0.4,
                temperature_c=28, num_passengers=2, ac_on=True,
            )
        finally:
            predict_mod.MODELS_DIR = orig
        assert isinstance(result, QuantilePrediction)
        assert result.source == "physics_fallback"
        assert len(result.warnings) > 0

    def test_fallback_quantile_ordering(self):
        import app.ml.predict as predict_mod
        orig = predict_mod.MODELS_DIR
        predict_mod.MODELS_DIR = Path("/nonexistent/path")
        try:
            p = QuantilePredictor()
            result = p.predict(
                ev=TINY_EV, distance_km=100, mean_gradient_mpkm=5,
                elevation_gain_per_km=2.5, elevation_loss_per_km=2.0,
                avg_speed_kmh=60, highway_fraction=0.4,
                temperature_c=28, num_passengers=2, ac_on=True,
            )
        finally:
            predict_mod.MODELS_DIR = orig
        assert result.p10_wh_per_km <= result.p50_wh_per_km <= result.p90_wh_per_km

    def test_ml_prediction_returns_quantile_prediction(self, tmp_path):
        predictor = self._predictor_with_model(tmp_path)
        assert predictor.is_ready

        result = predictor.predict(
            ev=TINY_EV, distance_km=100, mean_gradient_mpkm=5,
            elevation_gain_per_km=2.5, elevation_loss_per_km=2.0,
            avg_speed_kmh=60, highway_fraction=0.4,
            temperature_c=28, num_passengers=2, ac_on=True,
        )
        assert isinstance(result, QuantilePrediction)
        assert result.source == "ml"
        assert result.p10_wh_per_km > 0
        assert result.p50_wh_per_km > 0
        assert result.p90_wh_per_km > 0

    def test_ml_quantile_ordering_enforced(self, tmp_path):
        predictor = self._predictor_with_model(tmp_path)
        result = predictor.predict(
            ev=TINY_EV, distance_km=100, mean_gradient_mpkm=5,
            elevation_gain_per_km=2.5, elevation_loss_per_km=2.0,
            avg_speed_kmh=60, highway_fraction=0.4,
            temperature_c=28, num_passengers=2, ac_on=True,
        )
        assert result.p10_wh_per_km <= result.p50_wh_per_km
        assert result.p50_wh_per_km <= result.p90_wh_per_km

    def test_heavier_ev_predicted_higher(self, tmp_path):
        """For same trip, heavier EV should get higher p50."""
        predictor = self._predictor_with_model(tmp_path)

        light_ev = EVSpecs(
            battery_capacity_kwh=17.3, usable_battery_kwh=16.44,
            kerb_weight_kg=1045, drag_coefficient=0.35,
            frontal_area_m2=2.05, motor_power_kw=42,
        )
        heavy_ev = EVSpecs(
            battery_capacity_kwh=75.0, usable_battery_kwh=71.25,
            kerb_weight_kg=2150, drag_coefficient=0.33,
            frontal_area_m2=2.65, motor_power_kw=170,
        )
        kw = dict(
            distance_km=100, mean_gradient_mpkm=5,
            elevation_gain_per_km=2.5, elevation_loss_per_km=2.0,
            avg_speed_kmh=60, highway_fraction=0.4,
            temperature_c=28, num_passengers=2, ac_on=True,
        )
        r_light = predictor.predict(ev=light_ev, **kw)
        r_heavy = predictor.predict(ev=heavy_ev, **kw)
        # Heavier should predict higher (note: tiny model may not be well-calibrated,
        # but it should learn the direction from the physics-based training data)
        assert r_heavy.p50_wh_per_km > r_light.p50_wh_per_km
