"""
Phase 4 tests: validation and calibration pipeline.

Tests cover:
- validation_cases.csv loading (correct fields, row count, set membership)
- prediction function (single EV, fixed scenario)
- calibration (optimizer finds params that reduce RMSE vs prior)
- metrics (MAE, MAPE)
- validation_results.json structure (all required fields present)
- Gaadify cross-check (structure only, not value)
- Sensitivity runs (three mixes reported)
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from app.validation import (
    CALIBRATION_CASES,
    EV_SPECS,
    SCENARIO,
    VALIDATION_CSV,
    VALIDATION_JSON,
    _calibrate,
    _gaadify_cross_check,
    _load_validation_cases,
    _metrics,
    _per_case_result,
    _predict_wh_per_km,
    _sensitivity,
    _spearman,
    run_validation,
)
from app.physics_config import PhysicsConfig

PRIOR_CONFIG = PhysicsConfig()   # p_aux=0.25, eta_regen=0.65


# ── 1. Validation CSV ─────────────────────────────────────────────────────────

class TestValidationCSV:
    def test_csv_exists(self):
        assert VALIDATION_CSV.exists(), f"validation_cases.csv not found at {VALIDATION_CSV}"

    def test_loads_16_autocar_cases(self):
        cases = _load_validation_cases()
        autocar = [c for c in cases]
        assert len(autocar) == 16, f"Expected 16 Autocar cases, got {len(autocar)}"

    def test_all_have_measured_wh_per_km(self):
        for c in _load_validation_cases():
            assert c["measured_wh_per_km"] > 0, f"{c['ev_name']} has non-positive Wh/km"

    def test_five_calibration_cases(self):
        cases = _load_validation_cases()
        calib = [c for c in cases if c["ev_name"] in CALIBRATION_CASES]
        assert len(calib) == 5, f"Expected 5 calibration cases, got {len(calib)}"

    def test_eleven_test_cases(self):
        cases = _load_validation_cases()
        test = [c for c in cases if c["ev_name"] not in CALIBRATION_CASES]
        assert len(test) == 11, f"Expected 11 test cases, got {len(test)}"

    def test_calibration_set_contains_expected_evs(self):
        expected = {
            "MG Comet EV",
            "Tata Tiago EV 24 kWh",
            "Tata Nexon EV 45 kWh",
            "Tata Curvv EV",
            "Tata Harrier EV AWD",
        }
        assert expected == CALIBRATION_CASES

    def test_set_field_is_calibration_or_test(self):
        for c in _load_validation_cases():
            assert c["set"] in ("calibration", "test"), (
                f"{c['ev_name']} has invalid set '{c['set']}'"
            )

    def test_measured_wh_per_km_matches_km_per_kwh(self):
        """measured_wh_per_km should equal 1000 / km_per_kwh (within floating-point rounding)."""
        cases = _load_validation_cases()
        for c in cases:
            if c["km_per_kwh"] is not None and c["km_per_kwh"] > 0:
                expected = 1000.0 / c["km_per_kwh"]
                assert abs(c["measured_wh_per_km"] - expected) < 1.0, (
                    f"{c['ev_name']}: Wh/km={c['measured_wh_per_km']:.2f} "
                    f"but 1000/km_per_kWh={expected:.2f}"
                )

    def test_all_ev_names_in_specs_dict(self):
        """Every EV in the CSV must have physics specs in EV_SPECS."""
        for c in _load_validation_cases():
            assert c["ev_name"] in EV_SPECS, (
                f"No physics specs for '{c['ev_name']}' in EV_SPECS"
            )


# ── 2. Prediction function ────────────────────────────────────────────────────

class TestPredictWh:
    def test_returns_positive(self):
        result = _predict_wh_per_km("Tata Tiago EV 24 kWh", PRIOR_CONFIG)
        assert result > 0.0

    def test_heavier_ev_higher_wh_per_km(self):
        """Heavier EV should use more energy at the same speed on flat road."""
        light = _predict_wh_per_km("MG Comet EV", PRIOR_CONFIG)      # 1045 kg
        heavy = _predict_wh_per_km("Tata Harrier EV AWD", PRIOR_CONFIG)  # 2150 kg
        assert heavy > light

    def test_higher_p_aux_higher_wh(self):
        """More auxiliary power always increases Wh/km."""
        low_aux = _predict_wh_per_km("Tata Nexon EV 45 kWh", PhysicsConfig(p_aux_kw=0.0))
        hi_aux  = _predict_wh_per_km("Tata Nexon EV 45 kWh", PhysicsConfig(p_aux_kw=1.0))
        assert hi_aux > low_aux

    def test_city_fraction_affects_wh(self):
        """More city fraction → more stop-and-go → different Wh/km."""
        mostly_city = _predict_wh_per_km("Tata Nexon EV 45 kWh", PRIOR_CONFIG, city_fraction=0.70)
        mostly_hw   = _predict_wh_per_km("Tata Nexon EV 45 kWh", PRIOR_CONFIG, city_fraction=0.30)
        # They must differ (highway at 80 km/h has more aero, city has more stop-and-go)
        assert abs(mostly_city - mostly_hw) > 1.0


# ── 3. Calibration ────────────────────────────────────────────────────────────

class TestCalibration:
    def test_calibration_returns_two_params(self):
        cases = _load_validation_cases()
        calib = [c for c in cases if c["ev_name"] in CALIBRATION_CASES]
        p_aux, eta_regen = _calibrate(calib)
        assert isinstance(p_aux, float)
        assert isinstance(eta_regen, float)

    def test_calibrated_rmse_below_prior_rmse(self):
        """Calibrated params must give lower RMSE on calibration set than the prior."""
        from app.validation import _objective
        cases = _load_validation_cases()
        calib = [c for c in cases if c["ev_name"] in CALIBRATION_CASES]
        prior_rmse = _objective([0.25, 0.65], calib)
        p_aux, eta_regen = _calibrate(calib)
        calib_rmse = _objective([p_aux, eta_regen], calib)
        assert calib_rmse < prior_rmse, (
            f"Calibrated RMSE ({calib_rmse:.2f}) not below prior RMSE ({prior_rmse:.2f})"
        )

    def test_calibrated_params_in_bounds(self):
        cases = _load_validation_cases()
        calib = [c for c in cases if c["ev_name"] in CALIBRATION_CASES]
        p_aux, eta_regen = _calibrate(calib)
        assert 0.0 <= p_aux <= 2.0
        assert 0.30 <= eta_regen <= 0.90


# ── 4. Metrics ────────────────────────────────────────────────────────────────

class TestMetrics:
    SAMPLE_RESULTS = [
        {"predicted_wh_per_km": 100.0, "measured_wh_per_km": 90.0,
         "pct_error": 11.11, "claimed_wh_per_km": 80.0, "claimed_pct_error": -11.11},
        {"predicted_wh_per_km": 120.0, "measured_wh_per_km": 130.0,
         "pct_error": -7.69, "claimed_wh_per_km": 110.0, "claimed_pct_error": -15.38},
    ]

    def test_mae_wh_per_km(self):
        m = _metrics(self.SAMPLE_RESULTS)
        # MAE = (|100-90| + |120-130|) / 2 = (10 + 10) / 2 = 10
        assert abs(m["mae_wh_per_km"] - 10.0) < 0.01

    def test_mape_pct(self):
        m = _metrics(self.SAMPLE_RESULTS)
        # MAPE = (|11.11| + |-7.69|) / 2
        expected = (11.11 + 7.69) / 2
        assert abs(m["mape_pct"] - expected) < 0.1


# ── 5. Spearman ───────────────────────────────────────────────────────────────

class TestSpearman:
    def test_perfect_rank_correlation(self):
        results = [
            {"measured_wh_per_km": 80.0, "predicted_wh_per_km": 78.0},
            {"measured_wh_per_km": 100.0, "predicted_wh_per_km": 105.0},
            {"measured_wh_per_km": 130.0, "predicted_wh_per_km": 128.0},
            {"measured_wh_per_km": 160.0, "predicted_wh_per_km": 165.0},
        ]
        s = _spearman(results)
        assert s["rho"] == pytest.approx(1.0, abs=0.01)

    def test_rho_between_neg1_and_1(self):
        cases = _load_validation_cases()
        config = PhysicsConfig(p_aux_kw=1.0, eta_regen=0.30)
        results = [_per_case_result(c, config) for c in cases]
        s = _spearman(results)
        assert -1.0 <= s["rho"] <= 1.0

    def test_positive_correlation_with_calibrated_config(self):
        cases = _load_validation_cases()
        config = PhysicsConfig(p_aux_kw=1.0, eta_regen=0.30)
        results = [_per_case_result(c, config) for c in cases]
        s = _spearman(results)
        assert s["rho"] > 0.5, f"Expected positive Spearman correlation, got {s['rho']}"


# ── 6. Sensitivity ────────────────────────────────────────────────────────────

class TestSensitivity:
    def test_returns_three_scenarios(self):
        config = PhysicsConfig(p_aux_kw=1.0, eta_regen=0.30)
        s = _sensitivity(config)
        assert len(s) == 3

    def test_has_required_keys(self):
        config = PhysicsConfig(p_aux_kw=1.0, eta_regen=0.30)
        for entry in _sensitivity(config):
            assert "label" in entry
            assert "city_fraction" in entry
            assert "mape_pct" in entry
            assert "mae_wh_per_km" in entry

    def test_city_fractions_correct(self):
        config = PhysicsConfig(p_aux_kw=1.0, eta_regen=0.30)
        fracs = {s["city_fraction"] for s in _sensitivity(config)}
        assert fracs == {0.30, 0.50, 0.70}


# ── 7. Gaadify cross-check ────────────────────────────────────────────────────

class TestGaadify:
    def test_returns_required_keys(self):
        config = PhysicsConfig(p_aux_kw=1.0, eta_regen=0.30)
        cc = _gaadify_cross_check(config)
        for key in ("source", "source_url", "ev_name", "tested_km_approx",
                    "measured_wh_per_km", "predicted_wh_per_km", "pct_error"):
            assert key in cc, f"Missing key '{key}' in Gaadify cross-check"

    def test_measured_wh_based_on_185km(self):
        config = PhysicsConfig(p_aux_kw=1.0, eta_regen=0.30)
        cc = _gaadify_cross_check(config)
        # measured_wh_per_km = 24000 / 185 ≈ 129.73
        assert abs(cc["measured_wh_per_km"] - 24000.0 / 185.0) < 0.1

    def test_predicted_is_positive(self):
        config = PhysicsConfig(p_aux_kw=1.0, eta_regen=0.30)
        cc = _gaadify_cross_check(config)
        assert cc["predicted_wh_per_km"] > 0.0


# ── 8. Full run_validation pipeline ──────────────────────────────────────────

class TestRunValidation:
    """Integration test: run_validation() must produce a valid JSON file."""

    def test_produces_json_file(self, tmp_path, monkeypatch):
        """run_validation writes validation_results.json."""
        import app.validation as val_mod
        monkeypatch.setattr(val_mod, "VALIDATION_JSON", tmp_path / "validation_results.json")
        result = val_mod.run_validation()
        assert (tmp_path / "validation_results.json").exists()

    def test_output_has_required_top_level_keys(self, tmp_path, monkeypatch):
        import app.validation as val_mod
        monkeypatch.setattr(val_mod, "VALIDATION_JSON", tmp_path / "v.json")
        result = val_mod.run_validation()
        required = {
            "scenario", "calibration_params", "calibration_set", "test_set",
            "claimed_baseline", "spearman_rank_correlation", "sensitivity",
            "cross_check_gaadify", "results", "all_cases_combined",
        }
        for key in required:
            assert key in result, f"Missing top-level key '{key}'"

    def test_results_has_16_entries(self, tmp_path, monkeypatch):
        import app.validation as val_mod
        monkeypatch.setattr(val_mod, "VALIDATION_JSON", tmp_path / "v.json")
        result = val_mod.run_validation()
        assert len(result["results"]) == 16

    def test_calibration_mape_below_prior(self, tmp_path, monkeypatch):
        """Calibrated MAPE on calibration set must be below the prior MAPE."""
        import app.validation as val_mod
        monkeypatch.setattr(val_mod, "VALIDATION_JSON", tmp_path / "v.json")

        # Prior MAPE (no calibration)
        prior_cfg = PhysicsConfig()
        cases = _load_validation_cases()
        calib_cases = [c for c in cases if c["ev_name"] in CALIBRATION_CASES]
        prior_results = [_per_case_result(c, prior_cfg) for c in calib_cases]
        prior_mape = _metrics(prior_results)["mape_pct"]

        result = val_mod.run_validation()
        calib_mape = result["calibration_set"]["mape_pct"]
        assert calib_mape < prior_mape, (
            f"Calibrated MAPE ({calib_mape:.1f}%) should be below prior ({prior_mape:.1f}%)"
        )

    def test_spearman_rho_positive(self, tmp_path, monkeypatch):
        import app.validation as val_mod
        monkeypatch.setattr(val_mod, "VALIDATION_JSON", tmp_path / "v.json")
        result = val_mod.run_validation()
        assert result["spearman_rank_correlation"]["rho"] > 0.5

    def test_per_case_has_required_fields(self, tmp_path, monkeypatch):
        import app.validation as val_mod
        monkeypatch.setattr(val_mod, "VALIDATION_JSON", tmp_path / "v.json")
        result = val_mod.run_validation()
        required_fields = {
            "ev_name", "set", "measured_wh_per_km", "predicted_wh_per_km",
            "pct_error", "claimed_wh_per_km", "claimed_pct_error",
            "source_url", "data_verified",
        }
        for r in result["results"]:
            for f in required_fields:
                assert f in r, f"Missing field '{f}' in result for {r.get('ev_name')}"

    def test_sensitivity_has_baseline_scenario(self, tmp_path, monkeypatch):
        import app.validation as val_mod
        monkeypatch.setattr(val_mod, "VALIDATION_JSON", tmp_path / "v.json")
        result = val_mod.run_validation()
        labels = [s["label"] for s in result["sensitivity"]]
        assert any("50" in label for label in labels), "50/50 baseline must be in sensitivity"
