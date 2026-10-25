"""
Phase 6 tests: recommendation endpoint and pipeline.

All external HTTP calls are mocked (OSRM, Open-Elevation, OpenWeatherMap,
Nominatim). The ML predictor is replaced with a stub that returns fixed
quantiles so tests are deterministic.
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.database import get_db
from app.models import Base, EVModel
from app.schemas import RecommendRequest, RouteCoords
from app.services.recommend_service import (
    _carbon,
    _evaluate_ev,
    _verdict,
    ELECTRICITY_PRICE_INR_PER_KWH,
    GRID_EMISSION_KG_CO2_PER_KWH,
    PETROL_EMISSION_KG_CO2_PER_L,
)
from app.simulator import EVSpecs, SegmentData


# ── Test DB fixture ────────────────────────────────────────────────────────────

@pytest.fixture
def db():
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,   # all threads share one in-memory connection
    )
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()

    # Seed one EV per relevant body type for testing
    evs = [
        EVModel(
            model_name="MG Comet EV", body_type="hatchback", seats=4,
            claimed_range_km=230, claimed_cycle="MIDC", battery_capacity_kwh=17.3,
            usable_battery_kwh=16.44, kerb_weight_kg=1045, drag_coefficient=0.35,
            frontal_area_m2=2.05, motor_power_kw=42, drag_coefficient_assumed=True,
            frontal_area_assumed=True, usable_battery_assumed=True,
            drivetrain_eff_assumed=True, regen_eff_assumed=True, price_assumed=True,
            spec_source_url="https://example.com",
        ),
        EVModel(
            model_name="Tata Nexon EV 45 kWh", body_type="compact_suv", seats=5,
            claimed_range_km=489, claimed_cycle="MIDC", battery_capacity_kwh=45.0,
            usable_battery_kwh=42.75, kerb_weight_kg=1410, drag_coefficient=0.32,
            frontal_area_m2=2.35, motor_power_kw=130, drag_coefficient_assumed=True,
            frontal_area_assumed=True, usable_battery_assumed=True,
            drivetrain_eff_assumed=True, regen_eff_assumed=True, price_assumed=True,
            spec_source_url="https://example.com",
        ),
        EVModel(
            model_name="Tata Harrier EV AWD", body_type="suv", seats=5,
            claimed_range_km=622, claimed_cycle="MIDC", battery_capacity_kwh=75.0,
            usable_battery_kwh=71.25, kerb_weight_kg=2150, drag_coefficient=0.33,
            frontal_area_m2=2.65, motor_power_kw=170, drag_coefficient_assumed=True,
            frontal_area_assumed=True, usable_battery_assumed=True,
            drivetrain_eff_assumed=True, regen_eff_assumed=True, price_assumed=True,
            spec_source_url="https://example.com",
        ),
    ]
    for ev in evs:
        session.add(ev)
    session.commit()

    yield session
    session.close()


@pytest.fixture
def client(db):
    """
    Minimal FastAPI app (no lifespan) with DB overridden to the in-memory
    test session.  This matches the Phase 1/2 test pattern and avoids the
    lifespan's seed_ev_models call touching the real rangeiq.db file.
    """
    from fastapi import FastAPI
    from app.api.routes import router as api_router

    test_app = FastAPI()
    test_app.include_router(api_router, prefix="/api")
    test_app.dependency_overrides[get_db] = lambda: db

    with TestClient(test_app) as c:
        yield c


# ── Mock route info ────────────────────────────────────────────────────────────

MOCK_ROUTE_INFO = {
    "distance_km": 150.0,
    "duration_s": 7200.0,
    "avg_speed_kmh": 75.0,
    "elevation_gain_m": 500.0,
    "elevation_loss_m": 450.0,
    "start_elevation_m": 560.0,
    "end_elevation_m": 10.0,
    "temperature_c": 28.0,
    "waypoints": [{"lat": 18.52, "lon": 73.86}, {"lat": 19.07, "lon": 72.87}],
    "profile": [{"distance_km": 0, "elevation_m": 560, "gradient": 0}],
    "gradient_segments": [(0.0, 500_000.0 / 20)] * 20,   # 20 flat-ish segments
    "fallback_warnings": [],
    "route_fallback": False,
    "elevation_fallback": False,
    "weather_fallback": False,
}


def _mock_ml_pred():
    """Return a fixed ML prediction stub."""
    from app.ml.predict import QuantilePrediction
    pred = QuantilePrediction(
        p10_wh_per_km=110.0,   # efficient case (→ longer range)
        p50_wh_per_km=135.0,   # expected
        p90_wh_per_km=165.0,   # worst case (→ shorter range)
        source="ml",
    )
    return pred


# ── 1. Utility functions ──────────────────────────────────────────────────────

class TestVerdict:
    def test_suitable_when_p10_ge_required(self):
        assert _verdict(p10_range=200, p50_range=250, required_km=190) == "Suitable"

    def test_marginal_when_p50_ge_required_p10_lt(self):
        assert _verdict(p10_range=100, p50_range=200, required_km=150) == "Marginal"

    def test_not_suitable_when_p50_lt_required(self):
        assert _verdict(p10_range=80, p50_range=120, required_km=150) == "Not suitable"

    def test_exactly_at_boundary_suitable(self):
        assert _verdict(p10_range=150, p50_range=200, required_km=150) == "Suitable"

    def test_exactly_at_boundary_marginal(self):
        assert _verdict(p10_range=140, p50_range=150, required_km=150) == "Marginal"


class TestCarbon:
    def test_ev_co2_calculated(self):
        c = _carbon(trip_energy_kwh=30.0, trip_distance_km=150.0)
        expected_ev_co2 = 30.0 * GRID_EMISSION_KG_CO2_PER_KWH
        assert abs(c.ev_co2_kg - expected_ev_co2) < 0.01

    def test_petrol_co2_calculated(self):
        c = _carbon(trip_energy_kwh=30.0, trip_distance_km=150.0)
        litres = 150.0 / 15.0   # PETROL_KM_PER_L = 15
        expected = litres * PETROL_EMISSION_KG_CO2_PER_L
        assert abs(c.petrol_co2_kg - expected) < 0.01

    def test_ev_cost_calculated(self):
        c = _carbon(trip_energy_kwh=30.0, trip_distance_km=150.0)
        expected = 30.0 * ELECTRICITY_PRICE_INR_PER_KWH
        assert abs(c.ev_cost_inr - expected) < 0.01

    def test_label_is_illustrative(self):
        c = _carbon(10.0, 100.0)
        assert c.label == "illustrative"


# ── 2. _evaluate_ev ───────────────────────────────────────────────────────────

class TestEvaluateEv:
    def _fake_req(self):
        class R:
            driving_style = "normal"
            num_passengers = 2
            ac_on = True
            round_trip = False
        return R()

    def test_returns_ev_recommend_result(self, db):
        from app.schemas import EVRecommendResult
        ev_row = db.query(EVModel).filter(EVModel.model_name == "Tata Nexon EV 45 kWh").first()
        segs = [SegmentData(gradient=0.0, length_m=7500.0)] * 20
        route_info = dict(MOCK_ROUTE_INFO)

        with patch("app.services.recommend_service._get_predictor") as mock_pred:
            mock_pred.return_value = MagicMock(predict=lambda **kw: _mock_ml_pred())
            result = _evaluate_ev(
                ev_row, segs, route_info, self._fake_req(),
                required_km=165.0, trip_distance_km=150.0, warnings_out=[],
            )
        assert isinstance(result, EVRecommendResult)
        assert result.model_name == "Tata Nexon EV 45 kWh"

    def test_verdict_not_suitable_for_short_range_ev(self, db):
        ev_row = db.query(EVModel).filter(EVModel.model_name == "MG Comet EV").first()
        segs = [SegmentData(gradient=0.0, length_m=7500.0)] * 20
        route_info = dict(MOCK_ROUTE_INFO)

        with patch("app.services.recommend_service._get_predictor") as mock_pred:
            mock_pred.return_value = MagicMock(predict=lambda **kw: _mock_ml_pred())
            result = _evaluate_ev(
                ev_row, segs, route_info, self._fake_req(),
                required_km=165.0, trip_distance_km=150.0, warnings_out=[],
            )
        # MG Comet: usable=16.44 kWh / 0.135 kWh/km ≈ 121 km range → Not suitable for 165 km
        assert result.verdict == "Not suitable"

    def test_quantile_ordering_in_predicted_range(self, db):
        ev_row = db.query(EVModel).filter(EVModel.model_name == "Tata Nexon EV 45 kWh").first()
        segs = [SegmentData(gradient=0.0, length_m=7500.0)] * 20
        route_info = dict(MOCK_ROUTE_INFO)

        with patch("app.services.recommend_service._get_predictor") as mock_pred:
            mock_pred.return_value = MagicMock(predict=lambda **kw: _mock_ml_pred())
            result = _evaluate_ev(
                ev_row, segs, route_info, self._fake_req(),
                required_km=165.0, trip_distance_km=150.0, warnings_out=[],
            )
        # p10 (conservative) ≤ p50 (expected) ≤ p90 (optimistic)
        assert result.predicted_range.p10 <= result.predicted_range.p50
        assert result.predicted_range.p50 <= result.predicted_range.p90

    def test_trip_energy_positive(self, db):
        ev_row = db.query(EVModel).filter(EVModel.model_name == "Tata Nexon EV 45 kWh").first()
        segs = [SegmentData(gradient=0.0, length_m=7500.0)] * 20
        route_info = dict(MOCK_ROUTE_INFO)

        with patch("app.services.recommend_service._get_predictor") as mock_pred:
            mock_pred.return_value = MagicMock(predict=lambda **kw: _mock_ml_pred())
            result = _evaluate_ev(
                ev_row, segs, route_info, self._fake_req(),
                required_km=165.0, trip_distance_km=150.0, warnings_out=[],
            )
        assert result.trip_energy_kwh > 0.0

    def test_round_trip_larger_required_km(self, db):
        ev_row = db.query(EVModel).filter(EVModel.model_name == "Tata Nexon EV 45 kWh").first()
        segs = [SegmentData(gradient=0.0, length_m=7500.0)] * 20
        route_info = dict(MOCK_ROUTE_INFO)

        class _OneWay:
            driving_style = "normal"
            num_passengers = 2
            ac_on = True
            round_trip = False

        class _RoundTrip:
            driving_style = "normal"
            num_passengers = 2
            ac_on = True
            round_trip = True

        with patch("app.services.recommend_service._get_predictor") as mock_pred:
            mock_pred.return_value = MagicMock(predict=lambda **kw: _mock_ml_pred())
            r_one = _evaluate_ev(
                ev_row, segs, route_info, _OneWay(),
                required_km=165.0, trip_distance_km=150.0, warnings_out=[],
            )
            r_rt = _evaluate_ev(
                ev_row, segs, route_info, _RoundTrip(),
                required_km=330.0, trip_distance_km=300.0, warnings_out=[],
            )
        assert r_rt.trip_energy_kwh > r_one.trip_energy_kwh


# ── 3. POST /api/recommend via TestClient ─────────────────────────────────────

class TestRecommendEndpoint:
    """Integration tests — mock all external HTTP calls."""

    def _base_body(self):
        return {
            "route": {
                "start_lat": 18.5204, "start_lon": 73.8567,
                "end_lat": 19.0760, "end_lon": 72.8777,
            },
            "driving_style": "normal",
            "num_passengers": 2,
            "ac_on": True,
            "reserve_pct": 10.0,
            "round_trip": False,
        }

    def test_returns_200(self, client):
        with patch("app.services.recommend_service.get_full_route",
                   new=AsyncMock(return_value=MOCK_ROUTE_INFO)), \
             patch("app.services.recommend_service._get_predictor") as mp:
            mp.return_value = MagicMock(predict=lambda **kw: _mock_ml_pred())
            resp = client.post("/api/recommend", json=self._base_body())
        assert resp.status_code == 200

    def test_response_has_required_fields(self, client):
        with patch("app.services.recommend_service.get_full_route",
                   new=AsyncMock(return_value=MOCK_ROUTE_INFO)), \
             patch("app.services.recommend_service._get_predictor") as mp:
            mp.return_value = MagicMock(predict=lambda **kw: _mock_ml_pred())
            resp = client.post("/api/recommend", json=self._base_body())
        data = resp.json()
        for key in ("scan_id", "trip_distance_km", "required_km",
                    "feasible_count", "recommendations"):
            assert key in data, f"Missing key: {key}"

    def test_recommendations_sorted_by_verdict_then_margin(self, client):
        with patch("app.services.recommend_service.get_full_route",
                   new=AsyncMock(return_value=MOCK_ROUTE_INFO)), \
             patch("app.services.recommend_service._get_predictor") as mp:
            mp.return_value = MagicMock(predict=lambda **kw: _mock_ml_pred())
            resp = client.post("/api/recommend", json=self._base_body())
        recs = resp.json()["recommendations"]
        tier = {"Suitable": 0, "Marginal": 1, "Not suitable": 2}
        for i in range(len(recs) - 1):
            t_curr = tier[recs[i]["verdict"]]
            t_next = tier[recs[i + 1]["verdict"]]
            if t_curr == t_next:
                assert recs[i]["range_margin_km"] >= recs[i + 1]["range_margin_km"], \
                    "Within same tier, should be sorted by descending margin"
            else:
                assert t_curr <= t_next, "Tiers must be in Suitable→Marginal→Not suitable order"

    def test_body_type_filter(self, client):
        body = self._base_body()
        body["body_type"] = "hatchback"
        with patch("app.services.recommend_service.get_full_route",
                   new=AsyncMock(return_value=MOCK_ROUTE_INFO)), \
             patch("app.services.recommend_service._get_predictor") as mp:
            mp.return_value = MagicMock(predict=lambda **kw: _mock_ml_pred())
            resp = client.post("/api/recommend", json=body)
        recs = resp.json()["recommendations"]
        assert all(r["body_type"] == "hatchback" for r in recs)

    def test_reserve_pct_affects_required_km(self, client):
        body_10 = {**self._base_body(), "reserve_pct": 10.0}
        body_20 = {**self._base_body(), "reserve_pct": 20.0}
        with patch("app.services.recommend_service.get_full_route",
                   new=AsyncMock(return_value=MOCK_ROUTE_INFO)), \
             patch("app.services.recommend_service._get_predictor") as mp:
            mp.return_value = MagicMock(predict=lambda **kw: _mock_ml_pred())
            r10 = client.post("/api/recommend", json=body_10).json()
            r20 = client.post("/api/recommend", json=body_20).json()
        assert r20["required_km"] > r10["required_km"]

    def test_round_trip_doubles_trip_distance(self, client):
        body_ow = {**self._base_body(), "round_trip": False}
        body_rt = {**self._base_body(), "round_trip": True}
        with patch("app.services.recommend_service.get_full_route",
                   new=AsyncMock(return_value=MOCK_ROUTE_INFO)), \
             patch("app.services.recommend_service._get_predictor") as mp:
            mp.return_value = MagicMock(predict=lambda **kw: _mock_ml_pred())
            r_ow = client.post("/api/recommend", json=body_ow).json()
            r_rt = client.post("/api/recommend", json=body_rt).json()
        assert abs(r_rt["trip_distance_km"] - 2 * r_ow["trip_distance_km"]) < 0.1

    def test_per_result_has_energy_breakdown(self, client):
        with patch("app.services.recommend_service.get_full_route",
                   new=AsyncMock(return_value=MOCK_ROUTE_INFO)), \
             patch("app.services.recommend_service._get_predictor") as mp:
            mp.return_value = MagicMock(predict=lambda **kw: _mock_ml_pred())
            resp = client.post("/api/recommend", json=self._base_body())
        for r in resp.json()["recommendations"]:
            bd = r["energy_breakdown"]
            for term in ("aero_wh_per_km", "rolling_wh_per_km", "gradient_wh_per_km",
                         "inertia_wh_per_km", "regen_wh_per_km", "auxiliary_wh_per_km",
                         "hvac_wh_per_km", "total_wh_per_km"):
                assert term in bd, f"Missing breakdown term: {term}"

    def test_fallback_warnings_surfaced(self, client):
        route_with_warning = dict(MOCK_ROUTE_INFO,
                                  fallback_warnings=["Weather API unavailable; using season average."],
                                  weather_fallback=True)
        with patch("app.services.recommend_service.get_full_route",
                   new=AsyncMock(return_value=route_with_warning)), \
             patch("app.services.recommend_service._get_predictor") as mp:
            mp.return_value = MagicMock(predict=lambda **kw: _mock_ml_pred())
            resp = client.post("/api/recommend", json=self._base_body())
        assert len(resp.json()["fallback_warnings"]) > 0

    def test_scan_id_returned_and_persisted(self, client, db):
        with patch("app.services.recommend_service.get_full_route",
                   new=AsyncMock(return_value=MOCK_ROUTE_INFO)), \
             patch("app.services.recommend_service._get_predictor") as mp:
            mp.return_value = MagicMock(predict=lambda **kw: _mock_ml_pred())
            resp = client.post("/api/recommend", json=self._base_body())
        scan_id = resp.json()["scan_id"]
        assert scan_id is not None
        # Verify scan is in DB
        from app.models import PredictionScan
        scan = db.query(PredictionScan).filter(PredictionScan.id == scan_id).first()
        assert scan is not None

    def test_null_price_warning_when_price_filter_applied(self, client):
        body = {**self._base_body(), "price_min_lakh": 10.0, "price_max_lakh": 50.0}
        with patch("app.services.recommend_service.get_full_route",
                   new=AsyncMock(return_value=MOCK_ROUTE_INFO)), \
             patch("app.services.recommend_service._get_predictor") as mp:
            mp.return_value = MagicMock(predict=lambda **kw: _mock_ml_pred())
            resp = client.post("/api/recommend", json=body)
        # All prices are NULL → warning should appear
        warnings = resp.json()["fallback_warnings"]
        assert any("price" in w.lower() or "NULL" in w for w in warnings)


# ── 4. POST /api/predict ──────────────────────────────────────────────────────

class TestPredictEndpoint:
    def test_predict_with_distance_only(self, client, db):
        ev = db.query(EVModel).filter(EVModel.model_name == "Tata Nexon EV 45 kWh").first()
        with patch("app.services.recommend_service._get_predictor") as mp:
            mp.return_value = MagicMock(predict=lambda **kw: _mock_ml_pred())
            resp = client.post("/api/predict", json={
                "ev_model_ids": [ev.id],
                "daily_distance_km": 100.0,
            })
        assert resp.status_code == 200
        data = resp.json()
        assert "predictions" in data
        assert len(data["predictions"]) == 1

    def test_predict_unknown_ev_returns_404(self, client):
        resp = client.post("/api/predict", json={
            "ev_model_ids": [99999],
            "daily_distance_km": 100.0,
        })
        assert resp.status_code == 404

    def test_predict_missing_both_distance_and_route_returns_400(self, client, db):
        ev = db.query(EVModel).filter(EVModel.model_name == "Tata Nexon EV 45 kWh").first()
        resp = client.post("/api/predict", json={"ev_model_ids": [ev.id]})
        assert resp.status_code in (400, 422)


# ── 5. GET /api/geocode ───────────────────────────────────────────────────────

class TestGeocodeEndpoint:
    def test_geocode_success(self, client):
        with patch("app.api.routes.geocode",
                   new=AsyncMock(return_value=(18.5204, 73.8567))):
            resp = client.get("/api/geocode?place=Pune")
        assert resp.status_code == 200
        data = resp.json()
        assert "lat" in data and "lon" in data

    def test_geocode_not_found_returns_404(self, client):
        with patch("app.api.routes.geocode",
                   side_effect=ValueError("Location not found: 'xyzzy123'")):
            resp = client.get("/api/geocode?place=xyzzy123")
        assert resp.status_code == 404
