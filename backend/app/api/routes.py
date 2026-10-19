"""FastAPI route handlers."""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.database import get_db
from app.models import EVModel, PredictionScan
from app.schemas import EVModelOut, PredictRequest, PredictResponse
from app.ml.predict import predictor
from app.services.route_service import fetch_elevation_profile, fetch_temperature, SEASON_TEMPS
from app.services.suitability import compute_suitability
from app.services.carbon import compute_carbon_comparison
from app.simulator import EVSpecs, TripConditions, simulate_trip_energy

logger = logging.getLogger(__name__)
router = APIRouter()

BACKEND_ROOT = Path(__file__).resolve().parent.parent.parent
BENCHMARK_JSON = BACKEND_ROOT / "data" / "benchmark_results.json"
VALIDATION_JSON = BACKEND_ROOT / "data" / "validation_results.json"


@router.get("/ev-models", response_model=list[EVModelOut])
def list_ev_models(db: Session = Depends(get_db)):
    return db.query(EVModel).order_by(EVModel.model_name).all()


@router.get("/benchmark")
def get_benchmark():
    if not BENCHMARK_JSON.exists():
        raise HTTPException(503, "Benchmark not available — run training first")
    with open(BENCHMARK_JSON, encoding="utf-8") as f:
        return json.load(f)


@router.get("/validation")
def get_validation():
    if not VALIDATION_JSON.exists():
        raise HTTPException(503, "Validation not run yet")
    with open(VALIDATION_JSON, encoding="utf-8") as f:
        return json.load(f)


@router.get("/route-elevation")
async def route_elevation(
    start_lat: float,
    start_lon: float,
    end_lat: float,
    end_lon: float,
):
    return await fetch_elevation_profile(start_lat, start_lon, end_lat, end_lon)


@router.get("/history")
def list_history(limit: int = 20, db: Session = Depends(get_db)):
    scans = (
        db.query(PredictionScan)
        .order_by(PredictionScan.created_at.desc())
        .limit(min(limit, 50))
        .all()
    )
    return [
        {
            "id": s.id,
            "created_at": s.created_at.isoformat(),
            "input": json.loads(s.input_json),
            "output": json.loads(s.output_json),
        }
        for s in scans
    ]


@router.get("/scan/{scan_id}")
def get_scan(scan_id: int, db: Session = Depends(get_db)):
    scan = db.query(PredictionScan).filter(PredictionScan.id == scan_id).first()
    if not scan:
        raise HTTPException(404, "Scan not found")
    return {
        "id": scan.id,
        "created_at": scan.created_at.isoformat(),
        "input": json.loads(scan.input_json),
        "output": json.loads(scan.output_json),
    }


@router.post("/predict")
async def predict(req: PredictRequest, db: Session = Depends(get_db)):
    if not predictor.is_ready:
        raise HTTPException(503, "ML models not loaded — run training pipeline")

    t0 = time.perf_counter()
    api_warnings: list[str] = []
    route_profile: list[dict] = []

    # Resolve trip distance and elevation
    if req.route:
        elev = await fetch_elevation_profile(
            req.route.start_lat, req.route.start_lon,
            req.route.end_lat, req.route.end_lon,
        )
        trip_distance = elev["distance_km"]
        elev_gain = elev["elevation_gain_m"]
        elev_loss = elev["elevation_loss_m"]
        route_profile = elev.get("profile", [])
        if elev.get("warning"):
            api_warnings.append(elev["warning"])
        mid_lat = (req.route.start_lat + req.route.end_lat) / 2
        mid_lon = (req.route.start_lon + req.route.end_lon) / 2
        weather = await fetch_temperature(mid_lat, mid_lon, req.season)
    else:
        trip_distance = req.daily_distance_km or 50.0
        elev_gain = 80.0
        elev_loss = 80.0
        weather = {
            "temperature_c": SEASON_TEMPS.get(req.season, 30.0),
            "api_used": False,
            "warning": "Manual distance mode — using estimated flat terrain.",
        }
        api_warnings.append(weather["warning"])

    temperature = weather["temperature_c"]
    if weather.get("warning") and weather["warning"] not in api_warnings:
        api_warnings.append(weather["warning"])

    daily_requirement = trip_distance  # daily commute = trip distance

    ev_models = (
        db.query(EVModel).filter(EVModel.id.in_(req.ev_model_ids)).all()
    )
    if not ev_models:
        raise HTTPException(404, "No EV models found for given IDs")

    predictions = []
    for ev in ev_models:
        features = {
            "battery_capacity_kwh": ev.battery_capacity_kwh,
            "kerb_weight_kg": ev.kerb_weight_kg,
            "drag_coefficient": ev.drag_coefficient,
            "frontal_area_m2": ev.frontal_area_m2,
            "motor_power_kw": ev.motor_power_kw,
            "claimed_range_km": ev.claimed_range_km,
            "distance_km": trip_distance,
            "avg_speed_kmh": req.avg_speed_kmh,
            "elevation_gain_m": elev_gain,
            "elevation_loss_m": elev_loss,
            "temperature_c": temperature,
            "driving_style": req.driving_style,
            "num_passengers": req.num_passengers,
            "season": req.season,
        }

        quantiles = predictor.predict_quantiles(features)
        shap_bd = predictor.shap_breakdown(features, ev.claimed_range_km)

        suitability = compute_suitability(
            quantiles["p50"], quantiles["p10"], daily_requirement, ev.claimed_range_km
        )

        # Trip energy from physics simulator for carbon calc
        ev_specs = EVSpecs(
            ev.battery_capacity_kwh, ev.kerb_weight_kg, ev.drag_coefficient,
            ev.frontal_area_m2, ev.motor_power_kw, ev.claimed_range_km,
        )
        trip = TripConditions(
            distance_km=trip_distance,
            avg_speed_kmh=req.avg_speed_kmh,
            elevation_gain_m=elev_gain,
            elevation_loss_m=elev_loss,
            temperature_c=temperature,
            driving_style=req.driving_style,
            num_passengers=req.num_passengers,
        )
        energy_bd = simulate_trip_energy(ev_specs, trip, add_noise=False)
        carbon = compute_carbon_comparison(energy_bd.total_energy_kwh, trip_distance)

        predictions.append({
            "ev_model_id": ev.id,
            "model_name": ev.model_name,
            "claimed_range_km": ev.claimed_range_km,
            "predicted_range": quantiles,
            "energy_breakdown": shap_bd,
            "suitability": suitability,
            "carbon": carbon,
            "warnings": [],
        })

    total_ms = round((time.perf_counter() - t0) * 1000, 2)

    response_body = {
        "trip_distance_km": trip_distance,
        "temperature_c": temperature,
        "elevation_gain_m": elev_gain,
        "elevation_loss_m": elev_loss,
        "predictions": predictions,
        "route_profile": route_profile,
        "api_warnings": api_warnings,
        "execution_time_ms": total_ms,
    }

    # Persist scan for history
    scan = PredictionScan(
        ev_model_ids=json.dumps(req.ev_model_ids),
        input_json=req.model_dump_json(),
        output_json=json.dumps(response_body),
    )
    db.add(scan)
    db.commit()
    db.refresh(scan)

    response_body["scan_id"] = scan.id
    return response_body
