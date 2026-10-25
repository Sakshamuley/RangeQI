"""FastAPI route handlers — Phase 6 update."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from app.database import get_db
from app.models import BODY_TYPES, EVModel, PredictionScan
from app.schemas import (
    EVModelOut,
    PredictRequest,
    PredictResponse,
    RecommendRequest,
    RecommendResponse,
    EVPredictionResult,
    RangeQuantiles,
    EnergyTermBreakdown,
    CarbonComparison,
)
from app.services.route_service import get_full_route
from app.services.recommend_service import recommend as _recommend
from app.services.geocoding_service import geocode

logger = logging.getLogger(__name__)
router = APIRouter()

BACKEND_ROOT    = Path(__file__).resolve().parent.parent.parent
BENCHMARK_JSON  = BACKEND_ROOT / "data" / "benchmark_results.json"
VALIDATION_JSON = BACKEND_ROOT / "data" / "validation_results.json"


# ── EV model catalogue ────────────────────────────────────────────────────────

@router.get("/ev-models", response_model=list[EVModelOut])
def list_ev_models(
    body_type: Optional[str] = Query(None),
    price_min: Optional[float] = Query(None, ge=0),
    price_max: Optional[float] = Query(None, ge=0),
    seats_min: Optional[int] = Query(None, ge=1),
    db: Session = Depends(get_db),
):
    """
    List EV models with optional filters.
    NULL-price EVs are excluded from price-range filtering with a warning.
    """
    q = db.query(EVModel)

    if body_type:
        if body_type not in BODY_TYPES:
            raise HTTPException(
                400,
                f"Invalid body_type '{body_type}'. "
                f"Must be one of: {', '.join(BODY_TYPES)}",
            )
        q = q.filter(EVModel.body_type == body_type)

    if seats_min is not None:
        q = q.filter(EVModel.seats >= seats_min)

    evs = q.order_by(EVModel.model_name).all()

    null_price_count = 0
    if price_min is not None or price_max is not None:
        filtered: list[EVModel] = []
        for ev in evs:
            if ev.ex_showroom_price_lakh is None:
                null_price_count += 1
                continue
            if price_min is not None and ev.ex_showroom_price_lakh < price_min:
                continue
            if price_max is not None and ev.ex_showroom_price_lakh > price_max:
                continue
            filtered.append(ev)
        evs = filtered
        if null_price_count:
            logger.warning(
                "Price filter: %d EV(s) excluded (ex_showroom_price_lakh is NULL)",
                null_price_count,
            )

    return evs


@router.get("/body-types")
def list_body_types():
    """Return all supported EV body-type values."""
    return {"body_types": list(BODY_TYPES)}


# ── Route + elevation + weather ───────────────────────────────────────────────

@router.get("/route-elevation")
async def route_elevation(
    start_lat: float,
    start_lon: float,
    end_lat: float,
    end_lon: float,
    season: str = "summer",
    db: Session = Depends(get_db),
):
    """
    Fetch road geometry (OSRM), elevation (Open-Elevation batched POST)
    and weather (OpenWeatherMap) for a source→destination pair.

    Results are cached in the api_cache table by rounded coordinates.
    Every fallback is returned as a flag + human-readable message.
    """
    result = await get_full_route(
        start_lat, start_lon, end_lat, end_lon,
        season=season,
        db=db,
    )
    result = dict(result)
    result["gradient_segments"] = [
        {"gradient": g, "length_m": l}
        for g, l in result.get("gradient_segments", [])
    ]
    return result


@router.get("/geocode")
async def geocode_place(place: str, db: Session = Depends(get_db)):
    """
    Resolve a place name to lat/lon via Nominatim (OpenStreetMap).
    Results are cached for 7 days.
    """
    try:
        lat, lon = await geocode(place, db=db)
    except ValueError as exc:
        raise HTTPException(404, str(exc))
    except Exception as exc:
        raise HTTPException(502, f"Geocoding failed: {exc}")
    return {"place": place, "lat": lat, "lon": lon}


# ── Benchmark / validation data ───────────────────────────────────────────────

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


# ── History ───────────────────────────────────────────────────────────────────

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


# ── Recommend ─────────────────────────────────────────────────────────────────

@router.post("/recommend", response_model=RecommendResponse)
async def recommend(req: RecommendRequest, db: Session = Depends(get_db)):
    """
    Ranked EV recommendation for a given route.

    Inputs: route coordinates, optional price range, optional body type,
    and advanced driving-profile parameters.

    Returns a ranked list of EVs with:
      - p10/p50/p90 range bands (conservative / expected / optimistic)
      - verdict: Suitable / Marginal / Not suitable
      - range margin (km and %)
      - trip energy (kWh), estimated charging cost (₹)
      - per-term energy breakdown from the physics model
      - CO₂ comparison vs petrol (illustrative)

    All prices and CO₂ figures are labelled illustrative.
    Fallback warnings are returned when any upstream API is unavailable.
    """
    return await _recommend(req, db)


# ── Predict (single-model mode) ───────────────────────────────────────────────

@router.post("/predict")
async def predict(req: PredictRequest, db: Session = Depends(get_db)):
    """
    Check specific EV model(s) against a route — uses the same pipeline as /recommend.

    Provide ev_model_ids to select which EVs to evaluate.
    """
    from app.services.recommend_service import (
        _segments_from_route, _evaluate_ev, _get_predictor,
        _CALIBRATED_CONFIG, ELECTRICITY_PRICE_INR_PER_KWH,
    )
    from app.simulator import range_km as phys_range_km

    # Resolve route
    if req.route:
        route_info = await get_full_route(
            req.route.start_lat, req.route.start_lon,
            req.route.end_lat, req.route.end_lon,
            db=db,
        )
    elif req.daily_distance_km:
        # Simple flat-road approximation when only distance is given
        from app.simulator import SegmentData
        route_info = {
            "distance_km": req.daily_distance_km,
            "duration_s": req.daily_distance_km / 60.0 * 3600,
            "avg_speed_kmh": 60.0,
            "elevation_gain_m": 0.0,
            "elevation_loss_m": 0.0,
            "start_elevation_m": 0.0,
            "temperature_c": 28.0,
            "gradient_segments": [(0.0, req.daily_distance_km * 50.0)],
            "profile": [],
            "fallback_warnings": ["No route provided — using flat-road approximation at 60 km/h, 28 °C"],
            "route_fallback": True,
            "elevation_fallback": True,
            "weather_fallback": True,
        }
    else:
        raise HTTPException(400, "Provide route or daily_distance_km")

    api_warnings: list[str] = list(route_info.get("fallback_warnings", []))
    distance_km = route_info["distance_km"]
    trip_distance_km = distance_km * (2 if req.round_trip else 1)
    required_km = round(trip_distance_km * (1.0 + req.reserve_pct / 100.0), 1)
    segments = _segments_from_route(route_info["gradient_segments"])

    evs = db.query(EVModel).filter(EVModel.id.in_(req.ev_model_ids)).all()
    if not evs:
        raise HTTPException(404, "No EV models found for the given IDs")

    predictions = []
    ml_warns: list[str] = []
    for ev_row in evs:
        # Build a minimal RecommendRequest-like object for _evaluate_ev
        class _FakeReq:
            driving_style = req.driving_style
            num_passengers = req.num_passengers
            ac_on = req.ac_on
            round_trip = req.round_trip

        try:
            r = _evaluate_ev(
                ev_row, segments, route_info, _FakeReq(),
                required_km=required_km,
                trip_distance_km=trip_distance_km,
                warnings_out=ml_warns,
            )
            from app.schemas import EVPredictionResult
            predictions.append(EVPredictionResult(
                ev_model_id=r.ev_model_id,
                model_name=r.model_name,
                claimed_range_km=r.claimed_range_km,
                predicted_range=r.predicted_range,
                verdict=r.verdict,
                range_margin_km=r.range_margin_km,
                trip_energy_kwh=r.trip_energy_kwh,
                energy_breakdown=r.energy_breakdown,
                carbon=r.carbon,
                warnings=r.warnings,
            ))
        except Exception as exc:
            api_warnings.append(f"Could not evaluate {ev_row.model_name}: {exc}")

    for w in ml_warns:
        if w not in api_warnings:
            api_warnings.append(w)

    scan = PredictionScan(
        input_json=json.dumps(req.model_dump()),
        output_json=json.dumps({"n_evs": len(predictions)}),
    )
    db.add(scan)
    db.commit()
    db.refresh(scan)

    from app.schemas import PredictResponse
    return PredictResponse(
        scan_id=scan.id,
        trip_distance_km=round(trip_distance_km, 2),
        temperature_c=round(route_info["temperature_c"], 1),
        elevation_gain_m=round(route_info["elevation_gain_m"], 1),
        elevation_loss_m=round(route_info["elevation_loss_m"], 1),
        required_km=required_km,
        predictions=predictions,
        route_profile=route_info.get("profile", [])[:200],
        api_warnings=api_warnings,
    )
