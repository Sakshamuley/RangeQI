"""FastAPI route handlers — Phase 2 update."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from app.database import get_db
from app.models import BODY_TYPES, EVModel, PredictionScan
from app.schemas import EVModelOut, PredictRequest
from app.services.route_service import get_full_route

logger = logging.getLogger(__name__)
router = APIRouter()

BACKEND_ROOT = Path(__file__).resolve().parent.parent.parent
BENCHMARK_JSON = BACKEND_ROOT / "data" / "benchmark_results.json"
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
    # gradient_segments are tuples — make JSON-serialisable
    result = dict(result)
    result["gradient_segments"] = [
        {"gradient": g, "length_m": l}
        for g, l in result.get("gradient_segments", [])
    ]
    return result


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


# ── Predict / Recommend (stubs — implemented in Phases 3–6) ──────────────────

@router.post("/predict")
async def predict(req: PredictRequest, db: Session = Depends(get_db)):
    """Single-model prediction.  Implemented fully in Phase 3."""
    raise HTTPException(
        501,
        "POST /api/predict will be fully implemented after Phase 3 (physics rewrite). "
        "Use POST /api/recommend after Phase 6 for production use.",
    )


@router.post("/recommend")
async def recommend(db: Session = Depends(get_db)):
    """Ranked EV recommendation.  Implemented in Phase 6."""
    raise HTTPException(501, "POST /api/recommend will be implemented in Phase 6.")
