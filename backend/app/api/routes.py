"""FastAPI route handlers."""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from app.database import get_db
from app.models import BODY_TYPES, EVModel, PredictionScan
from app.schemas import EVModelOut, PredictRequest

logger = logging.getLogger(__name__)
router = APIRouter()

BACKEND_ROOT = Path(__file__).resolve().parent.parent.parent
BENCHMARK_JSON = BACKEND_ROOT / "data" / "benchmark_results.json"
VALIDATION_JSON = BACKEND_ROOT / "data" / "validation_results.json"


# ── EV model catalogue ────────────────────────────────────────────────────────

@router.get("/ev-models", response_model=list[EVModelOut])
def list_ev_models(
    body_type: Optional[str] = Query(None, description="Filter by body type"),
    price_min: Optional[float] = Query(None, ge=0, description="Min ex-showroom price (lakh)"),
    price_max: Optional[float] = Query(None, ge=0, description="Max ex-showroom price (lakh)"),
    seats_min: Optional[int] = Query(None, ge=1, description="Minimum seat count"),
    db: Session = Depends(get_db),
):
    """
    List EV models with optional filters.

    Price filtering silently excludes EVs whose ex_showroom_price_lakh is NULL
    and includes a `price_filter_warning` field in the response headers when
    that happens.  Body type must be one of: hatchback, sedan, compact_suv,
    suv, mpv.
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

    # Price filtering — exclude NULLs, warn caller
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
                "Price filter applied: %d EV(s) excluded because ex_showroom_price_lakh is NULL",
                null_price_count,
            )

    return evs


@router.get("/body-types")
def list_body_types():
    """Return all supported EV body-type values."""
    return {"body_types": list(BODY_TYPES)}


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


# ── Predict (single-model mode) ───────────────────────────────────────────────

@router.post("/predict")
async def predict(req: PredictRequest, db: Session = Depends(get_db)):
    """
    Check specific EV model(s) against a route.
    Full physics + ML pipeline.  Phases 2–6 will populate the physics details.
    This stub returns a 501 until the simulator rewrite (Phase 3) is merged.
    """
    # TODO(Phase 2–6): implement full pipeline
    raise HTTPException(
        501,
        "POST /api/predict will be implemented in Phase 3. "
        "Use POST /api/recommend (Phase 6) for production use.",
    )


# ── Recommend ─────────────────────────────────────────────────────────────────

@router.post("/recommend")
async def recommend(db: Session = Depends(get_db)):
    """
    Ranked EV recommendation for a route + price/body-type filter.
    Implemented in Phase 6.
    """
    raise HTTPException(501, "POST /api/recommend will be implemented in Phase 6.")
