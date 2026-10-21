"""RangeIQ FastAPI application entry point."""

from __future__ import annotations

import logging
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path

# Load .env before any other import so env vars are available immediately.
# The file is optional — the app works without it.
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent.parent / ".env", verbose=False)
except ImportError:
    pass  # python-dotenv not installed in CI minimal env

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Ensure backend root is on sys.path when running as module or script
BACKEND_ROOT = Path(__file__).resolve().parent.parent
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.database import SessionLocal, init_db
from app.seed import seed_ev_models
from app.api.routes import router

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("rangeiq")


def _ensure_training_pipeline() -> None:
    """Generate training data and train models if not already present."""
    try:
        from app.ml.train import TRAINING_CSV, MODELS_DIR, main as train_main
        from app.ml.data_gen import save_training_csv
    except Exception as exc:
        logger.warning("ML pipeline import failed: %s", exc)
        return

    if not TRAINING_CSV.exists():
        logger.info("Generating synthetic training data…")
        save_training_csv()

    quantile_path = MODELS_DIR / "quantile_models.joblib"
    if not quantile_path.exists():
        logger.info("Training ML models (this may take a minute)…")
        train_main()


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    db = SessionLocal()
    try:
        seed_ev_models(db)
    finally:
        db.close()

    _ensure_training_pipeline()

    # Load ML models if available
    try:
        from app.ml.predict import predictor
        predictor._load_models()
    except Exception as exc:
        logger.warning("ML model load skipped: %s", exc)

    # Run validation if available
    try:
        from app.validation import run_validation
        from app.ml.predict import predictor as pred
        run_validation(pred if pred.is_ready else None)
    except Exception as exc:
        logger.warning("Validation run skipped: %s", exc)

    logger.info("RangeIQ backend ready")
    yield


# CORS: restrict to the frontend origin.  Use FRONTEND_ORIGIN env var to
# override (e.g. in production).  Wildcard + credentials is forbidden by
# the CORS spec and was removed here.
_frontend_origin = os.getenv("FRONTEND_ORIGIN", "http://localhost:5173")
_allowed_origins = [o.strip() for o in _frontend_origin.split(",") if o.strip()]

app = FastAPI(
    title="RangeIQ API",
    description="AI-Powered EV Range & Suitability Simulator",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins,
    allow_credentials=False,   # never combine wildcard-like origins with credentials
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "Accept"],
)

app.include_router(router, prefix="/api")


@app.get("/")
def root():
    return {"service": "RangeIQ", "version": "2.0.0", "docs": "/docs", "status": "ok"}
