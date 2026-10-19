"""RangeIQ FastAPI application entry point."""

from __future__ import annotations

import logging
import sys
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Ensure backend root is on path when running as module or script
BACKEND_ROOT = Path(__file__).resolve().parent.parent
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.database import SessionLocal, init_db
from app.seed import seed_ev_models
from app.api.routes import router
from app.validation import run_validation
from app.ml.predict import predictor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("rangeiq")


def _ensure_training_pipeline() -> None:
    """Generate training data and train models if not present."""
    from app.ml.train import TRAINING_CSV, MODELS_DIR, main as train_main
    from app.ml.data_gen import save_training_csv

    if not TRAINING_CSV.exists():
        logger.info("Generating synthetic training data...")
        save_training_csv()

    quantile_path = MODELS_DIR / "quantile_models.joblib"
    if not quantile_path.exists():
        logger.info("Training ML models (this may take a minute)...")
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
    predictor._load_models()

    try:
        run_validation(predictor if predictor.is_ready else None)
    except Exception as exc:
        logger.warning("Validation run skipped: %s", exc)

    logger.info("RangeIQ backend ready")
    yield


app = FastAPI(
    title="RangeIQ API",
    description="AI-Powered EV Range & Suitability Simulator",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api")


@app.get("/")
def root():
    return {"service": "RangeIQ", "docs": "/docs", "status": "ok"}
