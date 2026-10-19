"""Seed ev_models table from CSV on startup."""

import csv
import logging
from pathlib import Path

from sqlalchemy.orm import Session

from app.models import EVModel

logger = logging.getLogger(__name__)

SEED_CSV = Path(__file__).resolve().parent.parent / "data" / "ev_models_seed.csv"


def seed_ev_models(db: Session) -> int:
    """
    Load EV model seed data into SQLite if the table is empty.
    Returns number of rows inserted.
    """
    if db.query(EVModel).count() > 0:
        logger.info("ev_models already seeded (%d rows)", db.query(EVModel).count())
        return 0

    if not SEED_CSV.exists():
        logger.error("Seed CSV not found: %s", SEED_CSV)
        return 0

    inserted = 0
    with open(SEED_CSV, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            db.add(
                EVModel(
                    model_name=row["model_name"],
                    battery_capacity_kwh=float(row["battery_capacity_kwh"]),
                    claimed_range_km=float(row["claimed_range_km"]),
                    kerb_weight_kg=float(row["kerb_weight_kg"]),
                    drag_coefficient=float(row["drag_coefficient"]),
                    frontal_area_m2=float(row["frontal_area_m2"]),
                    motor_power_kw=float(row["motor_power_kw"]),
                )
            )
            inserted += 1

    db.commit()
    logger.info("Seeded %d EV models from %s", inserted, SEED_CSV)
    return inserted
