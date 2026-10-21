"""
Seed ev_models table from ev_models_seed.csv.

Validation rules (loader refuses rows that violate any of these):
  - Required string fields must be non-empty.
  - body_type must be one of: hatchback, sedan, compact_suv, suv, mpv.
  - claimed_cycle must be one of: ARAI, MIDC, WLTP.
  - claimed_range_km in (0, 2000).
  - battery_capacity_kwh in (1, 300).
  - kerb_weight_kg in (500, 6000).
  - drag_coefficient in (0.10, 0.70).
  - frontal_area_m2 in (1.00, 5.00).
  - motor_power_kw in (1, 1000).
  - seats in (2, 9).
  - No duplicate model_name rows.

Prices: ex_showroom_price_lakh may be NULL (empty in CSV).
  NULL-price rows are included in the DB but excluded from price-range
  filter queries with a visible warning to the caller.

Usable battery: if usable_battery_kwh is empty/NULL in the CSV, the loader
  sets usable_battery_kwh = 0.95 × battery_capacity_kwh and
  usable_battery_assumed = True.
"""

from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Any

from sqlalchemy.orm import Session

from app.models import BODY_TYPES, CLAIMED_CYCLES, EVModel

logger = logging.getLogger(__name__)

SEED_CSV = Path(__file__).resolve().parent.parent / "data" / "ev_models_seed.csv"

# ── validation helpers ────────────────────────────────────────────────────────

_RANGE_RULES: dict[str, tuple[float, float]] = {
    "claimed_range_km": (0.0, 2000.0),
    "battery_capacity_kwh": (1.0, 300.0),
    "kerb_weight_kg": (500.0, 6000.0),
    "drag_coefficient": (0.10, 0.70),
    "frontal_area_m2": (1.00, 5.00),
    "motor_power_kw": (1.0, 1000.0),
}

_MANDATORY_PHYSICS = (
    "battery_capacity_kwh",
    "kerb_weight_kg",
    "drag_coefficient",
    "frontal_area_m2",
    "motor_power_kw",
)


def _parse_float(value: str, field: str) -> float:
    """Convert CSV string to float; raise ValueError on failure."""
    try:
        return float(value.strip())
    except (ValueError, AttributeError):
        raise ValueError(f"Field '{field}' must be a number, got: {value!r}")


def _parse_float_optional(value: str) -> float | None:
    """Return float or None if the value is empty/blank."""
    stripped = value.strip() if value else ""
    if not stripped:
        return None
    try:
        return float(stripped)
    except ValueError:
        return None


def _parse_bool(value: str, field: str) -> bool:
    stripped = value.strip().lower() if value else ""
    if stripped in ("true", "1", "yes"):
        return True
    if stripped in ("false", "0", "no"):
        return False
    raise ValueError(f"Field '{field}' must be true/false, got: {value!r}")


def validate_row(row: dict[str, str], line_number: int) -> dict[str, Any] | None:
    """
    Validate one CSV row.  Returns a cleaned dict on success or None if the
    row should be rejected (error is logged).
    """
    errors: list[str] = []

    model_name = row.get("model_name", "").strip()
    if not model_name:
        errors.append("model_name is required")

    body_type = row.get("body_type", "").strip()
    if body_type not in BODY_TYPES:
        errors.append(f"body_type '{body_type}' must be one of {BODY_TYPES}")

    claimed_cycle = row.get("claimed_cycle", "").strip()
    if claimed_cycle not in CLAIMED_CYCLES:
        errors.append(f"claimed_cycle '{claimed_cycle}' must be one of {CLAIMED_CYCLES}")

    # seats
    try:
        seats = int(row.get("seats", "").strip())
        if not (2 <= seats <= 9):
            errors.append(f"seats {seats} out of range [2, 9]")
    except (ValueError, AttributeError):
        errors.append("seats must be an integer")
        seats = 0

    # Mandatory physics: must parse and be in range
    floats: dict[str, float] = {}
    for field in _MANDATORY_PHYSICS:
        raw = row.get(field, "").strip()
        if not raw:
            errors.append(f"Mandatory physics field '{field}' is missing")
            continue
        try:
            val = _parse_float(raw, field)
        except ValueError as exc:
            errors.append(str(exc))
            continue
        lo, hi = _RANGE_RULES.get(field, (float("-inf"), float("inf")))
        if not (lo < val <= hi):
            errors.append(f"'{field}' = {val} out of range ({lo}, {hi}]")
        else:
            floats[field] = val

    # claimed_range_km (also mandatory, in range rules)
    claimed_range_raw = row.get("claimed_range_km", "").strip()
    if not claimed_range_raw:
        errors.append("claimed_range_km is required")
    else:
        try:
            cr = float(claimed_range_raw)
            lo, hi = _RANGE_RULES["claimed_range_km"]
            if not (lo < cr <= hi):
                errors.append(f"claimed_range_km {cr} out of range ({lo}, {hi}]")
            else:
                floats["claimed_range_km"] = cr
        except ValueError:
            errors.append(f"claimed_range_km must be a number, got {claimed_range_raw!r}")

    if errors:
        logger.error("Row %d (%s): rejected — %s", line_number, model_name or "?", "; ".join(errors))
        return None

    # Optional numeric fields
    price = _parse_float_optional(row.get("ex_showroom_price_lakh", ""))
    usable_raw = _parse_float_optional(row.get("usable_battery_kwh", ""))
    usable_assumed: bool
    if usable_raw is None:
        usable_kwh = round(floats["battery_capacity_kwh"] * 0.95, 2)
        usable_assumed = True
    else:
        usable_kwh = usable_raw
        usable_assumed = _parse_bool(row.get("usable_battery_assumed", "true"), "usable_battery_assumed")

    try:
        price_assumed = _parse_bool(row.get("price_assumed", "true"), "price_assumed")
    except ValueError:
        price_assumed = True

    try:
        cd_assumed = _parse_bool(row.get("drag_coefficient_assumed", "true"), "drag_coefficient_assumed")
        fa_assumed = _parse_bool(row.get("frontal_area_assumed", "true"), "frontal_area_assumed")
        dt_assumed = _parse_bool(row.get("drivetrain_eff_assumed", "true"), "drivetrain_eff_assumed")
        regen_assumed = _parse_bool(row.get("regen_eff_assumed", "true"), "regen_eff_assumed")
    except ValueError as exc:
        logger.warning("Row %d: flag parse error: %s", line_number, exc)
        cd_assumed = fa_assumed = dt_assumed = regen_assumed = True

    return {
        "model_name": model_name,
        "variant": row.get("variant", "").strip() or None,
        "body_type": body_type,
        "seats": seats,
        "ex_showroom_price_lakh": price,
        "price_date": row.get("price_date", "").strip() or None,
        "price_source_url": row.get("price_source_url", "").strip() or None,
        "price_assumed": price_assumed,
        "claimed_range_km": floats["claimed_range_km"],
        "claimed_cycle": claimed_cycle,
        "spec_source_url": row.get("spec_source_url", "").strip() or None,
        "battery_capacity_kwh": floats["battery_capacity_kwh"],
        "usable_battery_kwh": usable_kwh,
        "usable_battery_assumed": usable_assumed,
        "kerb_weight_kg": floats["kerb_weight_kg"],
        "drag_coefficient": floats["drag_coefficient"],
        "drag_coefficient_assumed": cd_assumed,
        "frontal_area_m2": floats["frontal_area_m2"],
        "frontal_area_assumed": fa_assumed,
        "motor_power_kw": floats["motor_power_kw"],
        "drivetrain_eff_assumed": dt_assumed,
        "regen_eff_assumed": regen_assumed,
    }


def seed_ev_models(db: Session) -> int:
    """
    Load EV model seed data into SQLite if the table is empty.

    Validates every row before inserting; rejects rows with missing mandatory
    physics specs or out-of-range values.  Returns number of rows inserted.
    """
    existing = db.query(EVModel).count()
    if existing > 0:
        logger.info("ev_models already seeded (%d rows) — skipping", existing)
        return 0

    if not SEED_CSV.exists():
        logger.error("Seed CSV not found: %s", SEED_CSV)
        return 0

    seen_names: set[str] = set()
    inserted = 0
    rejected = 0

    with open(SEED_CSV, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for line_no, raw_row in enumerate(reader, start=2):
            cleaned = validate_row(raw_row, line_no)
            if cleaned is None:
                rejected += 1
                continue

            name = cleaned["model_name"]
            if name in seen_names:
                logger.error("Row %d: duplicate model_name '%s' — skipped", line_no, name)
                rejected += 1
                continue
            seen_names.add(name)

            db.add(EVModel(**cleaned))
            inserted += 1

    db.commit()
    logger.info(
        "Seeded %d EV models from %s (%d row(s) rejected)",
        inserted, SEED_CSV, rejected,
    )
    if rejected:
        logger.warning(
            "%d seed row(s) were rejected due to validation errors — "
            "check logs above for details.",
            rejected,
        )
    return inserted
