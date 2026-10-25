"""SQLAlchemy ORM models for RangeIQ."""

from datetime import datetime

from sqlalchemy import Boolean, Column, DateTime, Float, Integer, String, Text
from sqlalchemy.orm import declarative_base

Base = declarative_base()

# Allowed values (enforced in seed loader, not as DB constraints)
BODY_TYPES = ("hatchback", "sedan", "compact_suv", "suv", "mpv")
CLAIMED_CYCLES = ("ARAI", "MIDC", "WLTP")


class EVModel(Base):
    """
    Real-world EV specifications sold in India.

    Physics specs (kerb_weight_kg, drag_coefficient, frontal_area_m2, motor_power_kw)
    are mandatory for the route energy model and may never be NULL.  All other
    assumed flags default to True; set to False only when the value comes from an
    official published source that can be independently verified.
    """

    __tablename__ = "ev_models"

    id = Column(Integer, primary_key=True, autoincrement=True)

    # Identity
    model_name = Column(String(120), unique=True, nullable=False)
    variant = Column(String(80))                           # e.g. "Long Range", "AWD"
    body_type = Column(String(20), nullable=False)         # hatchback|sedan|compact_suv|suv|mpv
    seats = Column(Integer, nullable=False)

    # ── Pricing ──────────────────────────────────────────────────────────────
    # NULL when the ex-showroom price has not been sourced to a verifiable URL.
    # NULL-price EVs are excluded from price-range filtering with a visible warning.
    ex_showroom_price_lakh = Column(Float)
    price_date = Column(String(20))                        # YYYY-MM format
    price_source_url = Column(String(512))
    price_assumed = Column(Boolean, nullable=False, default=True)

    # ── Range claim ──────────────────────────────────────────────────────────
    claimed_range_km = Column(Float, nullable=False)
    claimed_cycle = Column(String(10), nullable=False)     # ARAI|MIDC|WLTP
    spec_source_url = Column(String(512))

    # ── Battery ───────────────────────────────────────────────────────────────
    battery_capacity_kwh = Column(Float, nullable=False)   # nominal (gross) kWh
    # usable_battery_kwh: NULL means loader applied 0.95 × nominal (assumed).
    usable_battery_kwh = Column(Float)
    usable_battery_assumed = Column(Boolean, nullable=False, default=True)

    # ── Mandatory physics specs ───────────────────────────────────────────────
    kerb_weight_kg = Column(Float, nullable=False)

    drag_coefficient = Column(Float, nullable=False)
    drag_coefficient_assumed = Column(Boolean, nullable=False, default=True)

    frontal_area_m2 = Column(Float, nullable=False)
    frontal_area_assumed = Column(Boolean, nullable=False, default=True)

    motor_power_kw = Column(Float, nullable=False)

    # Drivetrain and regen efficiencies are parameters in the physics config;
    # these flags indicate that the default config values are assumed for this EV.
    drivetrain_eff_assumed = Column(Boolean, nullable=False, default=True)
    regen_eff_assumed = Column(Boolean, nullable=False, default=True)


class ApiCache(Base):
    """
    Key-value cache for external API responses (OSRM, Open-Elevation, weather).
    Keyed by a deterministic hash of the request parameters.
    """

    __tablename__ = "api_cache"

    id = Column(Integer, primary_key=True, autoincrement=True)
    key = Column(String(256), unique=True, nullable=False, index=True)
    json_data = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    ttl_seconds = Column(Integer, nullable=False, default=3600)


class PredictionScan(Base):
    """Persisted prediction for history / GET /api/scan/{id}."""

    __tablename__ = "prediction_scans"

    id = Column(Integer, primary_key=True, autoincrement=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    ev_model_ids = Column(Text)   # JSON array of int; NULL for /recommend (no pre-filter by ID)
    input_json = Column(Text, nullable=False)
    output_json = Column(Text, nullable=False)
