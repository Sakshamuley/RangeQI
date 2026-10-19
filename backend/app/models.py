"""SQLAlchemy ORM models for RangeIQ."""

from datetime import datetime

from sqlalchemy import Column, DateTime, Float, Integer, String, Text
from sqlalchemy.orm import declarative_base

Base = declarative_base()


class EVModel(Base):
    """Real-world EV specifications sold in India."""

    __tablename__ = "ev_models"

    id = Column(Integer, primary_key=True, autoincrement=True)
    model_name = Column(String(120), unique=True, nullable=False)
    battery_capacity_kwh = Column(Float, nullable=False)
    claimed_range_km = Column(Float, nullable=False)
    kerb_weight_kg = Column(Float, nullable=False)
    drag_coefficient = Column(Float, nullable=False)
    frontal_area_m2 = Column(Float, nullable=False)
    motor_power_kw = Column(Float, nullable=False)


class PredictionScan(Base):
    """Persisted prediction for history / GET /api/scan/{id}."""

    __tablename__ = "prediction_scans"

    id = Column(Integer, primary_key=True, autoincrement=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    ev_model_ids = Column(Text, nullable=False)  # JSON array
    input_json = Column(Text, nullable=False)
    output_json = Column(Text, nullable=False)
