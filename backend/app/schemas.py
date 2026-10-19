"""Pydantic request/response schemas."""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field, model_validator

DrivingStyle = Literal["aggressive", "normal", "calm"]
Season = Literal["summer", "monsoon", "winter", "spring"]


class RouteCoords(BaseModel):
    start_lat: float
    start_lon: float
    end_lat: float
    end_lon: float


class PredictRequest(BaseModel):
    ev_model_ids: list[int] = Field(..., min_length=1)
    daily_distance_km: Optional[float] = Field(None, gt=0)
    route: Optional[RouteCoords] = None
    driving_style: DrivingStyle = "normal"
    season: Season = "summer"
    num_passengers: int = Field(1, ge=1, le=7)
    avg_speed_kmh: float = Field(50.0, gt=0, le=130)

    @model_validator(mode="after")
    def require_distance_or_route(self):
        if self.daily_distance_km is None and self.route is None:
            raise ValueError("Provide daily_distance_km or route coordinates")
        return self


class RangeQuantiles(BaseModel):
    p10: float
    p50: float
    p90: float


class ShapContribution(BaseModel):
    feature: str
    label: str
    km_impact: float


class SuitabilityResult(BaseModel):
    verdict: str
    safe_verdict: str
    buffer_pct: float
    p10_buffer_pct: float
    daily_requirement_km: float
    claimed_vs_requirement_pct: float
    message: str


class CarbonComparison(BaseModel):
    ev_co2_kg: float
    petrol_co2_kg: float
    co2_saved_kg: float
    ev_cost_inr: float
    petrol_cost_inr: float
    cost_saved_inr: float
    trip_distance_km: float
    trip_energy_kwh: float


class EVPredictionResult(BaseModel):
    ev_model_id: int
    model_name: str
    claimed_range_km: float
    predicted_range: RangeQuantiles
    energy_breakdown: list[ShapContribution]
    suitability: SuitabilityResult
    carbon: CarbonComparison
    warnings: list[str] = []


class PredictResponse(BaseModel):
    scan_id: int
    trip_distance_km: float
    temperature_c: float
    elevation_gain_m: float
    elevation_loss_m: float
    predictions: list[EVPredictionResult]
    route_profile: list[dict] = []
    api_warnings: list[str] = []


class EVModelOut(BaseModel):
    id: int
    model_name: str
    battery_capacity_kwh: float
    claimed_range_km: float
    kerb_weight_kg: float
    drag_coefficient: float
    frontal_area_m2: float
    motor_power_kw: float

    class Config:
        from_attributes = True
