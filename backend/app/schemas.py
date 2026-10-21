"""Pydantic request/response schemas for RangeIQ."""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field, model_validator

DrivingStyle = Literal["aggressive", "normal", "calm"]
BodyType = Literal["hatchback", "sedan", "compact_suv", "suv", "mpv"]


# ── Input schemas ─────────────────────────────────────────────────────────────

class RouteCoords(BaseModel):
    """Source and destination as decimal-degree coordinates."""
    start_lat: float = Field(..., ge=-90.0, le=90.0)
    start_lon: float = Field(..., ge=-180.0, le=180.0)
    end_lat: float = Field(..., ge=-90.0, le=90.0)
    end_lon: float = Field(..., ge=-180.0, le=180.0)


class PredictRequest(BaseModel):
    """
    POST /api/predict — check a specific EV model against a route.
    At least one of daily_distance_km or route must be provided.
    """
    ev_model_ids: list[int] = Field(..., min_length=1)
    daily_distance_km: Optional[float] = Field(None, gt=0)
    route: Optional[RouteCoords] = None
    driving_style: DrivingStyle = "normal"
    num_passengers: int = Field(1, ge=1, le=7)
    ac_on: bool = True
    reserve_pct: float = Field(10.0, ge=5.0, le=20.0)
    round_trip: bool = False

    @model_validator(mode="after")
    def require_distance_or_route(self) -> "PredictRequest":
        if self.daily_distance_km is None and self.route is None:
            raise ValueError("Provide daily_distance_km or route coordinates")
        return self


class RecommendRequest(BaseModel):
    """
    POST /api/recommend — find ranked EVs for a route within a price/body-type filter.
    The user does NOT pick an EV model; the system shortlists and ranks them.
    """
    route: RouteCoords
    price_min_lakh: Optional[float] = Field(None, ge=0)
    price_max_lakh: Optional[float] = Field(None, ge=0)
    body_type: Optional[BodyType] = None
    # Advanced (optional)
    driving_style: DrivingStyle = "normal"
    num_passengers: int = Field(1, ge=1, le=7)
    ac_on: bool = True
    reserve_pct: float = Field(10.0, ge=5.0, le=20.0)
    round_trip: bool = False


# ── Output schemas ────────────────────────────────────────────────────────────

class RangeQuantiles(BaseModel):
    """p10 = conservative, p50 = expected, p90 = optimistic."""
    p10: float
    p50: float
    p90: float


class EnergyTermBreakdown(BaseModel):
    """Per-term energy in Wh/km, summed over the trip."""
    aero_wh_per_km: float
    rolling_wh_per_km: float
    gradient_wh_per_km: float
    inertia_wh_per_km: float
    regen_wh_per_km: float        # negative = energy returned
    auxiliary_wh_per_km: float
    hvac_wh_per_km: float
    total_wh_per_km: float


class SuitabilityResult(BaseModel):
    """
    Verdict logic (Phase 6):
      required_km = trip_distance × (2 if round_trip else 1) × (1 + reserve_pct/100)
      Suitable    if p10_range >= required_km
      Marginal    if p50_range >= required_km > p10_range
      Not suitable if p50_range < required_km
    """
    verdict: Literal["Suitable", "Marginal", "Not suitable"]
    required_km: float
    p10_range_km: float
    p50_range_km: float
    range_margin_km: float          # p50_range - required_km
    range_margin_pct: float


class CarbonComparison(BaseModel):
    """Illustrative CO₂ and cost comparison (label 'illustrative' in UI)."""
    ev_co2_kg: float
    petrol_co2_kg: float
    co2_saved_kg: float
    ev_cost_inr: float
    petrol_cost_inr: float
    cost_saved_inr: float
    trip_distance_km: float
    trip_energy_kwh: float
    label: str = "illustrative"


class EVRecommendResult(BaseModel):
    """Single EV entry in the recommendation response."""
    ev_model_id: int
    model_name: str
    variant: Optional[str] = None
    body_type: str
    ex_showroom_price_lakh: Optional[float] = None
    price_assumed: bool
    claimed_range_km: float
    claimed_cycle: str
    predicted_range: RangeQuantiles
    verdict: Literal["Suitable", "Marginal", "Not suitable"]
    range_margin_km: float
    range_margin_pct: float
    trip_energy_kwh: float
    energy_wh_per_km: float
    trip_cost_inr: float
    energy_breakdown: EnergyTermBreakdown
    carbon: CarbonComparison
    tie_group: Optional[int] = None   # EVs with margins within 5pp share a group
    warnings: list[str] = []


class RecommendResponse(BaseModel):
    scan_id: int
    trip_distance_km: float
    route_distance_km: float          # road distance (from OSRM)
    temperature_c: float
    elevation_gain_m: float
    elevation_loss_m: float
    required_km: float
    feasible_count: int
    recommendations: list[EVRecommendResult]
    route_profile: list[dict] = []
    fallback_warnings: list[str] = []


class EVPredictionResult(BaseModel):
    """Result for POST /api/predict (single-model mode)."""
    ev_model_id: int
    model_name: str
    claimed_range_km: float
    predicted_range: RangeQuantiles
    verdict: Literal["Suitable", "Marginal", "Not suitable"]
    range_margin_km: float
    trip_energy_kwh: float
    energy_breakdown: EnergyTermBreakdown
    carbon: CarbonComparison
    warnings: list[str] = []


class PredictResponse(BaseModel):
    scan_id: int
    trip_distance_km: float
    temperature_c: float
    elevation_gain_m: float
    elevation_loss_m: float
    required_km: float
    predictions: list[EVPredictionResult]
    route_profile: list[dict] = []
    api_warnings: list[str] = []


# ── EV model read schema ──────────────────────────────────────────────────────

class EVModelOut(BaseModel):
    id: int
    model_name: str
    variant: Optional[str] = None
    body_type: str
    seats: int
    ex_showroom_price_lakh: Optional[float] = None
    price_assumed: bool
    claimed_range_km: float
    claimed_cycle: str
    battery_capacity_kwh: float
    usable_battery_kwh: Optional[float] = None
    usable_battery_assumed: bool
    kerb_weight_kg: float
    drag_coefficient: float
    drag_coefficient_assumed: bool
    frontal_area_m2: float
    frontal_area_assumed: bool
    motor_power_kw: float
    drivetrain_eff_assumed: bool
    regen_eff_assumed: bool

    model_config = {"from_attributes": True}
