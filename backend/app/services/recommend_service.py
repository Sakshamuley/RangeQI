"""
Recommendation pipeline — Phase 6.

Given a route (via RouteCoords or lat/lon), a price/body-type filter, and
optional driving-profile parameters, this module:

  1. Fetches route geometry + elevation + weather (Phase 2 services)
  2. Filters EVs from the database
  3. For each EV:
       a. Runs the physics model (simulate_energy) over the OSRM segments
       b. Runs the ML quantile predictor (falls back to physics if model absent)
       c. Derives p10/p50/p90 RANGE km (from p90/p50/p10 Wh/km)
       d. Applies the reserve percentage and round-trip flag to compute verdict
       e. Computes trip energy (kWh), estimated charging cost, and CO₂ comparison
  4. Sorts by: (suitability tier, descending range margin)
  5. Returns all results, flagging infeasible EVs

Suitability tiers:
  Suitable    — p10 range ≥ required_km (worst-case estimate still covers trip)
  Marginal    — p50 range ≥ required_km > p10 range
  Not suitable — p50 range < required_km

Key constants (honest assumptions):
  ELECTRICITY_PRICE_INR_PER_KWH  ₹8 per kWh (representative Indian charging cost; assumed)
  PETROL_PRICE_INR_PER_L         ₹100 per litre (approximate; assumed)
  PETROL_FUEL_EFFICIENCY_KM_PER_L 15 km/l (assumed average)
  GRID_EMISSION_FACTOR_KG_CO2_PER_KWH  0.72 (CEA 2022 national grid average)
  PETROL_EMISSION_KG_CO2_PER_L   2.31 (combustion; well-known constant)
"""

from __future__ import annotations

import json
import logging
import math
from typing import Any, Literal, Optional

from sqlalchemy.orm import Session

from app.models import BODY_TYPES, EVModel, PredictionScan
from app.physics_config import PhysicsConfig
from app.schemas import (
    CarbonComparison,
    EVRecommendResult,
    EnergyTermBreakdown,
    RangeQuantiles,
    RecommendRequest,
    RecommendResponse,
)
from app.services.route_service import get_full_route
from app.simulator import EVSpecs, SegmentData, simulate_energy, range_km as phys_range_km
from app.ml.predict import QuantilePredictor

logger = logging.getLogger(__name__)

# ── Honest assumption constants ───────────────────────────────────────────────
# All marked as assumed; no real-time tariff or emission data fetched.
ELECTRICITY_PRICE_INR_PER_KWH    = 8.0    # ₹/kWh — assumed public charging
PETROL_PRICE_INR_PER_L           = 100.0  # ₹/l  — approximate
PETROL_KM_PER_L                  = 15.0   # km/l — assumed average petrol car
GRID_EMISSION_KG_CO2_PER_KWH     = 0.72   # CEA 2022 national average
PETROL_EMISSION_KG_CO2_PER_L     = 2.31   # combustion (IPCC standard)

# Calibrated config (Phase 4 values)
_CALIBRATED_CONFIG = PhysicsConfig(p_aux_kw=1.0, eta_regen=0.30)

# Singleton ML predictor (loaded once per process)
_predictor: Optional[QuantilePredictor] = None


def _get_predictor() -> QuantilePredictor:
    global _predictor
    if _predictor is None:
        _predictor = QuantilePredictor()
    return _predictor


def _segments_from_route(gradient_segments: list) -> list[SegmentData]:
    """Convert route_service gradient_segments to simulator SegmentData list."""
    result = []
    for item in gradient_segments:
        if isinstance(item, (list, tuple)):
            g, l = float(item[0]), float(item[1])
        else:
            g, l = float(item["gradient"]), float(item["length_m"])
        result.append(SegmentData(gradient=g, length_m=l))
    return result


def _carbon(
    trip_energy_kwh: float,
    trip_distance_km: float,
) -> CarbonComparison:
    ev_co2 = trip_energy_kwh * GRID_EMISSION_KG_CO2_PER_KWH
    petrol_litres = trip_distance_km / PETROL_KM_PER_L
    petrol_co2 = petrol_litres * PETROL_EMISSION_KG_CO2_PER_L
    ev_cost = trip_energy_kwh * ELECTRICITY_PRICE_INR_PER_KWH
    petrol_cost = petrol_litres * PETROL_PRICE_INR_PER_L
    return CarbonComparison(
        ev_co2_kg=round(ev_co2, 3),
        petrol_co2_kg=round(petrol_co2, 3),
        co2_saved_kg=round(petrol_co2 - ev_co2, 3),
        ev_cost_inr=round(ev_cost, 2),
        petrol_cost_inr=round(petrol_cost, 2),
        cost_saved_inr=round(petrol_cost - ev_cost, 2),
        trip_distance_km=round(trip_distance_km, 2),
        trip_energy_kwh=round(trip_energy_kwh, 3),
        label="illustrative",
    )


def _verdict(
    p10_range: float,
    p50_range: float,
    required_km: float,
) -> Literal["Suitable", "Marginal", "Not suitable"]:
    if p10_range >= required_km:
        return "Suitable"
    if p50_range >= required_km:
        return "Marginal"
    return "Not suitable"


TIER_ORDER = {"Suitable": 0, "Marginal": 1, "Not suitable": 2}


def _evaluate_ev(
    ev_row: EVModel,
    segments: list[SegmentData],
    route_info: dict,
    req: RecommendRequest,
    required_km: float,
    trip_distance_km: float,
    warnings_out: list[str],
) -> EVRecommendResult:
    """Run physics + ML for one EV and return the result object."""
    usable = ev_row.usable_battery_kwh or ev_row.battery_capacity_kwh * 0.95
    ev = EVSpecs(
        battery_capacity_kwh=ev_row.battery_capacity_kwh,
        usable_battery_kwh=usable,
        kerb_weight_kg=ev_row.kerb_weight_kg,
        drag_coefficient=ev_row.drag_coefficient,
        frontal_area_m2=ev_row.frontal_area_m2,
        motor_power_kw=ev_row.motor_power_kw,
    )

    # ── Physics model (calibrated config, OSRM segments) ─────────────────────
    phys_result = simulate_energy(
        ev=ev,
        segments=segments,
        avg_speed_kmh=route_info["avg_speed_kmh"],
        temperature_c=route_info["temperature_c"],
        driving_style=req.driving_style,
        num_passengers=req.num_passengers,
        ac_on=req.ac_on,
        altitude_m=route_info.get("start_elevation_m", 0.0),
        config=_CALIBRATED_CONFIG,
    )

    # ── ML quantile predictor ─────────────────────────────────────────────────
    distance_km = route_info["distance_km"]
    gain_m  = route_info["elevation_gain_m"]
    loss_m  = route_info["elevation_loss_m"]
    mean_grad = ((gain_m + loss_m) / 2.0) / max(distance_km, 1.0)
    gain_per_km = gain_m / max(distance_km, 1.0)
    loss_per_km = loss_m / max(distance_km, 1.0)

    predictor = _get_predictor()
    ml_pred = predictor.predict(
        ev=ev,
        distance_km=distance_km,
        mean_gradient_mpkm=mean_grad,
        elevation_gain_per_km=gain_per_km,
        elevation_loss_per_km=loss_per_km,
        avg_speed_kmh=route_info["avg_speed_kmh"],
        highway_fraction=min(route_info["avg_speed_kmh"] / 90.0, 1.0),
        temperature_c=route_info["temperature_c"],
        num_passengers=req.num_passengers,
        ac_on=req.ac_on,
        driving_style=req.driving_style,
        elevation_gain_m=gain_m,
        elevation_loss_m=loss_m,
    )
    for w in ml_pred.warnings:
        if w not in warnings_out:
            warnings_out.append(w)

    # ── Range bands ───────────────────────────────────────────────────────────
    # ML gives Wh/km quantiles; higher Wh/km = worse (less range)
    # p10_wh_per_km → optimistic range (p90 range)
    # p90_wh_per_km → conservative range (p10 range)
    p10_range = phys_range_km(ev, ml_pred.p90_wh_per_km)   # conservative
    p50_range = phys_range_km(ev, ml_pred.p50_wh_per_km)   # expected
    p90_range = phys_range_km(ev, ml_pred.p10_wh_per_km)   # optimistic

    # Apply round-trip multiplier to required distance only
    eff_required = required_km  # already includes round-trip factor from caller

    verd = _verdict(p10_range, p50_range, eff_required)
    margin_km  = round(p50_range - eff_required, 1)
    margin_pct = round(margin_km / max(eff_required, 1.0) * 100, 1)

    # ── Energy and cost ───────────────────────────────────────────────────────
    # Use physics p50 (= calibrated config result) for energy accounting
    wh_per_km       = phys_result.total_wh_per_km
    trip_energy_kwh = round(wh_per_km * trip_distance_km / 1000.0, 3)
    trip_cost_inr   = round(trip_energy_kwh * ELECTRICITY_PRICE_INR_PER_KWH, 2)

    breakdown = EnergyTermBreakdown(
        aero_wh_per_km=phys_result.aero_wh_per_km,
        rolling_wh_per_km=phys_result.rolling_wh_per_km,
        gradient_wh_per_km=phys_result.gradient_wh_per_km,
        inertia_wh_per_km=phys_result.inertia_wh_per_km,
        regen_wh_per_km=phys_result.regen_wh_per_km,
        auxiliary_wh_per_km=phys_result.auxiliary_wh_per_km,
        hvac_wh_per_km=phys_result.hvac_wh_per_km,
        total_wh_per_km=phys_result.total_wh_per_km,
    )

    ev_warnings: list[str] = []
    if ev_row.drag_coefficient_assumed:
        ev_warnings.append(f"Cd for {ev_row.model_name} is assumed (not published)")
    if ev_row.usable_battery_assumed:
        ev_warnings.append(f"Usable battery for {ev_row.model_name} is assumed (0.95 × nominal)")
    if ml_pred.source == "physics_fallback":
        ev_warnings.append("ML model unavailable; range band uses physics fallback (±15%)")

    return EVRecommendResult(
        ev_model_id=ev_row.id,
        model_name=ev_row.model_name,
        variant=ev_row.variant,
        body_type=ev_row.body_type,
        ex_showroom_price_lakh=ev_row.ex_showroom_price_lakh,
        price_assumed=ev_row.price_assumed,
        claimed_range_km=ev_row.claimed_range_km,
        claimed_cycle=ev_row.claimed_cycle,
        predicted_range=RangeQuantiles(
            p10=round(p10_range, 1),
            p50=round(p50_range, 1),
            p90=round(p90_range, 1),
        ),
        verdict=verd,
        range_margin_km=margin_km,
        range_margin_pct=margin_pct,
        trip_energy_kwh=trip_energy_kwh,
        energy_wh_per_km=round(wh_per_km, 2),
        trip_cost_inr=trip_cost_inr,
        energy_breakdown=breakdown,
        carbon=_carbon(trip_energy_kwh, trip_distance_km),
        warnings=ev_warnings,
    )


async def recommend(req: RecommendRequest, db: Session) -> RecommendResponse:
    """
    Full recommendation pipeline.

    Returns a RecommendResponse with all EVs sorted by suitability then
    range margin.  Never raises — fallbacks are surfaced as warnings.
    """
    # ── 1. Route + elevation + weather ───────────────────────────────────────
    route_info = await get_full_route(
        req.route.start_lat, req.route.start_lon,
        req.route.end_lat, req.route.end_lon,
        db=db,
    )
    fallback_warnings: list[str] = list(route_info["fallback_warnings"])

    distance_km = route_info["distance_km"]
    trip_distance_km = distance_km * (2 if req.round_trip else 1)
    reserve_factor   = 1.0 + req.reserve_pct / 100.0
    required_km      = round(trip_distance_km * reserve_factor, 1)

    segments = _segments_from_route(route_info["gradient_segments"])

    # ── 2. Filter EV catalogue ────────────────────────────────────────────────
    q = db.query(EVModel)
    if req.body_type:
        q = q.filter(EVModel.body_type == req.body_type)
    evs = q.order_by(EVModel.model_name).all()

    null_price_excluded = 0
    if req.price_min_lakh is not None or req.price_max_lakh is not None:
        filtered = []
        for ev in evs:
            if ev.ex_showroom_price_lakh is None:
                null_price_excluded += 1
                continue
            if req.price_min_lakh and ev.ex_showroom_price_lakh < req.price_min_lakh:
                continue
            if req.price_max_lakh and ev.ex_showroom_price_lakh > req.price_max_lakh:
                continue
            filtered.append(ev)
        evs = filtered
        if null_price_excluded:
            fallback_warnings.append(
                f"{null_price_excluded} EV(s) excluded from price filter "
                f"because ex-showroom price is not yet published."
            )

    # ── 3. Evaluate each EV ───────────────────────────────────────────────────
    results: list[EVRecommendResult] = []
    ml_warnings: list[str] = []

    for ev_row in evs:
        try:
            result = _evaluate_ev(
                ev_row, segments, route_info, req,
                required_km=required_km,
                trip_distance_km=trip_distance_km,
                warnings_out=ml_warnings,
            )
            results.append(result)
        except Exception as exc:
            logger.warning("EV %s evaluation failed: %s", ev_row.model_name, exc)
            fallback_warnings.append(
                f"Could not evaluate {ev_row.model_name}: {exc}"
            )

    for w in ml_warnings:
        if w not in fallback_warnings:
            fallback_warnings.append(w)

    if not results and evs:
        fallback_warnings.append(
            "No EV could be evaluated. Check that all required physics specs are populated."
        )
    elif not evs:
        fallback_warnings.append(
            "No EVs match the given filters. "
            "Note: all prices are currently NULL; price filters exclude all EVs."
        )

    # ── 4. Sort: tier first, then descending margin ───────────────────────────
    results.sort(key=lambda r: (TIER_ORDER[r.verdict], -r.range_margin_km))

    feasible_count = sum(1 for r in results if r.verdict in ("Suitable", "Marginal"))

    # ── 5. Persist scan ───────────────────────────────────────────────────────
    scan = PredictionScan(
        input_json=json.dumps(req.model_dump()),
        output_json=json.dumps({
            "feasible_count": feasible_count,
            "top_verdict": results[0].verdict if results else None,
        }),
    )
    db.add(scan)
    db.commit()
    db.refresh(scan)

    route_profile = route_info.get("profile", [])

    return RecommendResponse(
        scan_id=scan.id,
        trip_distance_km=round(trip_distance_km, 2),
        route_distance_km=round(distance_km, 2),
        temperature_c=round(route_info["temperature_c"], 1),
        elevation_gain_m=round(route_info["elevation_gain_m"], 1),
        elevation_loss_m=round(route_info["elevation_loss_m"], 1),
        required_km=required_km,
        feasible_count=feasible_count,
        recommendations=results,
        route_profile=route_profile[:200],   # cap to keep response lean
        fallback_warnings=fallback_warnings,
    )
