"""
Route service — aggregates OSRM, elevation and weather into one call.

This module is the single entry point for route data used by the
prediction and recommendation pipelines (Phases 3 and 6).

Every fallback is surfaced as a flag + human-readable message so the
frontend can display a banner listing which data sources were unavailable.
"""

from __future__ import annotations

import logging
from typing import Any

from app.services.osrm_service import (
    SEGMENT_LENGTH_M,
    fetch_osrm_route,
    haversine_m,
)
from app.services.elevation_service import get_elevation_profile
from app.services.weather_service import SEASON_TEMPS, fetch_temperature

logger = logging.getLogger(__name__)

# ── Cache TTLs ────────────────────────────────────────────────────────────────
ROUTE_TTL_S = 86_400       # route geometry: 24 h
ELEVATION_TTL_S = 86_400   # elevation: 24 h (terrain doesn't change)
WEATHER_TTL_S = 1_800      # weather: 30 min


async def get_full_route(
    lat1: float,
    lon1: float,
    lat2: float,
    lon2: float,
    season: str = "summer",
    db: Any = None,
) -> dict[str, Any]:
    """
    Fetch route geometry + elevation + weather for a source→destination pair.

    Caches route and elevation results in the api_cache table when a
    SQLAlchemy session (db) is supplied.

    Returns:
      distance_km, duration_s, avg_speed_kmh
      elevation_gain_m, elevation_loss_m
      start_elevation_m, end_elevation_m
      temperature_c
      waypoints        [{lat, lon}, ...]  at ~500 m intervals
      profile          [{distance_km, elevation_m, gradient}, ...]
      gradient_segments [(gradient, length_m), ...]
      fallback_warnings list[str]  — empty when all APIs succeeded
      route_fallback    bool
      elevation_fallback bool
      weather_fallback   bool
    """
    warnings: list[str] = []

    # ── 1. Route geometry (OSRM) ──────────────────────────────────────────────
    route_result = await _cached_route(lat1, lon1, lat2, lon2, db)
    if route_result["fallback"]:
        warnings.append(route_result["fallback_reason"])

    waypoints = route_result["waypoints"]

    # ── 2. Elevation (Open-Elevation batched POST) ────────────────────────────
    elev_result = await _cached_elevation(lat1, lon1, lat2, lon2, waypoints, db)
    if elev_result["elevation_fallback"]:
        warnings.append(elev_result["elevation_fallback_reason"])

    # ── 3. Weather (OpenWeatherMap) ───────────────────────────────────────────
    mid_lat = (lat1 + lat2) / 2
    mid_lon = (lon1 + lon2) / 2
    wx = await fetch_temperature(mid_lat, mid_lon, season)
    if wx["fallback"]:
        warnings.append(wx["fallback_reason"])

    return {
        "distance_km": route_result["distance_km"],
        "duration_s": route_result["duration_s"],
        "avg_speed_kmh": route_result["avg_speed_kmh"],
        "elevation_gain_m": elev_result["elevation_gain_m"],
        "elevation_loss_m": elev_result["elevation_loss_m"],
        "start_elevation_m": elev_result["start_elevation_m"],
        "end_elevation_m": elev_result["end_elevation_m"],
        "temperature_c": wx["temperature_c"],
        "waypoints": waypoints,
        "profile": elev_result["profile"],
        "gradient_segments": elev_result["gradient_segments"],
        "fallback_warnings": warnings,
        "route_fallback": route_result["fallback"],
        "elevation_fallback": elev_result["elevation_fallback"],
        "weather_fallback": wx["fallback"],
    }


# ── Cache wrappers ────────────────────────────────────────────────────────────

async def _cached_route(
    lat1: float, lon1: float,
    lat2: float, lon2: float,
    db: Any,
) -> dict:
    if db is not None:
        from app.services.cache import cache_get, cache_set, make_key
        key = make_key("osrm", lat1=round(lat1, 3), lon1=round(lon1, 3),
                       lat2=round(lat2, 3), lon2=round(lon2, 3))
        cached = cache_get(db, key)
        if cached is not None:
            return cached

    result = await fetch_osrm_route(lat1, lon1, lat2, lon2)

    if db is not None:
        cache_set(db, key, result, ttl_seconds=ROUTE_TTL_S)

    return result


async def _cached_elevation(
    lat1: float, lon1: float,
    lat2: float, lon2: float,
    waypoints: list[dict],
    db: Any,
) -> dict:
    if db is not None:
        from app.services.cache import cache_get, cache_set, make_key
        key = make_key("elev", lat1=round(lat1, 3), lon1=round(lon1, 3),
                       lat2=round(lat2, 3), lon2=round(lon2, 3),
                       n=len(waypoints))
        cached = cache_get(db, key)
        if cached is not None:
            return cached

    result = await get_elevation_profile(waypoints, segment_m=SEGMENT_LENGTH_M)

    if db is not None:
        # gradient_segments contains tuples — serialize as lists
        result_serializable = dict(result)
        result_serializable["gradient_segments"] = [
            list(t) for t in result["gradient_segments"]
        ]
        cache_set(db, key, result_serializable, ttl_seconds=ELEVATION_TTL_S)
        # restore tuple form for the caller
        result["gradient_segments"] = [
            tuple(t) for t in result_serializable["gradient_segments"]
        ]

    return result
