"""
OSRM route service.

Fetches real road geometry, road distance and duration from the OSRM
routing engine.  The default endpoint is the public demo instance at
router.project-osrm.org; set OSRM_BASE_URL in the environment to point
at a self-hosted instance.

Reference: OSRM HTTP API v5 — http://project-osrm.org/docs/v5.24.0/api/
Geometry is GeoJSON LineString with [lon, lat] coordinate order.
"""

from __future__ import annotations

import logging
import math
import os
from typing import Any

import httpx

logger = logging.getLogger(__name__)

OSRM_BASE_URL = os.getenv(
    "OSRM_BASE_URL",
    "http://router.project-osrm.org/route/v1/driving",
)

# Every ~500 m along the road a waypoint is placed for elevation sampling.
# Phase 3 uses these waypoints as physics-model segments.
SEGMENT_LENGTH_M: float = 500.0


# ── Pure geometry helpers ─────────────────────────────────────────────────────

def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in metres."""
    r = 6_371_000.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(max(0.0, a)))


def interpolate_polyline(
    coords: list[list[float]],
    segment_m: float = SEGMENT_LENGTH_M,
) -> list[dict[str, float]]:
    """
    Resample a GeoJSON polyline (list of [lon, lat] pairs) to waypoints
    spaced approximately segment_m metres apart.

    Always includes the first and last coordinate.
    Returns a list of {"lat": float, "lon": float} dicts.
    """
    if not coords:
        return []

    result: list[dict[str, float]] = [{"lat": coords[0][1], "lon": coords[0][0]}]
    if len(coords) == 1:
        return result

    accumulated = 0.0  # metres accumulated since last sample point

    for i in range(1, len(coords)):
        prev_lon, prev_lat = coords[i - 1]
        curr_lon, curr_lat = coords[i]
        edge_dist = haversine_m(prev_lat, prev_lon, curr_lat, curr_lon)

        if edge_dist == 0.0:
            continue

        # Walk along this edge, placing sample points every segment_m metres.
        t = 0.0  # fractional position along edge [0, 1]
        while True:
            needed = segment_m - accumulated
            remaining = edge_dist * (1.0 - t)
            if remaining < needed:
                accumulated += remaining
                break
            # Place one more waypoint
            t += needed / edge_dist
            accumulated = 0.0
            result.append({
                "lat": prev_lat + t * (curr_lat - prev_lat),
                "lon": prev_lon + t * (curr_lon - prev_lon),
            })

    # Always append last coord if not already there
    last = {"lat": coords[-1][1], "lon": coords[-1][0]}
    tail = result[-1]
    if abs(tail["lat"] - last["lat"]) > 1e-8 or abs(tail["lon"] - last["lon"]) > 1e-8:
        result.append(last)

    return result


def parse_osrm_geometry(data: dict[str, Any]) -> dict[str, Any]:
    """
    Extract route distance, duration and GeoJSON coordinates from an OSRM
    /route response.  Raises ValueError if the response is malformed.
    """
    routes = data.get("routes")
    if not routes:
        raise ValueError("OSRM response contains no routes")
    route = routes[0]
    geom = route.get("geometry", {})
    coords = geom.get("coordinates")
    if not coords:
        raise ValueError("OSRM route geometry is empty")
    return {
        "distance_m": float(route["distance"]),
        "duration_s": float(route["duration"]),
        "coordinates": coords,  # [[lon, lat], ...]
    }


def _haversine_fallback(
    lat1: float, lon1: float, lat2: float, lon2: float, reason: str
) -> dict[str, Any]:
    """Straight-line fallback when OSRM is unavailable."""
    dist_m = haversine_m(lat1, lon1, lat2, lon2)
    # Assume 60 km/h average — flagged as assumed
    assumed_speed_kmh = 60.0
    duration_s = (dist_m / 1000.0) / assumed_speed_kmh * 3600.0
    # Simple two-point geometry along straight line
    waypoints = interpolate_polyline(
        [[lon1, lat1], [lon2, lat2]], segment_m=SEGMENT_LENGTH_M
    )
    return {
        "distance_km": round(dist_m / 1000.0, 3),
        "duration_s": round(duration_s, 1),
        "avg_speed_kmh": assumed_speed_kmh,
        "waypoints": waypoints,
        "fallback": True,
        "fallback_reason": (
            f"OSRM route unavailable ({reason}). "
            "Using straight-line distance and assumed speed of 60 km/h."
        ),
    }


# ── Thin HTTP wrapper (replaced by monkeypatch in tests) ─────────────────────

async def _osrm_get(url: str) -> dict:
    """Fetch OSRM route JSON.  Isolated so tests can monkeypatch this."""
    async with httpx.AsyncClient(timeout=15.0) as client:
        resp = await client.get(url)
        resp.raise_for_status()
        return resp.json()


# ── Public API ────────────────────────────────────────────────────────────────

async def fetch_osrm_route(
    lat1: float, lon1: float,
    lat2: float, lon2: float,
) -> dict[str, Any]:
    """
    Fetch road route from OSRM and return geometry segmented at SEGMENT_LENGTH_M.

    Returns dict with keys:
      distance_km, duration_s, avg_speed_kmh, waypoints [{lat, lon}],
      fallback (bool), fallback_reason (str | None).
    """
    url = (
        f"{OSRM_BASE_URL}/{lon1},{lat1};{lon2},{lat2}"
        f"?overview=full&geometries=geojson"
    )
    try:
        data = await _osrm_get(url)
        parsed = parse_osrm_geometry(data)
        waypoints = interpolate_polyline(parsed["coordinates"], SEGMENT_LENGTH_M)
        dist_km = parsed["distance_m"] / 1000.0
        dur_s = parsed["duration_s"]
        speed_kmh = (dist_km / (dur_s / 3600.0)) if dur_s > 0 else 50.0
        logger.info(
            "OSRM: %.1f km, %.0f s, %.1f km/h, %d waypoints",
            dist_km, dur_s, speed_kmh, len(waypoints),
        )
        return {
            "distance_km": round(dist_km, 3),
            "duration_s": round(dur_s, 1),
            "avg_speed_kmh": round(speed_kmh, 1),
            "waypoints": waypoints,
            "fallback": False,
            "fallback_reason": None,
        }
    except Exception as exc:
        logger.warning("OSRM request failed: %s — using straight-line fallback", exc)
        return _haversine_fallback(lat1, lon1, lat2, lon2, str(exc))
