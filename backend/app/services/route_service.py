"""External API integrations: elevation and weather with graceful fallbacks."""

from __future__ import annotations

import logging
import math
import os
import time
from typing import Any

import httpx

logger = logging.getLogger(__name__)

OPEN_ELEVATION_URL = "https://api.open-elevation.com/api/v1/lookup"
OPENWEATHER_URL = "https://api.openweathermap.org/data/2.5/weather"

# Defaults when APIs fail — flat terrain, moderate Indian climate
DEFAULT_ELEVATION_GAIN_M = 50.0
DEFAULT_ELEVATION_LOSS_M = 50.0
DEFAULT_TEMPERATURE_C = 30.0

SEASON_TEMPS = {
    "summer": 38.0,
    "monsoon": 28.0,
    "winter": 18.0,
    "spring": 26.0,
}


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance between two coordinates."""
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def interpolate_route_points(
    start_lat: float,
    start_lon: float,
    end_lat: float,
    end_lon: float,
    sample_km: float = 1.0,
) -> list[dict[str, float]]:
    """Sample points along straight-line route every ~sample_km."""
    total_km = haversine_km(start_lat, start_lon, end_lat, end_lon)
    if total_km < 0.5:
        return [
            {"latitude": start_lat, "longitude": start_lon},
            {"latitude": end_lat, "longitude": end_lon},
        ]
    n = max(int(total_km / sample_km), 2)
    points = []
    for i in range(n + 1):
        t = i / n
        points.append(
            {
                "latitude": start_lat + t * (end_lat - start_lat),
                "longitude": start_lon + t * (end_lon - start_lon),
            }
        )
    return points


async def fetch_elevation_profile(
    start_lat: float,
    start_lon: float,
    end_lat: float,
    end_lon: float,
) -> dict[str, Any]:
    """
    Fetch elevation via Open-Elevation API, sample every ~1 km.
    On failure returns flat-terrain defaults and warning flag.
    """
    t0 = time.perf_counter()
    points = interpolate_route_points(start_lat, start_lon, end_lat, end_lon)
    distance_km = haversine_km(start_lat, start_lon, end_lat, end_lon)
    warning = None
    profile: list[dict] = []

    try:
        locations = "|".join(f"{p['latitude']},{p['longitude']}" for p in points)
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.get(f"{OPEN_ELEVATION_URL}?locations={locations}")
            resp.raise_for_status()
            data = resp.json()

        elevations = [r["elevation"] for r in data.get("results", [])]
        if len(elevations) < 2:
            raise ValueError("Insufficient elevation data")

        gain, loss = 0.0, 0.0
        for i in range(1, len(elevations)):
            delta = elevations[i] - elevations[i - 1]
            dist_frac = distance_km / max(len(elevations) - 1, 1)
            profile.append(
                {
                    "distance_km": round(i * dist_frac, 2),
                    "elevation_m": round(elevations[i], 1),
                }
            )
            if delta > 0:
                gain += delta
            else:
                loss += abs(delta)

        elapsed = time.perf_counter() - t0
        return {
            "distance_km": round(distance_km, 2),
            "elevation_gain_m": round(gain, 1),
            "elevation_loss_m": round(loss, 1),
            "profile": profile,
            "start_elevation_m": round(elevations[0], 1),
            "end_elevation_m": round(elevations[-1], 1),
            "api_used": True,
            "warning": None,
            "execution_time_ms": round(elapsed * 1000, 2),
        }
    except Exception as exc:
        logger.warning("Open-Elevation API failed: %s — using defaults", exc)
        warning = f"Elevation API unavailable ({exc}). Using flat-terrain estimate."
        return {
            "distance_km": round(distance_km, 2),
            "elevation_gain_m": DEFAULT_ELEVATION_GAIN_M,
            "elevation_loss_m": DEFAULT_ELEVATION_LOSS_M,
            "profile": [],
            "start_elevation_m": 200.0,
            "end_elevation_m": 200.0,
            "api_used": False,
            "warning": warning,
            "execution_time_ms": round((time.perf_counter() - t0) * 1000, 2),
        }


async def fetch_temperature(lat: float, lon: float, season: str) -> dict[str, Any]:
    """
    Fetch current weather via OpenWeatherMap; fallback to season average.
    Requires OPENWEATHER_API_KEY env var for live data.
    """
    t0 = time.perf_counter()
    api_key = os.getenv("OPENWEATHER_API_KEY", "")

    if api_key:
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(
                    OPENWEATHER_URL,
                    params={"lat": lat, "lon": lon, "appid": api_key, "units": "metric"},
                )
                resp.raise_for_status()
                temp = resp.json()["main"]["temp"]
                return {
                    "temperature_c": round(float(temp), 1),
                    "api_used": True,
                    "warning": None,
                    "execution_time_ms": round((time.perf_counter() - t0) * 1000, 2),
                }
        except Exception as exc:
            logger.warning("OpenWeatherMap API failed: %s", exc)

    temp = SEASON_TEMPS.get(season, DEFAULT_TEMPERATURE_C)
    return {
        "temperature_c": temp,
        "api_used": False,
        "warning": "Weather API unavailable. Using season-average temperature.",
        "execution_time_ms": round((time.perf_counter() - t0) * 1000, 2),
    }
