"""
Weather service with OpenWeatherMap integration and seasonal fallback.

Reads OPENWEATHER_API_KEY from the environment (via load_dotenv in main.py).
When the key is absent or the API call fails, falls back to season-average
temperatures and sets a visible fallback flag.

Season averages are representative of North/West India plains:
  summer (Apr–Jun): 38 °C
  monsoon (Jul–Sep): 28 °C
  winter (Nov–Feb):  18 °C
  spring (Mar):      26 °C
"""

from __future__ import annotations

import logging
import os
from typing import Any

import httpx

logger = logging.getLogger(__name__)

OPENWEATHER_URL = "https://api.openweathermap.org/data/2.5/weather"

SEASON_TEMPS: dict[str, float] = {
    "summer": 38.0,
    "monsoon": 28.0,
    "winter": 18.0,
    "spring": 26.0,
}
DEFAULT_TEMPERATURE_C = 30.0


# ── Thin HTTP wrapper ─────────────────────────────────────────────────────────

async def _weather_get(url: str, params: dict) -> dict:
    """Fetch OpenWeatherMap data.  Isolated for monkeypatching in tests."""
    async with httpx.AsyncClient(timeout=10.0) as client:
        resp = await client.get(url, params=params)
        resp.raise_for_status()
        return resp.json()


# ── Public API ────────────────────────────────────────────────────────────────

async def fetch_temperature(
    lat: float,
    lon: float,
    season: str = "summer",
) -> dict[str, Any]:
    """
    Fetch current ambient temperature for a location.

    Priority:
      1. OpenWeatherMap live reading (if OPENWEATHER_API_KEY is set)
      2. Season-average fallback (always available)

    Returns dict with keys:
      temperature_c (float)
      api_used (bool)
      fallback (bool)
      fallback_reason (str | None)
    """
    api_key = os.getenv("OPENWEATHER_API_KEY", "").strip()

    if api_key:
        try:
            data = await _weather_get(
                OPENWEATHER_URL,
                {"lat": lat, "lon": lon, "appid": api_key, "units": "metric"},
            )
            temp = float(data["main"]["temp"])
            logger.info("OpenWeatherMap: %.1f °C at (%.3f, %.3f)", temp, lat, lon)
            return {
                "temperature_c": round(temp, 1),
                "api_used": True,
                "fallback": False,
                "fallback_reason": None,
            }
        except Exception as exc:
            logger.warning("OpenWeatherMap failed: %s", exc)
            reason = (
                f"Weather API error ({exc}). "
                f"Using season-average temperature for '{season}'."
            )
    else:
        reason = (
            "OPENWEATHER_API_KEY not set. "
            f"Using season-average temperature for '{season}' "
            "(set the key in backend/.env for live data)."
        )

    temp = SEASON_TEMPS.get(season, DEFAULT_TEMPERATURE_C)
    return {
        "temperature_c": temp,
        "api_used": False,
        "fallback": True,
        "fallback_reason": reason,
    }
