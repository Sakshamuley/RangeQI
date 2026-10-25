"""
Geocoding via Nominatim (OpenStreetMap) — no API key required.

Returns (lat, lon) for a place name or raises ValueError if not found.
Results are cached in the api_cache table.

Rate limit: Nominatim asks for max 1 req/s; we cache aggressively (7 days)
to avoid hammering the free service.
"""

from __future__ import annotations

import logging
from typing import Any

import httpx

logger = logging.getLogger(__name__)

NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
GEOCODE_TTL_S = 7 * 86_400   # 7 days — place names don't move
USER_AGENT = "RangeIQ/1.0 (educational EV range tool; contact@rangeiq.local)"


async def geocode(place: str, db: Any = None) -> tuple[float, float]:
    """
    Resolve a place name to (lat, lon).

    Args:
        place: human-readable location (e.g. "Pune, Maharashtra")
        db:    SQLAlchemy session for caching (optional)

    Returns:
        (lat, lon) as floats

    Raises:
        ValueError: if Nominatim returns no results
        httpx.HTTPError: on network failure
    """
    if db is not None:
        from app.services.cache import cache_get, cache_set, make_key
        key = make_key("geo", place=place.strip().lower())
        cached = cache_get(db, key)
        if cached is not None:
            return float(cached["lat"]), float(cached["lon"])

    async with httpx.AsyncClient(timeout=10.0) as client:
        resp = await client.get(
            NOMINATIM_URL,
            params={"q": place, "format": "json", "limit": 1},
            headers={"User-Agent": USER_AGENT},
        )
        resp.raise_for_status()
        results = resp.json()

    if not results:
        raise ValueError(f"Location not found: '{place}'. Try a more specific name.")

    lat = float(results[0]["lat"])
    lon = float(results[0]["lon"])

    if db is not None:
        cache_set(db, key, {"lat": lat, "lon": lon}, ttl_seconds=GEOCODE_TTL_S)

    return lat, lon
