"""
Open-Elevation service with batched POST requests.

Uses the Open-Elevation public API (https://api.open-elevation.com).
Elevation data is fetched via POST to avoid URL-length limits on large
polylines (GET with many locations hits server limits).

Smoothing: centered moving average over 3 samples, applied before gradient
computation to reduce GPS/DEM noise.

Reference: Open-Elevation API docs — https://open-elevation.com
"""

from __future__ import annotations

import logging
from typing import Any

import httpx

logger = logging.getLogger(__name__)

OPEN_ELEVATION_URL = "https://api.open-elevation.com/api/v1/lookup"
ELEVATION_BATCH_SIZE = 100   # points per POST request
SMOOTH_WINDOW = 3            # samples for moving-average smoothing


# ── Pure helpers (no I/O) ─────────────────────────────────────────────────────

def smooth_elevations(elevations: list[float], window: int = SMOOTH_WINDOW) -> list[float]:
    """
    Centered moving-average smoothing.

    Window is centred on each sample; edge samples use a smaller window.
    Reduces DEM/GPS noise before gradient computation.
    """
    n = len(elevations)
    if n < 2:
        return list(elevations)
    half = window // 2
    result: list[float] = []
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        result.append(sum(elevations[lo:hi]) / (hi - lo))
    return result


def compute_elevation_profile(
    waypoints: list[dict[str, float]],
    elevations: list[float],
    segment_m: float = 500.0,
) -> dict[str, Any]:
    """
    Given waypoints [{lat, lon}] and matched elevations (same length),
    return elevation statistics and the per-segment profile.

    Returns:
      elevation_gain_m   total ascent (m)
      elevation_loss_m   total descent (m, positive)
      start_elevation_m
      end_elevation_m
      profile            list of {distance_km, elevation_m, gradient}
      gradient_segments  list of (gradient_fraction, length_m) — used by physics
    """
    from app.services.osrm_service import haversine_m

    n = min(len(waypoints), len(elevations))
    if n < 2:
        return {
            "elevation_gain_m": 0.0,
            "elevation_loss_m": 0.0,
            "start_elevation_m": elevations[0] if elevations else 0.0,
            "end_elevation_m": elevations[-1] if elevations else 0.0,
            "profile": [],
            "gradient_segments": [],
        }

    gain = 0.0
    loss = 0.0
    cum_dist = 0.0
    profile: list[dict] = []
    gradient_segments: list[tuple[float, float]] = []

    # First profile point
    profile.append({
        "distance_km": 0.0,
        "elevation_m": round(elevations[0], 1),
        "gradient": 0.0,
    })

    for i in range(1, n):
        dist_m = haversine_m(
            waypoints[i - 1]["lat"], waypoints[i - 1]["lon"],
            waypoints[i]["lat"], waypoints[i]["lon"],
        )
        cum_dist += dist_m
        delta = elevations[i] - elevations[i - 1]
        if delta > 0:
            gain += delta
        else:
            loss += abs(delta)

        gradient = delta / dist_m if dist_m > 0 else 0.0
        gradient_segments.append((gradient, dist_m))
        profile.append({
            "distance_km": round(cum_dist / 1000.0, 3),
            "elevation_m": round(elevations[i], 1),
            "gradient": round(gradient, 4),
        })

    return {
        "elevation_gain_m": round(gain, 1),
        "elevation_loss_m": round(loss, 1),
        "start_elevation_m": round(elevations[0], 1),
        "end_elevation_m": round(elevations[n - 1], 1),
        "profile": profile,
        "gradient_segments": gradient_segments,
    }


# ── Thin HTTP wrapper (monkeypatched in tests) ────────────────────────────────

async def _elevation_post(locations: list[dict]) -> list[float]:
    """POST one batch to Open-Elevation.  Returns list of elevation floats."""
    payload = {"locations": locations}
    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.post(OPEN_ELEVATION_URL, json=payload)
        resp.raise_for_status()
        data = resp.json()
    results = data.get("results", [])
    if not results:
        raise ValueError("Open-Elevation returned empty results")
    return [float(r["elevation"]) for r in results]


# ── Public API ────────────────────────────────────────────────────────────────

async def fetch_elevations_batched(
    waypoints: list[dict[str, float]],
) -> tuple[list[float], bool, str | None]:
    """
    Fetch elevations for all waypoints using batched POST requests.

    Returns (elevations, fallback, fallback_reason).
    On failure returns all-zero elevations (flat terrain) and fallback=True.
    """
    elevations: list[float] = []
    try:
        for i in range(0, len(waypoints), ELEVATION_BATCH_SIZE):
            batch = waypoints[i: i + ELEVATION_BATCH_SIZE]
            locations = [{"latitude": w["lat"], "longitude": w["lon"]} for w in batch]
            batch_elevs = await _elevation_post(locations)
            elevations.extend(batch_elevs)
        logger.info("Open-Elevation: fetched %d elevations", len(elevations))
        return elevations, False, None
    except Exception as exc:
        logger.warning("Open-Elevation failed: %s — using flat terrain", exc)
        return (
            [0.0] * len(waypoints),
            True,
            f"Elevation API unavailable ({exc}). Using flat-terrain estimate (0 m gain/loss).",
        )


async def get_elevation_profile(
    waypoints: list[dict[str, float]],
    segment_m: float = 500.0,
) -> dict[str, Any]:
    """
    Full pipeline: fetch → smooth → profile.

    Returns dict with elevation stats + fallback flags.
    """
    raw_elevations, fallback, fallback_reason = await fetch_elevations_batched(waypoints)
    smoothed = smooth_elevations(raw_elevations, window=SMOOTH_WINDOW)
    stats = compute_elevation_profile(waypoints, smoothed, segment_m=segment_m)
    stats["elevation_fallback"] = fallback
    stats["elevation_fallback_reason"] = fallback_reason
    return stats
