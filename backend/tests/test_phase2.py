"""
Phase 2 tests — route geometry, elevation, weather and fallback flags.

All external HTTP calls are monkeypatched so tests run without network access.
Async tests use the anyio backend (already installed as a FastAPI dependency).
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

BACKEND = Path(__file__).resolve().parent.parent
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))


# ═══════════════════════════════════════════════════════════════════════════════
# Pure helpers — geometry and smoothing (no I/O)
# ═══════════════════════════════════════════════════════════════════════════════

class TestHaversine:
    def test_zero_distance(self):
        from app.services.osrm_service import haversine_m
        assert haversine_m(18.52, 73.86, 18.52, 73.86) == pytest.approx(0.0, abs=1e-3)

    def test_known_distance(self):
        from app.services.osrm_service import haversine_m
        # Pune → Mumbai straight line ≈ 120 km
        dist = haversine_m(18.5204, 73.8567, 19.0760, 72.8777)
        assert 115_000 < dist < 125_000, f"Expected ~120 km, got {dist/1000:.1f} km"


class TestInterpolatePolyline:
    def test_single_point_returns_one_waypoint(self):
        from app.services.osrm_service import interpolate_polyline
        result = interpolate_polyline([[73.0, 18.0]])
        assert len(result) == 1
        assert result[0] == {"lat": 18.0, "lon": 73.0}

    def test_two_points_short_edge_returns_endpoints(self):
        from app.services.osrm_service import interpolate_polyline
        # 100m apart — shorter than 500m segment, so only start+end
        result = interpolate_polyline([[0.0, 0.0], [0.001, 0.0]], segment_m=500.0)
        assert len(result) >= 2
        assert result[0]["lon"] == pytest.approx(0.0)
        assert result[-1]["lon"] == pytest.approx(0.001)

    def test_segment_count_roughly_correct(self):
        from app.services.osrm_service import interpolate_polyline
        # ~10 km straight line → ~20 waypoints at 500 m
        coords = [[0.0, 0.0], [0.0897, 0.0]]  # ≈10 km east
        result = interpolate_polyline(coords, segment_m=500.0)
        assert 18 <= len(result) <= 23, f"Expected ~20 waypoints, got {len(result)}"

    def test_first_and_last_preserved(self):
        from app.services.osrm_service import interpolate_polyline
        coords = [[73.86, 18.52], [73.0, 19.0]]
        result = interpolate_polyline(coords, segment_m=500.0)
        assert result[0]["lat"] == pytest.approx(18.52)
        assert result[0]["lon"] == pytest.approx(73.86)
        assert result[-1]["lat"] == pytest.approx(19.0)
        assert result[-1]["lon"] == pytest.approx(73.0)

    def test_multi_segment_polyline(self):
        from app.services.osrm_service import interpolate_polyline
        # Three waypoints forming a path of ~2 km each
        coords = [[0.0, 0.0], [0.018, 0.0], [0.036, 0.0]]
        result = interpolate_polyline(coords, segment_m=500.0)
        # Total ~4 km → ~8 segments → ~9 points
        assert len(result) >= 7


class TestParseOSRMGeometry:
    def test_valid_response_parsed(self):
        from app.services.osrm_service import parse_osrm_geometry
        data = {
            "routes": [{
                "distance": 144_500.0,
                "duration": 5_800.0,
                "geometry": {
                    "type": "LineString",
                    "coordinates": [[73.86, 18.52], [73.0, 19.0]],
                },
            }]
        }
        result = parse_osrm_geometry(data)
        assert result["distance_m"] == pytest.approx(144_500.0)
        assert result["duration_s"] == pytest.approx(5_800.0)
        assert len(result["coordinates"]) == 2

    def test_empty_routes_raises(self):
        from app.services.osrm_service import parse_osrm_geometry
        with pytest.raises(ValueError, match="no routes"):
            parse_osrm_geometry({"routes": []})

    def test_missing_geometry_raises(self):
        from app.services.osrm_service import parse_osrm_geometry
        with pytest.raises(ValueError):
            parse_osrm_geometry({
                "routes": [{"distance": 1000, "duration": 60, "geometry": {"coordinates": []}}]
            })


# ═══════════════════════════════════════════════════════════════════════════════
# Elevation helpers
# ═══════════════════════════════════════════════════════════════════════════════

class TestSmoothElevations:
    def test_flat_unchanged(self):
        from app.services.elevation_service import smooth_elevations
        flat = [100.0] * 10
        result = smooth_elevations(flat, window=3)
        assert all(abs(v - 100.0) < 1e-9 for v in result)

    def test_spike_reduced(self):
        from app.services.elevation_service import smooth_elevations
        elev = [100.0, 100.0, 200.0, 100.0, 100.0]
        result = smooth_elevations(elev, window=3)
        # The spike at index 2 should be reduced
        assert result[2] < 200.0
        assert result[2] > 100.0

    def test_trend_preserved(self):
        from app.services.elevation_service import smooth_elevations
        # Monotonically increasing
        elev = [float(i * 10) for i in range(10)]
        result = smooth_elevations(elev, window=3)
        # Should still be monotonically increasing
        assert all(result[i] <= result[i + 1] for i in range(len(result) - 1))

    def test_short_list_unchanged(self):
        from app.services.elevation_service import smooth_elevations
        short = [50.0]
        assert smooth_elevations(short, window=3) == [50.0]

    def test_output_length_equals_input(self):
        from app.services.elevation_service import smooth_elevations
        elev = [float(i) for i in range(20)]
        result = smooth_elevations(elev, window=3)
        assert len(result) == len(elev)


class TestComputeElevationProfile:
    def _waypoints(self, n: int, spacing_m: float = 500.0) -> list[dict]:
        """Waypoints along a straight east-west line."""
        # Approx: 1° lat ≈ 111 km, so spacing_m / 111000 degrees
        return [
            {"lat": 18.52 + i * (spacing_m / 111_000), "lon": 73.86}
            for i in range(n)
        ]

    def test_flat_road_zero_gain_loss(self):
        from app.services.elevation_service import compute_elevation_profile
        waypoints = self._waypoints(5)
        elevations = [100.0] * 5
        result = compute_elevation_profile(waypoints, elevations)
        assert result["elevation_gain_m"] == pytest.approx(0.0, abs=0.1)
        assert result["elevation_loss_m"] == pytest.approx(0.0, abs=0.1)

    def test_pure_climb(self):
        from app.services.elevation_service import compute_elevation_profile
        waypoints = self._waypoints(5)
        # climb 10 m per segment
        elevations = [100.0 + i * 10 for i in range(5)]
        result = compute_elevation_profile(waypoints, elevations)
        assert result["elevation_gain_m"] == pytest.approx(40.0, abs=1.0)
        assert result["elevation_loss_m"] == pytest.approx(0.0, abs=0.1)

    def test_pure_descent(self):
        from app.services.elevation_service import compute_elevation_profile
        waypoints = self._waypoints(5)
        elevations = [100.0 - i * 10 for i in range(5)]
        result = compute_elevation_profile(waypoints, elevations)
        assert result["elevation_gain_m"] == pytest.approx(0.0, abs=0.1)
        assert result["elevation_loss_m"] == pytest.approx(40.0, abs=1.0)

    def test_profile_length(self):
        from app.services.elevation_service import compute_elevation_profile
        n = 6
        waypoints = self._waypoints(n)
        elevations = [100.0] * n
        result = compute_elevation_profile(waypoints, elevations)
        assert len(result["profile"]) == n
        assert len(result["gradient_segments"]) == n - 1

    def test_profile_distance_increases(self):
        from app.services.elevation_service import compute_elevation_profile
        n = 5
        waypoints = self._waypoints(n)
        elevations = [100.0] * n
        result = compute_elevation_profile(waypoints, elevations)
        dists = [p["distance_km"] for p in result["profile"]]
        assert dists[0] == pytest.approx(0.0)
        assert all(dists[i] < dists[i + 1] for i in range(len(dists) - 1))


# ═══════════════════════════════════════════════════════════════════════════════
# Async tests — OSRM fallback
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.anyio
async def test_osrm_fallback_on_http_error(monkeypatch):
    """When _osrm_get raises, fetch_osrm_route must return a haversine fallback."""
    import app.services.osrm_service as svc

    async def _fail(url: str):
        raise ConnectionError("simulated network error")

    monkeypatch.setattr(svc, "_osrm_get", _fail)

    result = await svc.fetch_osrm_route(18.52, 73.86, 19.08, 72.88)

    assert result["fallback"] is True
    assert result["fallback_reason"] is not None
    assert result["distance_km"] > 0          # haversine still provides distance
    assert len(result["waypoints"]) >= 2      # at least start + end


@pytest.mark.anyio
async def test_osrm_success(monkeypatch):
    """When _osrm_get returns a valid OSRM response, fallback must be False."""
    import app.services.osrm_service as svc

    async def _ok(url: str) -> dict:
        return {
            "routes": [{
                "distance": 150_000.0,
                "duration": 6_000.0,
                "geometry": {
                    "type": "LineString",
                    "coordinates": [[73.86, 18.52], [73.0, 19.0]],
                },
            }]
        }

    monkeypatch.setattr(svc, "_osrm_get", _ok)

    result = await svc.fetch_osrm_route(18.52, 73.86, 19.08, 72.88)

    assert result["fallback"] is False
    assert result["distance_km"] == pytest.approx(150.0, rel=1e-3)
    assert result["avg_speed_kmh"] == pytest.approx(90.0, rel=0.05)
    assert len(result["waypoints"]) >= 2


# ═══════════════════════════════════════════════════════════════════════════════
# Async tests — elevation fallback
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.anyio
async def test_elevation_fallback_on_http_error(monkeypatch):
    """When _elevation_post raises, fallback=True and elevations are all zero."""
    import app.services.elevation_service as svc

    async def _fail(locations):
        raise ConnectionError("simulated error")

    monkeypatch.setattr(svc, "_elevation_post", _fail)

    waypoints = [{"lat": 18.52, "lon": 73.86}, {"lat": 19.0, "lon": 72.88}]
    elevations, fallback, reason = await svc.fetch_elevations_batched(waypoints)

    assert fallback is True
    assert reason is not None
    assert all(e == 0.0 for e in elevations)


@pytest.mark.anyio
async def test_elevation_success(monkeypatch):
    """When _elevation_post succeeds, fallback=False and correct values returned."""
    import app.services.elevation_service as svc

    async def _ok(locations):
        return [float(10 * i) for i in range(len(locations))]

    monkeypatch.setattr(svc, "_elevation_post", _ok)

    waypoints = [{"lat": 18.0 + i * 0.01, "lon": 73.0} for i in range(5)]
    elevations, fallback, reason = await svc.fetch_elevations_batched(waypoints)

    assert fallback is False
    assert reason is None
    assert len(elevations) == 5
    assert elevations[0] == pytest.approx(0.0)
    assert elevations[-1] == pytest.approx(40.0)


@pytest.mark.anyio
async def test_elevation_batching(monkeypatch):
    """Large waypoint lists must be split into batches (each ≤ ELEVATION_BATCH_SIZE)."""
    import app.services.elevation_service as svc

    call_sizes: list[int] = []

    async def _record(locations):
        call_sizes.append(len(locations))
        return [0.0] * len(locations)

    monkeypatch.setattr(svc, "_elevation_post", _record)

    waypoints = [{"lat": 0.0, "lon": 0.0}] * 250  # 3 batches of 100, 100, 50
    await svc.fetch_elevations_batched(waypoints)

    assert len(call_sizes) == 3, f"Expected 3 batches, got {len(call_sizes)}: {call_sizes}"
    assert max(call_sizes) <= svc.ELEVATION_BATCH_SIZE


# ═══════════════════════════════════════════════════════════════════════════════
# Async tests — weather fallback
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.anyio
async def test_weather_fallback_when_no_api_key(monkeypatch):
    """Without an API key, weather service must return season-average and fallback=True."""
    import app.services.weather_service as svc

    monkeypatch.delenv("OPENWEATHER_API_KEY", raising=False)
    result = await svc.fetch_temperature(18.52, 73.86, season="winter")

    assert result["fallback"] is True
    assert result["api_used"] is False
    assert result["temperature_c"] == svc.SEASON_TEMPS["winter"]
    assert result["fallback_reason"] is not None


@pytest.mark.anyio
async def test_weather_fallback_on_http_error(monkeypatch):
    """When the API key is set but the call fails, fall back to season average."""
    import app.services.weather_service as svc

    monkeypatch.setenv("OPENWEATHER_API_KEY", "fake-key-123")

    async def _fail(url, params):
        raise ConnectionError("simulated error")

    monkeypatch.setattr(svc, "_weather_get", _fail)

    result = await svc.fetch_temperature(18.52, 73.86, season="summer")

    assert result["fallback"] is True
    assert result["temperature_c"] == svc.SEASON_TEMPS["summer"]


@pytest.mark.anyio
async def test_weather_success(monkeypatch):
    """When API key is set and call succeeds, return live temperature with fallback=False."""
    import app.services.weather_service as svc

    monkeypatch.setenv("OPENWEATHER_API_KEY", "fake-key-123")

    async def _ok(url, params):
        return {"main": {"temp": 32.5}}

    monkeypatch.setattr(svc, "_weather_get", _ok)

    result = await svc.fetch_temperature(18.52, 73.86, season="summer")

    assert result["fallback"] is False
    assert result["api_used"] is True
    assert result["temperature_c"] == pytest.approx(32.5)


# ═══════════════════════════════════════════════════════════════════════════════
# Cache helpers
# ═══════════════════════════════════════════════════════════════════════════════

class TestCacheHelpers:
    def _make_db(self):
        """Create a minimal in-memory SQLite session for cache tests."""
        from sqlalchemy import create_engine
        from sqlalchemy.orm import sessionmaker
        from app.models import Base
        engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
        Base.metadata.create_all(bind=engine)
        Session = sessionmaker(bind=engine)
        return Session()

    def test_cache_miss_returns_none(self):
        from app.services.cache import cache_get
        db = self._make_db()
        assert cache_get(db, "nonexistent") is None

    def test_cache_set_then_get(self):
        from app.services.cache import cache_get, cache_set
        db = self._make_db()
        data = {"distance_km": 120.5, "items": [1, 2, 3]}
        cache_set(db, "test:key", data, ttl_seconds=3600)
        result = cache_get(db, "test:key")
        assert result == data

    def test_expired_entry_returns_none(self):
        from app.services.cache import cache_get, cache_set
        from app.models import ApiCache
        from datetime import datetime, timedelta
        db = self._make_db()
        data = {"x": 1}
        cache_set(db, "old:key", data, ttl_seconds=60)
        # Manually back-date the entry
        entry = db.query(ApiCache).filter(ApiCache.key == "old:key").first()
        entry.created_at = datetime.utcnow() - timedelta(seconds=120)
        db.commit()
        assert cache_get(db, "old:key") is None

    def test_cache_overwrite(self):
        from app.services.cache import cache_get, cache_set
        db = self._make_db()
        cache_set(db, "key", {"v": 1})
        cache_set(db, "key", {"v": 2})
        assert cache_get(db, "key") == {"v": 2}

    def test_make_key_deterministic(self):
        from app.services.cache import make_key
        k1 = make_key("osrm", lat1=18.52, lon1=73.86, lat2=19.08, lon2=72.88)
        k2 = make_key("osrm", lat1=18.52, lon1=73.86, lat2=19.08, lon2=72.88)
        assert k1 == k2

    def test_make_key_different_for_different_params(self):
        from app.services.cache import make_key
        k1 = make_key("osrm", lat1=18.52, lon1=73.86)
        k2 = make_key("osrm", lat1=19.0, lon1=72.88)
        assert k1 != k2
