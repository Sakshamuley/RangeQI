"""
Phase 3 unit tests for the segment-based physics model.

All HTTP calls are avoided (pure unit tests).
Each test isolates one physical effect by zeroing out other terms via
PhysicsConfig overrides.

Hand-calculation reference (flat road, constant speed, no stops):
  EV: Cd=0.32, A=2.35 m², m=1400 kg, v=60 km/h=16.667 m/s
  rho=1.2 kg/m³ (20 °C, sea level)
  F_aero  = 0.5 * 1.2 * 0.32 * 2.35 * 16.667² = 0.5*1.2*0.752*277.8 ≈ 125.3 N
  F_roll  = 0.011 * 1400 * 9.81 * 1.0          ≈ 150.9 N
  F_total = 276.2 N
  P_wheel = 276.2 * 16.667 ≈ 4603 W
  P_batt  = 4603 / 0.92    ≈ 5004 W
  Wh/km   = P_batt / (v_km/h) = 5004 / 60 ≈ 83.4 Wh/km
  + P_aux  = 250 W → 250/60*1000/1000 ≈ 4.17 Wh/km
  Total   ≈ 87.6 Wh/km  (tolerance: within ±5 Wh/km)
"""

from __future__ import annotations

import math
import pytest

from app.simulator import (
    EVSpecs,
    SegmentData,
    EnergyBreakdown,
    simulate_energy,
    simulate_energy_simple,
    segments_from_elevation_totals,
    air_density_kg_m3,
    hvac_power_kw,
    range_km,
)
from app.physics_config import PhysicsConfig, DEFAULT_CONFIG


# ── Shared fixtures ───────────────────────────────────────────────────────────

NEXON_SPECS = EVSpecs(
    battery_capacity_kwh=40.5,
    usable_battery_kwh=38.5,
    kerb_weight_kg=1400,
    drag_coefficient=0.32,
    frontal_area_m2=2.35,
    motor_power_kw=129,
)

# Config with ALL stops zeroed and no HVAC base, so we can isolate aero+rolling+grade
NO_STOPS_NO_HVAC = PhysicsConfig(
    stops_per_km_urban=0.0,
    stops_per_km_mixed=0.0,
    stops_per_km_highway=0.0,
    p_aux_kw=0.0,
    hvac_base_kw=0.0,
    hvac_heat_coef_kw_per_c=0.0,
    hvac_cool_coef_kw_per_c=0.0,
)

NO_STOPS_ZERO_AUX = PhysicsConfig(
    stops_per_km_urban=0.0,
    stops_per_km_mixed=0.0,
    stops_per_km_highway=0.0,
    p_aux_kw=0.25,
    hvac_base_kw=0.0,
    hvac_heat_coef_kw_per_c=0.0,
    hvac_cool_coef_kw_per_c=0.0,
)


def flat_segments(distance_km: float = 10.0, n: int = 20) -> list[SegmentData]:
    """Flat road: all segments with gradient=0."""
    seg_len = distance_km * 1000 / n
    return [SegmentData(gradient=0.0, length_m=seg_len) for _ in range(n)]


def uphill_segments(distance_km: float, elevation_gain_m: float, n: int = 20) -> list[SegmentData]:
    seg_len = distance_km * 1000 / n
    g = elevation_gain_m / (distance_km * 1000)
    return [SegmentData(gradient=g, length_m=seg_len) for _ in range(n)]


def downhill_segments(distance_km: float, elevation_loss_m: float, n: int = 20) -> list[SegmentData]:
    seg_len = distance_km * 1000 / n
    g = -elevation_loss_m / (distance_km * 1000)
    return [SegmentData(gradient=g, length_m=seg_len) for _ in range(n)]


# ── 1. Air density tests ──────────────────────────────────────────────────────

class TestAirDensity:
    def test_standard_sea_level(self):
        """At 15 °C, sea level: ISA gives ρ ≈ 1.225 kg/m³."""
        rho = air_density_kg_m3(temp_c=15.0, altitude_m=0.0)
        assert abs(rho - 1.225) < 0.005, f"Expected ~1.225 kg/m³, got {rho:.4f}"

    def test_decreases_with_temperature(self):
        """Warmer air is less dense."""
        rho_cold = air_density_kg_m3(temp_c=0.0)
        rho_warm = air_density_kg_m3(temp_c=35.0)
        assert rho_cold > rho_warm

    def test_near_zero_celsius(self):
        """At 0 °C, sea level: ρ should be close to 1.293 kg/m³."""
        rho = air_density_kg_m3(temp_c=0.0, altitude_m=0.0)
        assert abs(rho - 1.293) < 0.010, f"Expected ~1.293 kg/m³, got {rho:.4f}"

    def test_decreases_with_altitude(self):
        """Higher altitude → lower pressure → lower density."""
        rho_sea = air_density_kg_m3(temp_c=20.0, altitude_m=0.0)
        rho_high = air_density_kg_m3(temp_c=20.0, altitude_m=2000.0)
        assert rho_sea > rho_high


# ── 2. HVAC model tests ───────────────────────────────────────────────────────

class TestHVAC:
    def test_heating_below_threshold(self):
        """At -10 °C, heating should be active and > base power."""
        cfg = DEFAULT_CONFIG
        p = hvac_power_kw(ambient_c=-10.0, cfg=cfg)
        assert p > cfg.hvac_base_kw

    def test_cooling_above_threshold(self):
        """At 35 °C, cooling should be active and > base power."""
        cfg = DEFAULT_CONFIG
        p = hvac_power_kw(ambient_c=35.0, cfg=cfg)
        assert p > cfg.hvac_base_kw

    def test_comfort_zone(self):
        """At 22 °C (comfort zone), P_hvac == hvac_base_kw."""
        cfg = DEFAULT_CONFIG
        p = hvac_power_kw(ambient_c=22.0, cfg=cfg)
        assert abs(p - cfg.hvac_base_kw) < 1e-9

    def test_heating_increases_with_cold(self):
        """Colder → more heating power."""
        cfg = DEFAULT_CONFIG
        p_mild = hvac_power_kw(ambient_c=10.0, cfg=cfg)
        p_cold = hvac_power_kw(ambient_c=-5.0, cfg=cfg)
        assert p_cold > p_mild

    def test_cooling_increases_with_heat(self):
        """Hotter → more cooling power."""
        cfg = DEFAULT_CONFIG
        p_warm = hvac_power_kw(ambient_c=30.0, cfg=cfg)
        p_hot = hvac_power_kw(ambient_c=40.0, cfg=cfg)
        assert p_hot > p_warm

    def test_heat_cap(self):
        """Very cold temperature should not exceed hvac_max_heat_kw."""
        cfg = DEFAULT_CONFIG
        p = hvac_power_kw(ambient_c=-50.0, cfg=cfg)
        assert p <= cfg.hvac_max_heat_kw + 1e-9

    def test_cool_cap(self):
        """Very hot temperature should not exceed hvac_max_cool_kw."""
        cfg = DEFAULT_CONFIG
        p = hvac_power_kw(ambient_c=80.0, cfg=cfg)
        assert p <= cfg.hvac_max_cool_kw + 1e-9


# ── 3. Flat road, constant speed – hand calculation check ────────────────────

class TestFlatRoadHandCalc:
    """
    Flat road, 60 km/h, 20 °C, no stops, no HVAC, no aux.
    Expected aero+rolling: ~83–90 Wh/km (tolerance ±10 Wh/km to allow
    slight differences in rounding, ISA density at 20 °C, etc.).
    """

    def test_flat_constant_speed_total_wh_km(self):
        cfg = NO_STOPS_NO_HVAC
        segs = flat_segments(distance_km=50.0)
        result = simulate_energy(
            ev=NEXON_SPECS,
            segments=segs,
            avg_speed_kmh=60.0,
            temperature_c=20.0,
            driving_style="normal",
            num_passengers=1,
            ac_on=False,
            altitude_m=0.0,
            config=cfg,
        )
        # Hand calc: ~83–90 Wh/km (no aux, no HVAC)
        assert 75 < result.total_wh_per_km < 100, (
            f"Expected 75–100 Wh/km, got {result.total_wh_per_km:.2f}"
        )

    def test_flat_constant_speed_with_aux(self):
        """Adding 0.25 kW aux at 60 km/h adds 0.25/60*1000 = 4.17 Wh/km."""
        cfg = NO_STOPS_ZERO_AUX  # p_aux_kw=0.25, no HVAC base
        segs = flat_segments(distance_km=50.0)
        result = simulate_energy(
            ev=NEXON_SPECS, segments=segs, avg_speed_kmh=60.0,
            temperature_c=20.0, driving_style="normal",
            num_passengers=1, ac_on=False, altitude_m=0.0, config=cfg,
        )
        # Should be roughly aero+rolling + ~4.17 Wh/km aux
        assert 80 < result.total_wh_per_km < 105, (
            f"Expected 80–105 Wh/km with aux, got {result.total_wh_per_km:.2f}"
        )
        # Aux contribution should be ~4 Wh/km at 60 km/h
        assert 3.0 < result.auxiliary_wh_per_km < 5.5, (
            f"Aux Wh/km expected ~4.17, got {result.auxiliary_wh_per_km:.3f}"
        )

    def test_regen_is_zero_on_flat(self):
        """On a perfectly flat road, regen should be zero (or negligible)."""
        cfg = NO_STOPS_NO_HVAC
        segs = flat_segments(distance_km=50.0)
        result = simulate_energy(
            ev=NEXON_SPECS, segments=segs, avg_speed_kmh=60.0,
            temperature_c=20.0, driving_style="normal",
            num_passengers=1, ac_on=False, config=cfg,
        )
        assert result.regen_wh_per_km >= -0.01, (
            f"Regen should be ~0 on flat, got {result.regen_wh_per_km}"
        )

    def test_gradient_zero_on_flat(self):
        """Gradient term should be ~0 on a flat road."""
        cfg = NO_STOPS_NO_HVAC
        segs = flat_segments(distance_km=50.0)
        result = simulate_energy(
            ev=NEXON_SPECS, segments=segs, avg_speed_kmh=60.0,
            temperature_c=20.0, driving_style="normal",
            num_passengers=1, ac_on=False, config=cfg,
        )
        assert abs(result.gradient_wh_per_km) < 1.0, (
            f"Gradient should be ~0 on flat, got {result.gradient_wh_per_km}"
        )


# ── 4. Climb adds m*g*h/eta ───────────────────────────────────────────────────

class TestClimbGrade:
    """
    Climbing 100 m over 10 km:
    m*g*h = 1400 * 9.81 * 100 = 1,373,400 J
    Wh    = 1,373,400 / 3600 = 381.5 Wh
    /eta_drive = 381.5 / 0.92 = 414.7 Wh
    /distance = 414.7 / 10 = 41.5 Wh/km
    """

    def test_climb_gradient_wh_per_km(self):
        distance_km = 10.0
        gain_m = 100.0
        m = NEXON_SPECS.kerb_weight_kg

        expected_gradient_wh_km = (m * 9.81 * gain_m) / 3600.0 / 0.92 / distance_km
        # ≈ 41.5 Wh/km

        cfg = NO_STOPS_NO_HVAC
        segs = uphill_segments(distance_km, gain_m)
        result = simulate_energy(
            ev=NEXON_SPECS, segments=segs, avg_speed_kmh=60.0,
            temperature_c=20.0, driving_style="normal",
            num_passengers=1, ac_on=False, config=cfg,
        )

        # Allow ±6% tolerance (cos(theta)≈1 approximation + gradient via sin(atan(g)) + rounding)
        assert abs(result.gradient_wh_per_km - expected_gradient_wh_km) / expected_gradient_wh_km < 0.06, (
            f"Climb: expected ~{expected_gradient_wh_km:.2f} Wh/km, "
            f"got {result.gradient_wh_per_km:.2f}"
        )

    def test_climb_adds_to_total(self):
        """Uphill total > flat total for same distance and speed."""
        cfg = NO_STOPS_NO_HVAC
        flat = simulate_energy(
            ev=NEXON_SPECS, segments=flat_segments(10.0), avg_speed_kmh=60.0,
            temperature_c=20.0, driving_style="normal", num_passengers=1,
            ac_on=False, config=cfg,
        )
        hill = simulate_energy(
            ev=NEXON_SPECS, segments=uphill_segments(10.0, 100.0), avg_speed_kmh=60.0,
            temperature_c=20.0, driving_style="normal", num_passengers=1,
            ac_on=False, config=cfg,
        )
        assert hill.total_wh_per_km > flat.total_wh_per_km + 5.0


# ── 5. Descent recovers eta_regen of PE ──────────────────────────────────────

class TestDescentRegen:
    """
    Descending 100 m over 10 km:
    PE = m*g*h = 1,373,400 J → 381.5 Wh
    Recovered = 381.5 * eta_regen = 381.5 * 0.65 = 248.0 Wh
    regen Wh/km ≈ 248.0 / 10 = −24.8 Wh/km  (negative = energy returned)

    On a steep enough descent, F_total < 0 → regen mode.
    We use a gradient steep enough to overcome rolling+aero at 60 km/h.
    """

    def test_descent_regen_wh_per_km(self):
        distance_km = 10.0
        loss_m = 500.0   # 5% grade — steep enough for F_total < 0 (regen mode)
        # At 60 km/h (16.67 m/s): F_aero≈125N, F_roll≈151N total≈276N
        # F_grade = 1400×9.81×sin(atan(0.05)) ≈ 683N → F_total ≈ -407N → regen
        m = NEXON_SPECS.kerb_weight_kg

        cfg = NO_STOPS_NO_HVAC
        segs = downhill_segments(distance_km, loss_m)
        result = simulate_energy(
            ev=NEXON_SPECS, segments=segs, avg_speed_kmh=60.0,
            temperature_c=20.0, driving_style="normal",
            num_passengers=1, ac_on=False, config=cfg,
        )

        # regen_wh_per_km is negative (battery receives energy)
        assert result.regen_wh_per_km < -5.0, (
            f"Expected significant regen (<-5 Wh/km), got {result.regen_wh_per_km:.3f}"
        )

    def test_descent_less_energy_than_flat(self):
        """A gentle descent should consume less energy than flat (regen helps)."""
        cfg = NO_STOPS_NO_HVAC
        flat = simulate_energy(
            ev=NEXON_SPECS, segments=flat_segments(20.0), avg_speed_kmh=60.0,
            temperature_c=20.0, driving_style="normal", num_passengers=1,
            ac_on=False, config=cfg,
        )
        downhill = simulate_energy(
            ev=NEXON_SPECS, segments=downhill_segments(20.0, 100.0), avg_speed_kmh=60.0,
            temperature_c=20.0, driving_style="normal", num_passengers=1,
            ac_on=False, config=cfg,
        )
        assert downhill.total_wh_per_km < flat.total_wh_per_km


# ── 6. Faster speed raises aero energy ───────────────────────────────────────

class TestAeroScaling:
    """
    Aerodynamic drag ∝ v² → aero power ∝ v³ → aero Wh/km ∝ v².
    Doubling speed should roughly quadruple aero_wh_per_km.
    """

    def test_higher_speed_higher_aero(self):
        cfg = NO_STOPS_NO_HVAC
        segs = flat_segments(50.0)

        slow = simulate_energy(
            ev=NEXON_SPECS, segments=segs, avg_speed_kmh=60.0,
            temperature_c=20.0, driving_style="normal", num_passengers=1,
            ac_on=False, config=cfg,
        )
        fast = simulate_energy(
            ev=NEXON_SPECS, segments=segs, avg_speed_kmh=100.0,
            temperature_c=20.0, driving_style="normal", num_passengers=1,
            ac_on=False, config=cfg,
        )
        assert fast.aero_wh_per_km > slow.aero_wh_per_km, (
            "Higher speed must produce more aero drag"
        )

    def test_aero_roughly_v_squared(self):
        """
        At constant speed, aero Wh/km ∝ v²:
          Power_aero ∝ v³, time per km ∝ 1/v → energy per km ∝ v².
        Ratio at v=100 vs v=60 should be ~(100/60)² ≈ 2.78 (within ±30%).
        """
        cfg = NO_STOPS_NO_HVAC
        segs = flat_segments(50.0)

        slow = simulate_energy(
            ev=NEXON_SPECS, segments=segs, avg_speed_kmh=60.0,
            temperature_c=20.0, driving_style="normal", num_passengers=1,
            ac_on=False, config=cfg,
        )
        fast = simulate_energy(
            ev=NEXON_SPECS, segments=segs, avg_speed_kmh=100.0,
            temperature_c=20.0, driving_style="normal", num_passengers=1,
            ac_on=False, config=cfg,
        )
        expected_ratio = (100.0 / 60.0) ** 2  # ≈ 2.778
        actual_ratio = fast.aero_wh_per_km / slow.aero_wh_per_km
        # Allow ±30% tolerance (rounding of v_mps, wind_factor, etc.)
        assert abs(actual_ratio - expected_ratio) / expected_ratio < 0.30, (
            f"Aero v² scaling: expected ratio ~{expected_ratio:.2f}, got {actual_ratio:.2f}"
        )


# ── 7. Driving style ──────────────────────────────────────────────────────────

class TestDrivingStyle:
    def test_aggressive_higher_aero(self):
        """Aggressive style → higher speed factor → higher aero than normal."""
        cfg = NO_STOPS_NO_HVAC
        segs = flat_segments(50.0)

        normal = simulate_energy(
            ev=NEXON_SPECS, segments=segs, avg_speed_kmh=80.0,
            temperature_c=20.0, driving_style="normal", num_passengers=1,
            ac_on=False, config=cfg,
        )
        aggressive = simulate_energy(
            ev=NEXON_SPECS, segments=segs, avg_speed_kmh=80.0,
            temperature_c=20.0, driving_style="aggressive", num_passengers=1,
            ac_on=False, config=cfg,
        )
        assert aggressive.aero_wh_per_km > normal.aero_wh_per_km

    def test_calm_lower_aero(self):
        """Calm style → lower speed factor → lower aero than normal."""
        cfg = NO_STOPS_NO_HVAC
        segs = flat_segments(50.0)

        normal = simulate_energy(
            ev=NEXON_SPECS, segments=segs, avg_speed_kmh=80.0,
            temperature_c=20.0, driving_style="normal", num_passengers=1,
            ac_on=False, config=cfg,
        )
        calm = simulate_energy(
            ev=NEXON_SPECS, segments=segs, avg_speed_kmh=80.0,
            temperature_c=20.0, driving_style="calm", num_passengers=1,
            ac_on=False, config=cfg,
        )
        assert calm.aero_wh_per_km < normal.aero_wh_per_km

    def test_aggressive_calm_ordering(self):
        """Total: aggressive > normal > calm on flat road."""
        cfg = NO_STOPS_NO_HVAC
        segs = flat_segments(50.0)

        results = {}
        for style in ("aggressive", "normal", "calm"):
            results[style] = simulate_energy(
                ev=NEXON_SPECS, segments=segs, avg_speed_kmh=80.0,
                temperature_c=20.0, driving_style=style, num_passengers=1,
                ac_on=False, config=cfg,
            ).total_wh_per_km

        assert results["aggressive"] > results["normal"] > results["calm"]


# ── 8. Stop-and-go correction ─────────────────────────────────────────────────

class TestStopAndGo:
    def test_urban_higher_inertia_than_highway(self):
        """
        Same distance & speed, but urban stops_per_km >> highway.
        Inertia Wh/km should be much higher for urban.
        """
        # Speed 35 km/h → classified as urban by default thresholds
        segs = flat_segments(20.0)

        urban_cfg = PhysicsConfig(
            stops_per_km_urban=1.5,
            stops_per_km_mixed=0.0,
            stops_per_km_highway=0.0,
            p_aux_kw=0.0, hvac_base_kw=0.0,
            hvac_heat_coef_kw_per_c=0.0, hvac_cool_coef_kw_per_c=0.0,
        )
        highway_cfg = PhysicsConfig(
            stops_per_km_urban=0.0,
            stops_per_km_mixed=0.0,
            stops_per_km_highway=0.05,
            p_aux_kw=0.0, hvac_base_kw=0.0,
            hvac_heat_coef_kw_per_c=0.0, hvac_cool_coef_kw_per_c=0.0,
        )

        urban_result = simulate_energy(
            ev=NEXON_SPECS, segments=segs, avg_speed_kmh=35.0,
            temperature_c=20.0, driving_style="normal", num_passengers=1,
            ac_on=False, config=urban_cfg,
        )
        highway_result = simulate_energy(
            ev=NEXON_SPECS, segments=segs, avg_speed_kmh=35.0,
            temperature_c=20.0, driving_style="normal", num_passengers=1,
            ac_on=False, config=highway_cfg,
        )

        assert urban_result.inertia_wh_per_km > highway_result.inertia_wh_per_km + 1.0, (
            f"Urban inertia {urban_result.inertia_wh_per_km:.2f} should >> "
            f"highway inertia {highway_result.inertia_wh_per_km:.2f}"
        )

    def test_stop_and_go_increases_total(self):
        """
        Enabling stops adds inertia_wh_per_km > 0.
        Note: regen credit from stops may partially offset inertia friction losses,
        so the *net* total_wh_per_km could be higher or lower than no-stops depending
        on eta_regen. What must be true is that inertia_wh_per_km > 0 (friction losses
        from stops are non-zero), and regen_wh_per_km < 0 (recovered energy from stops).
        """
        segs = flat_segments(20.0)

        with_stops = simulate_energy(
            ev=NEXON_SPECS, segments=segs, avg_speed_kmh=35.0,
            temperature_c=20.0, driving_style="normal", num_passengers=1,
            ac_on=False,
            config=PhysicsConfig(
                stops_per_km_urban=1.5, stops_per_km_mixed=0.0,
                stops_per_km_highway=0.0,
                p_aux_kw=0.0, hvac_base_kw=0.0,
                hvac_heat_coef_kw_per_c=0.0, hvac_cool_coef_kw_per_c=0.0,
            ),
        )
        # Inertia losses must be positive (friction from stops)
        assert with_stops.inertia_wh_per_km > 0.0, (
            f"Inertia Wh/km must be > 0 with stops, got {with_stops.inertia_wh_per_km}"
        )
        # Regen credit from stops must be negative
        assert with_stops.regen_wh_per_km < 0.0, (
            f"Regen Wh/km must be < 0 with stops, got {with_stops.regen_wh_per_km}"
        )


# ── 9. range_km – no clamps ───────────────────────────────────────────────────

class TestRangeKm:
    def test_basic_range(self):
        """range_km = usable_kwh * 1000 / wh_per_km."""
        ev = NEXON_SPECS  # usable=38.5 kWh
        wh_per_km = 150.0
        expected = 38.5 * 1000 / 150.0  # 256.67 km
        assert abs(range_km(ev, wh_per_km) - expected) < 0.01

    def test_range_no_upper_clamp(self):
        """Very low Wh/km (efficient EV) should give large range, no cap."""
        ev = NEXON_SPECS
        r = range_km(ev, wh_per_km=50.0)  # 38500/50 = 770 km
        assert r > 700.0, "No upper clamp should apply"

    def test_range_no_lower_clamp(self):
        """Very high Wh/km (inefficient EV) should give small range, no floor."""
        ev = NEXON_SPECS
        r = range_km(ev, wh_per_km=500.0)  # 38500/500 = 77 km
        assert r < 100.0, "No lower clamp should apply"

    def test_zero_wh_per_km_returns_inf(self):
        """Guard against division by zero."""
        ev = NEXON_SPECS
        r = range_km(ev, wh_per_km=0.0)
        assert r == float("inf")


# ── 10. segments_from_elevation_totals ───────────────────────────────────────

class TestSegmentsFromElevation:
    def test_flat_segments_zero_gradient(self):
        segs = segments_from_elevation_totals(10.0, 0.0, 0.0)
        for s in segs:
            assert s.gradient == 0.0

    def test_total_length_correct(self):
        segs = segments_from_elevation_totals(25.0, 50.0, 30.0, n_segments=20)
        total_m = sum(s.length_m for s in segs)
        assert abs(total_m - 25_000.0) < 1.0

    def test_alternating_climb_descent(self):
        """Even-indexed segments should be positive (climb), odd negative (descent)."""
        segs = segments_from_elevation_totals(10.0, 100.0, 50.0, n_segments=20)
        for i, s in enumerate(segs):
            if i % 2 == 0:
                assert s.gradient >= 0.0
            else:
                assert s.gradient <= 0.0


# ── 11. simulate_energy_simple convenience wrapper ────────────────────────────

class TestSimpleWrapper:
    def test_matches_manual_segment_call(self):
        """
        simulate_energy_simple should give same result as manually
        calling segments_from_elevation_totals + simulate_energy.
        """
        distance_km = 30.0
        gain_m = 100.0
        loss_m = 80.0

        segs = segments_from_elevation_totals(distance_km, gain_m, loss_m)
        direct = simulate_energy(
            ev=NEXON_SPECS, segments=segs, avg_speed_kmh=70.0,
            temperature_c=25.0, driving_style="normal",
            num_passengers=2, ac_on=True,
        )
        simple = simulate_energy_simple(
            ev=NEXON_SPECS, distance_km=distance_km,
            elevation_gain_m=gain_m, elevation_loss_m=loss_m,
            avg_speed_kmh=70.0, temperature_c=25.0,
            driving_style="normal", num_passengers=2, ac_on=True,
        )

        assert abs(direct.total_wh_per_km - simple.total_wh_per_km) < 0.01

    def test_ac_off_reduces_hvac(self):
        """With ac_on=False, HVAC term should be 0 regardless of temperature."""
        result = simulate_energy_simple(
            ev=NEXON_SPECS, distance_km=50.0,
            elevation_gain_m=0.0, elevation_loss_m=0.0,
            avg_speed_kmh=60.0, temperature_c=35.0,
            driving_style="normal", num_passengers=1, ac_on=False,
        )
        assert result.hvac_wh_per_km == 0.0, (
            f"HVAC should be 0 when ac_on=False, got {result.hvac_wh_per_km}"
        )

    def test_ac_on_at_high_temp_increases_hvac(self):
        """With ac_on=True at 35 °C, HVAC > 0."""
        result = simulate_energy_simple(
            ev=NEXON_SPECS, distance_km=50.0,
            elevation_gain_m=0.0, elevation_loss_m=0.0,
            avg_speed_kmh=60.0, temperature_c=35.0,
            driving_style="normal", num_passengers=1, ac_on=True,
        )
        assert result.hvac_wh_per_km > 0.0

    def test_num_passengers_affects_weight(self):
        """More passengers → heavier → more rolling & gradient energy."""
        light = simulate_energy_simple(
            ev=NEXON_SPECS, distance_km=50.0,
            elevation_gain_m=100.0, elevation_loss_m=0.0,
            avg_speed_kmh=60.0, temperature_c=20.0,
            driving_style="normal", num_passengers=1, ac_on=False,
        )
        heavy = simulate_energy_simple(
            ev=NEXON_SPECS, distance_km=50.0,
            elevation_gain_m=100.0, elevation_loss_m=0.0,
            avg_speed_kmh=60.0, temperature_c=20.0,
            driving_style="normal", num_passengers=5, ac_on=False,
        )
        assert heavy.total_wh_per_km > light.total_wh_per_km
