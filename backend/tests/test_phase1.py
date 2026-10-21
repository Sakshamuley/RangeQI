"""Phase 1 tests — EV database, seed loader, and catalogue endpoints."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# Put backend on path
BACKEND = Path(__file__).resolve().parent.parent
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from app.models import Base, EVModel, BODY_TYPES
from app.database import get_db
from app.api.routes import router
from app.seed import validate_row, seed_ev_models

# ── In-memory test DB ─────────────────────────────────────────────────────────

TEST_ENGINE = create_engine(
    "sqlite:///:memory:", connect_args={"check_same_thread": False}
)
TestingSession = sessionmaker(bind=TEST_ENGINE)


def override_get_db():
    db = TestingSession()
    try:
        yield db
    finally:
        db.close()


# Minimal FastAPI app — NO lifespan — so we control DB state entirely
_test_app = FastAPI()
_test_app.include_router(router, prefix="/api")
_test_app.dependency_overrides[get_db] = override_get_db


@pytest.fixture(autouse=True)
def fresh_tables():
    """Create all tables before each test, drop them after."""
    Base.metadata.create_all(bind=TEST_ENGINE)
    yield
    Base.metadata.drop_all(bind=TEST_ENGINE)


@pytest.fixture()
def db():
    """A SQLAlchemy session bound to the test engine."""
    session = TestingSession()
    try:
        yield session
    finally:
        session.close()


@pytest.fixture()
def client(db):
    """
    A TestClient whose DB dependency is overridden to use the *same*
    session as the `db` fixture so inserts are visible to the HTTP handler.
    """
    def _override():
        yield db

    _test_app.dependency_overrides[get_db] = _override
    with TestClient(_test_app, raise_server_exceptions=True) as c:
        yield c
    _test_app.dependency_overrides.clear()


# ── Helper: minimal valid CSV row ─────────────────────────────────────────────

def _row(**overrides) -> dict[str, str]:
    base = {
        "model_name": "Test EV",
        "variant": "",
        "body_type": "hatchback",
        "seats": "5",
        "ex_showroom_price_lakh": "",
        "price_date": "",
        "price_source_url": "",
        "price_assumed": "true",
        "claimed_range_km": "300",
        "claimed_cycle": "MIDC",
        "spec_source_url": "",
        "battery_capacity_kwh": "30",
        "usable_battery_kwh": "",
        "usable_battery_assumed": "true",
        "kerb_weight_kg": "1300",
        "drag_coefficient": "0.30",
        "drag_coefficient_assumed": "true",
        "frontal_area_m2": "2.30",
        "frontal_area_assumed": "true",
        "motor_power_kw": "80",
        "drivetrain_eff_assumed": "true",
        "regen_eff_assumed": "true",
    }
    base.update(overrides)
    return base


# ── Helper: insert EVModel directly ──────────────────────────────────────────

def _ev(db, **kwargs):
    defaults = dict(
        model_name="Test Hatch",
        body_type="hatchback",
        seats=5,
        price_assumed=True,
        claimed_range_km=300.0,
        claimed_cycle="MIDC",
        battery_capacity_kwh=30.0,
        usable_battery_kwh=28.5,
        usable_battery_assumed=True,
        kerb_weight_kg=1300.0,
        drag_coefficient=0.30,
        drag_coefficient_assumed=True,
        frontal_area_m2=2.30,
        frontal_area_assumed=True,
        motor_power_kw=80.0,
        drivetrain_eff_assumed=True,
        regen_eff_assumed=True,
    )
    defaults.update(kwargs)
    obj = EVModel(**defaults)
    db.add(obj)
    db.commit()
    db.refresh(obj)
    return obj


# ═══════════════════════════════════════════════════════════════════════════════
# Seed loader unit tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestValidateRow:
    def test_valid_row_accepted(self):
        result = validate_row(_row(), line_number=2)
        assert result is not None
        assert result["model_name"] == "Test EV"
        assert result["claimed_range_km"] == pytest.approx(300.0)

    def test_missing_model_name_rejected(self):
        assert validate_row(_row(model_name=""), line_number=2) is None

    def test_invalid_body_type_rejected(self):
        assert validate_row(_row(body_type="truck"), line_number=2) is None

    def test_invalid_claimed_cycle_rejected(self):
        assert validate_row(_row(claimed_cycle="NEDC"), line_number=2) is None

    def test_drag_below_minimum_rejected(self):
        assert validate_row(_row(drag_coefficient="0.01"), line_number=2) is None

    def test_drag_above_maximum_rejected(self):
        assert validate_row(_row(drag_coefficient="0.99"), line_number=2) is None

    def test_weight_below_minimum_rejected(self):
        assert validate_row(_row(kerb_weight_kg="200"), line_number=2) is None

    def test_missing_motor_power_rejected(self):
        assert validate_row(_row(motor_power_kw=""), line_number=2) is None

    def test_missing_battery_rejected(self):
        assert validate_row(_row(battery_capacity_kwh=""), line_number=2) is None

    def test_zero_claimed_range_rejected(self):
        assert validate_row(_row(claimed_range_km="0"), line_number=2) is None

    def test_seats_below_2_rejected(self):
        assert validate_row(_row(seats="1"), line_number=2) is None

    def test_seats_above_9_rejected(self):
        assert validate_row(_row(seats="10"), line_number=2) is None

    def test_null_price_accepted(self):
        result = validate_row(_row(ex_showroom_price_lakh=""), line_number=2)
        assert result is not None
        assert result["ex_showroom_price_lakh"] is None
        assert result["price_assumed"] is True

    def test_usable_battery_defaults_to_95_pct(self):
        result = validate_row(_row(usable_battery_kwh=""), line_number=2)
        assert result is not None
        assert result["usable_battery_kwh"] == pytest.approx(30.0 * 0.95, rel=1e-3)
        assert result["usable_battery_assumed"] is True

    def test_explicit_usable_battery_preserved(self):
        result = validate_row(
            _row(usable_battery_kwh="27.5", usable_battery_assumed="false"),
            line_number=2,
        )
        assert result is not None
        assert result["usable_battery_kwh"] == pytest.approx(27.5)
        assert result["usable_battery_assumed"] is False

    def test_cd_not_assumed_when_flag_false(self):
        result = validate_row(_row(drag_coefficient_assumed="false"), line_number=2)
        assert result is not None
        assert result["drag_coefficient_assumed"] is False


class TestSeedLoaderDuplicates:
    def test_duplicate_model_name_rejected(self, tmp_path, db):
        csv_header = (
            "model_name,variant,body_type,seats,ex_showroom_price_lakh,price_date,"
            "price_source_url,price_assumed,claimed_range_km,claimed_cycle,"
            "spec_source_url,battery_capacity_kwh,usable_battery_kwh,"
            "usable_battery_assumed,kerb_weight_kg,drag_coefficient,"
            "drag_coefficient_assumed,frontal_area_m2,frontal_area_assumed,"
            "motor_power_kw,drivetrain_eff_assumed,regen_eff_assumed\n"
        )
        data_row = (
            "Dup EV,,hatchback,5,,,,true,300,MIDC,,30,,true,"
            "1300,0.30,true,2.30,true,80,true,true\n"
        )
        csv_file = tmp_path / "seed.csv"
        csv_file.write_text(csv_header + data_row + data_row, encoding="utf-8")

        import app.seed as seed_mod
        original = seed_mod.SEED_CSV
        seed_mod.SEED_CSV = csv_file
        try:
            inserted = seed_ev_models(db)
        finally:
            seed_mod.SEED_CSV = original

        assert inserted == 1
        assert db.query(EVModel).count() == 1

    def test_already_seeded_skips(self, db):
        """If the table already has rows, seed_ev_models returns 0."""
        _ev(db)
        result = seed_ev_models(db)
        assert result == 0


# ═══════════════════════════════════════════════════════════════════════════════
# API endpoint tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestBodyTypesEndpoint:
    def test_returns_all_body_types(self, client):
        resp = client.get("/api/body-types")
        assert resp.status_code == 200
        bt = resp.json()["body_types"]
        for expected in BODY_TYPES:
            assert expected in bt

    def test_no_duplicates(self, client):
        resp = client.get("/api/body-types")
        bt = resp.json()["body_types"]
        assert len(bt) == len(set(bt))


class TestEVModelsEndpoint:
    def test_list_all_returns_seeded_rows(self, client, db):
        _ev(db, model_name="EV A", body_type="hatchback")
        _ev(db, model_name="EV B", body_type="compact_suv")
        resp = client.get("/api/ev-models")
        assert resp.status_code == 200
        names = [e["model_name"] for e in resp.json()]
        assert "EV A" in names
        assert "EV B" in names

    def test_filter_body_type(self, client, db):
        _ev(db, model_name="Hatch 1", body_type="hatchback")
        _ev(db, model_name="SUV 1", body_type="compact_suv")
        resp = client.get("/api/ev-models?body_type=hatchback")
        assert resp.status_code == 200
        data = resp.json()
        assert all(e["body_type"] == "hatchback" for e in data)
        assert len(data) == 1

    def test_invalid_body_type_returns_400(self, client):
        resp = client.get("/api/ev-models?body_type=bus")
        assert resp.status_code == 400

    def test_filter_seats_min(self, client, db):
        _ev(db, model_name="4-seater", body_type="hatchback", seats=4)
        _ev(db, model_name="5-seater", body_type="hatchback", seats=5)
        _ev(db, model_name="6-seater", body_type="mpv", seats=6)
        resp = client.get("/api/ev-models?seats_min=5")
        assert resp.status_code == 200
        data = resp.json()
        assert all(e["seats"] >= 5 for e in data)
        assert len(data) == 2

    def test_price_filter_excludes_null_prices(self, client, db):
        _ev(db, model_name="Priced EV", body_type="hatchback",
            ex_showroom_price_lakh=12.0, price_assumed=False)
        _ev(db, model_name="No-price EV", body_type="hatchback",
            ex_showroom_price_lakh=None, price_assumed=True)
        resp = client.get("/api/ev-models?price_min=10&price_max=20")
        assert resp.status_code == 200
        names = [e["model_name"] for e in resp.json()]
        assert "Priced EV" in names
        assert "No-price EV" not in names

    def test_price_filter_range(self, client, db):
        _ev(db, model_name="Cheap EV", body_type="hatchback",
            ex_showroom_price_lakh=8.0, price_assumed=False)
        _ev(db, model_name="Mid EV", body_type="hatchback",
            ex_showroom_price_lakh=15.0, price_assumed=False)
        _ev(db, model_name="Expensive EV", body_type="suv",
            ex_showroom_price_lakh=35.0, price_assumed=False)
        resp = client.get("/api/ev-models?price_min=10&price_max=20")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 1
        assert data[0]["model_name"] == "Mid EV"

    def test_combined_body_type_and_seats_filter(self, client, db):
        _ev(db, model_name="Compact 5", body_type="compact_suv", seats=5)
        _ev(db, model_name="MPV 6", body_type="mpv", seats=6)
        _ev(db, model_name="Hatch 5", body_type="hatchback", seats=5)
        resp = client.get("/api/ev-models?body_type=compact_suv&seats_min=5")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 1
        assert data[0]["model_name"] == "Compact 5"

    def test_ev_model_has_assumed_flags(self, client, db):
        _ev(db, drag_coefficient_assumed=True, frontal_area_assumed=True)
        resp = client.get("/api/ev-models")
        assert resp.status_code == 200
        ev = resp.json()[0]
        assert "drag_coefficient_assumed" in ev
        assert "usable_battery_assumed" in ev
        assert "claimed_cycle" in ev
