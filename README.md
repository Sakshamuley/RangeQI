# RangeIQ — Honest EV Suitability Simulator

RangeIQ helps you answer: **"Can this EV actually complete my journey?"**

It does this with a segment-by-segment physics model (calibrated against journalist road tests),
a quantile-ML uncertainty estimator, and live route data — returning a **p10 / p50 / p90 range
band** per EV, never a false-precision single number.

> **What we don't claim.**
> EV range is noisy. RangeIQ shows calibrated estimates with honest uncertainty bands.
> The ML layer is trained on simulated data. Accuracy against reality is measured by
> the road-test validation (MAPE ≈ 9%), not assumed.

---

## Quick Start

### Backend (Python 3.11 / FastAPI)

```bash
cd backend
python -m venv venv

# Windows
venv\Scripts\activate
# macOS / Linux
# source venv/bin/activate

pip install -r requirements.txt

# Copy and edit the environment file (all keys are optional)
cp .env.example .env

uvicorn app.main:app --reload --port 8000
```

On first startup the backend automatically:

1. Creates `data/rangeiq.db` (SQLite) and seeds **16 Indian EV models**
2. Generates **4,800 synthetic training rows** (16 EVs × 300 trips × 40 MC samples → physics mean)
3. Trains **HistGradientBoosting quantile models** (p10 / p50 / p90 Wh/km)
4. Runs real-world validation against **16 Autocar India road-test cases**

Interactive docs: <http://127.0.0.1:8000/docs>

### Frontend (React 18 / Vite)

```bash
cd frontend
npm install        # or: pnpm install
npm run dev
```

Open <http://localhost:5173>

The Vite dev server proxies `/api` → `http://127.0.0.1:8000`, so no CORS config is needed
during development.

### Optional: Live Weather

Set `OPENWEATHER_API_KEY` in `backend/.env` for live temperature data.
Without it, season-average temperatures are used and a fallback warning is shown in the UI.

---

## How It Works

### 1. Physics model (segment-by-segment)

For each OSRM road segment the simulator computes:

```
F_total = F_aero + F_rolling + F_gradient + F_inertia
E_segment = F_total × length / (η_drive × 1000)   [kWh]
E_segment -= E_regen                                [regenerative recovery]
E_aux = (P_aux + P_hvac) × time                    [auxiliary + HVAC]
```

- **Air density** — ISA model at route altitude
- **HVAC** — per Yuksel & Michalek 2015; only active when `ac_on=True`
- **Stop-and-go correction** — inertia penalty per Ahn & Rakha 2016
- **Driving style** — speed/acceleration factors (calm / normal / aggressive)

Two parameters were calibrated against road-test data (Phase 4):

| Parameter | Calibrated value | Meaning |
|-----------|-----------------|---------|
| `p_aux_kw` | 1.0 kW | Auxiliary electrical load |
| `eta_regen` | 0.30 | Regenerative braking efficiency |

### 2. ML uncertainty estimator

- **Architecture**: `HistGradientBoostingRegressor` with quantile loss
- **Training data**: 4,800 rows, all **physics-simulated** (not real telematics)
- **Cross-validation**: GroupKFold(4) by EV model — prevents data leakage across vehicles
- **CV MAE (p50)**: ≈ 7.7 Wh/km on simulated data
- **Real-world accuracy**: see Validation below

### 3. Route data

| Source | Data | Fallback |
|--------|------|---------|
| OSRM (OpenStreetMap) | Road geometry, distance, duration | Straight-line haversine |
| Open-Elevation | Elevation profile → gradients | Flat road |
| OpenWeatherMap | Temperature at destination | Season average |
| Nominatim | Place name → lat/lon geocoding | — |

All fallbacks are surfaced as visible warnings in the UI — never silently substituted.

### 4. Suitability verdict

```
required_km = route_distance × (2 if round_trip) × (1 + reserve_pct / 100)

Suitable     if p10_range ≥ required_km   # worst-case still covers the trip
Marginal     if p50_range ≥ required_km   # expected covers it; worst-case doesn't
Not suitable if p50_range < required_km
```

---

## Validation

Compared against **16 Autocar India journalist road-test** results (28 Apr 2026 edition),
cross-checked via Gaadify.

| Metric | Value |
|--------|-------|
| Baseline MAPE (uncalibrated) | 25.1% |
| Calibrated MAPE | 9.2% |
| Test-set MAPE | 9.1% |
| Spearman ρ | 0.741 (p = 0.001) |

Validation scenario: 50% city (35 km/h) + 50% highway (80 km/h), flat, 28°C, 2 passengers, AC on.

---

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/api/recommend` | Ranked EV recommendations for a route |
| `POST` | `/api/predict` | Single-model range check |
| `GET`  | `/api/geocode` | Place name → lat/lon (Nominatim) |
| `GET`  | `/api/ev-models` | EV catalogue with optional filters |
| `GET`  | `/api/body-types` | Supported body-type values |
| `GET`  | `/api/route-elevation` | Road geometry + elevation + weather |
| `GET`  | `/api/benchmark` | ML model CV metrics |
| `GET`  | `/api/validation` | Road-test validation results |
| `GET`  | `/api/history` | Recent prediction scans |
| `GET`  | `/api/scan/{id}` | Single saved scan |

---

## Project Structure

```
backend/
  app/
    main.py               # FastAPI app + lifespan (seed → train → validate)
    models.py             # SQLAlchemy ORM (EVModel, ApiCache, PredictionScan)
    simulator.py          # Segment-by-segment physics (F=ma, ISA density, HVAC)
    physics_config.py     # PhysicsConfig dataclass + PARAM_CATALOGUE
    validation.py         # Road-test calibration + MAPE/Spearman reporting
    seed.py               # DB seeder with 10-rule validation
    ml/
      data_gen.py         # Monte Carlo synthetic training data
      train.py            # HistGBT quantile training (GroupKFold)
      predict.py          # QuantilePredictor with physics fallback
    services/
      route_service.py    # Unified route facade (OSRM + elevation + weather)
      osrm_service.py     # OSRM road geometry
      elevation_service.py# Open-Elevation batched POST + smoothing
      weather_service.py  # OpenWeatherMap + season fallback
      geocoding_service.py# Nominatim with 7-day cache
      recommend_service.py# Full recommendation pipeline
      cache.py            # SHA-256 keyed SQLite cache
    api/
      routes.py           # All REST handlers
  data/
    ev_models_seed.csv    # 16 Indian EVs (NULL prices — not yet published)
    validation_cases.csv  # 16 Autocar India road-test cases
    rangeiq.db            # SQLite (gitignored, auto-created)
    training_data.csv     # Generated (gitignored)
    benchmark_results.json# Generated (gitignored)
    validation_results.json# Generated (gitignored)
  tests/
    test_phase1.py … test_phase6.py   # 196 tests
  .env.example
  requirements.txt

frontend/
  src/
    pages/
      InputPage.jsx       # Geocoded search + filters → /api/recommend
      ResultsPage.jsx     # Verdict groups + range bands + breakdown
      AboutPage.jsx       # Methodology + live validation numbers
      BenchmarkPage.jsx   # ML CV metrics + road-test table
      HistoryPage.jsx     # Past scans
    components/
      VerdictBadge.jsx    # Suitable / Marginal / Not suitable pill
      RangeBand.jsx       # p10–p90 bar with required-km marker
      EnergyChart.jsx     # Horizontal bar chart (7 physics terms)
      CarbonCallout.jsx   # CO₂ + cost (labelled illustrative)
      RouteMap.jsx        # Leaflet map + elevation SVG
    api/client.js         # Axios wrappers for all endpoints

docs/
  ASSUMPTIONS.md          # Every physics parameter with source + type
```

---

## Data Honesty Policy

- **16 EV models** seeded from publicly available manufacturer spec sheets.
- **All ex-showroom prices are NULL** — not yet independently sourced. Price filters
  exclude NULL-price EVs and show a warning.
- **Drag coefficient and frontal area** are marked `assumed` when not officially published.
- **Training data is fully synthetic** — generated by the physics simulator. The ML
  layer is an emulator and uncertainty estimator, not a model trained on real telematics.
- **No invented citations.** References used:
  - Fiori et al., Appl. Energy (2016)
  - Ahn & Rakha, Appl. Energy 168 (2016)
  - Dollinger & Fischerauer, Energies 14(18) 5804 (2021)
  - Gillespie, *Fundamentals of Vehicle Dynamics* (1992)
  - Yuksel & Michalek, Environ. Sci. Technol. (2015)
  - ARAI AIS-040 (MIDC cycle definition)

---

## Running Tests

```bash
cd backend
# Windows (activate venv first)
venv\Scripts\activate
python -m pytest tests/ -v
```

196 tests across 6 phases. All external HTTP calls are mocked.

---

## Environment Variables

See `backend/.env.example` for all supported variables.

| Variable | Default | Purpose |
|----------|---------|---------|
| `OPENWEATHER_API_KEY` | — | Live weather; falls back to season average if absent |
| `FRONTEND_ORIGIN` | `http://localhost:5173` | CORS allowed origin |
| `DATABASE_URL` | `sqlite:///./data/rangeiq.db` | SQLAlchemy connection string |

---

## Development Notes

- **No authentication** — this is a local demo tool, not a multi-user service.
- **No real-time pricing** — all cost figures are illustrative (₹8/kWh, ₹100/L petrol).
- **CO₂ figures** use India CEA 2022 grid factor (0.72 kg CO₂/kWh) and IPCC combustion
  constant (2.31 kg CO₂/L petrol). Labelled "illustrative" everywhere in the UI.
- The `SuitabilityCard` and old `predict`-flow components are superseded by Phase 6–7
  but kept in git history.
