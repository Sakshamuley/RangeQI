# RangeIQ — AI-Powered EV Range & Suitability Simulator

Predicts **real-world** EV driving range (not manufacturer lab claims) using physics-informed simulation + machine learning, with honest uncertainty bands.

## Quick Start

### Backend (Python / FastAPI)

```bash
cd backend
python -m venv venv
# Windows:
venv\Scripts\activate
# macOS/Linux:
# source venv/bin/activate

pip install -r requirements.txt
# If xgboost/shap downloads are slow, install core first:
# pip install fastapi uvicorn[standard] sqlalchemy pandas numpy scikit-learn joblib httpx pydantic pydantic-settings
# pip install xgboost shap

uvicorn app.main:app --reload --port 8000
# Or on Windows: .\run.ps1
```

On first startup, the backend automatically:
1. Seeds 19 Indian EV models into SQLite
2. Generates 8,000 synthetic training trips
3. Trains Linear Regression, Random Forest, XGBoost + quantile models
4. Runs real-world validation against 12 journalist test-drive cases

API docs: http://127.0.0.1:8000/docs

### Frontend (React / Vite)

```bash
cd frontend
npm install
npm run dev
```

Open http://localhost:5173

### Optional: Live Weather

Set `OPENWEATHER_API_KEY` in a `.env` file in `backend/` for live temperature data. Without it, season-average temperatures are used.

## Project Structure

```
backend/
  app/
    simulator.py      # Physics energy model (commented formulas)
    ml/train.py         # Model training + quantile regression
    ml/predict.py       # Inference + SHAP
    services/           # Route elevation, suitability, carbon
    api/routes.py       # REST endpoints
  data/
    ev_models_seed.csv
    training_data.csv   # Generated on startup
    benchmark_results.json
    validation_results.json
frontend/
  src/pages/            # Input, Results, Benchmark, History, About
```

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/predict` | Range prediction with p10/p50/p90 bands |
| GET | `/api/ev-models` | List all EV specs |
| GET | `/api/route-elevation` | Elevation profile for map |
| GET | `/api/benchmark` | ML model comparison metrics |
| GET | `/api/validation` | Real-world test-drive validation |
| GET | `/api/scan/{id}` | Retrieve saved prediction |

## Key Design Choices

- **Always show a range (p10–p90)**, never a single false-precision number
- **Manufacturer claimed range** shown alongside for contrast
- **Training data is physics-simulation-based** (labeled clearly in Methodology page)
- **Graceful API fallbacks** — flat terrain / season temps if external APIs fail
