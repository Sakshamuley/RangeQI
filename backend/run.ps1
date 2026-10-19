# RangeIQ backend launcher (Windows)
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

if (-not (Test-Path ".\venv\Scripts\python.exe")) {
    python -m venv venv
}

$py = ".\venv\Scripts\python.exe"
$pip = ".\venv\Scripts\pip.exe"

# Core deps first (fast); xgboost/shap are large — install if missing
& $pip install fastapi "uvicorn[standard]" sqlalchemy pandas numpy scikit-learn joblib httpx python-dotenv pydantic pydantic-settings
& $pip install xgboost shap 2>$null

& $py -m uvicorn app.main:app --reload --port 8000
