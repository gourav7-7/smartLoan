# ============================================================================
# SmartLoan Credit Default Prediction — Dockerfile
# ============================================================================
# Python 3.10-slim matches the existing .pyc files (cpython-310).
# WORKDIR is /app because paths.py resolves ROOT_DIR via parents[3]:
#   /app/src/smartLoan/config/paths.py → parents[3] = /app
# All relative artifact paths (artifacts/, logs/, mlflow.db) land in /app.
# ============================================================================

FROM python:3.10-slim

# ── System dependencies ───────────────────────────────────────────────────────
# build-essential: required by lightgbm, xgboost C extensions
# git:             mlflow may need it for run metadata
# libgomp1:        OpenMP runtime needed by LightGBM
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        git \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# ── Working directory ─────────────────────────────────────────────────────────
# All relative paths in model_prediction.py (Path("artifacts/models")),
# logger.py (logs/), and mlflow (sqlite:///mlflow.db) resolve here.
WORKDIR /app

# ── Python package installation ───────────────────────────────────────────────
# Dependencies are installed before copying source so Docker can cache this
# layer — rebuilds after code-only changes skip the slow pip install step.
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
        # ── Web framework ────────────────────────────────────────────────────
        "fastapi==0.111.0" \
        "uvicorn[standard]==0.29.0" \
        "python-multipart==0.0.9" \
        # ── Data validation ───────────────────────────────────────────────────
        # pydantic v2 for schemas/request_schema.py (uses model_dump())
        # pydantic-settings for config/settings.py (uses BaseSettings)
        "pydantic==2.7.1" \
        "pydantic-settings==2.2.1" \
        # ── ML experiment tracking ────────────────────────────────────────────
        "mlflow==2.12.2" \
        # ── Core ML ───────────────────────────────────────────────────────────
        "scikit-learn==1.4.2" \
        "imbalanced-learn==0.12.2" \
        "lightgbm==4.3.0" \
        "xgboost==2.0.3" \
        # ── Data processing ───────────────────────────────────────────────────
        "pandas==2.2.2" \
        "numpy==1.26.4" \
        "joblib==1.4.0" \
        # ── Visualisation (model_evaluation.py) ──────────────────────────────
        "matplotlib==3.8.4" \
        "seaborn==0.13.2" \
        # ── Data ingestion (data_ingestion.py uses Kaggle CLI) ────────────────
        "kaggle==1.6.12"

# ── Copy source code ──────────────────────────────────────────────────────────
# Only src/ is copied — no notebooks, no .git, no local artifacts.
# .dockerignore should exclude: __pycache__, *.pyc, artifacts/, logs/, mlflow.db
COPY src/ ./src/

# ── Python path ───────────────────────────────────────────────────────────────
# No setup.py is present in the project, so we add src/ to PYTHONPATH
# so that `import smartLoan` resolves to /app/src/smartLoan.
ENV PYTHONPATH=/app/src

# ── Runtime directories ───────────────────────────────────────────────────────
# Create all directories that components write to at runtime.
# These are bind-mounted in production; pre-creating avoids permission errors
# if the container runs without a volume.
RUN mkdir -p \
        artifacts/raw \
        artifacts/processed \
        artifacts/models \
        artifacts/evaluation \
        logs

# ── Environment variables ─────────────────────────────────────────────────────
# MLflow uses SQLite inside the container by default.
# Override MLFLOW_TRACKING_URI in docker-compose or at runtime to point to
# a remote MLflow server for multi-user / persistent experiment tracking.
ENV MLFLOW_TRACKING_URI=sqlite:///mlflow.db

# Kaggle credentials — required by data_ingestion.py to download the dataset.
# DO NOT hardcode values here. Pass at runtime:
#   docker run -e KAGGLE_USERNAME=... -e KAGGLE_KEY=... smartloan
# Or via docker-compose environment section / .env file.
ENV KAGGLE_USERNAME=""
ENV KAGGLE_KEY=""

# Prevent Python from buffering stdout/stderr — ensures logs appear immediately.
ENV PYTHONUNBUFFERED=1

# ── Port ──────────────────────────────────────────────────────────────────────
EXPOSE 8000

# ── Entrypoint ────────────────────────────────────────────────────────────────
# smartLoan.api.main:app → /app/src/smartLoan/api/main.py → app object
# --host 0.0.0.0 binds to all interfaces (required inside Docker)
# --workers 1: single worker keeps MLflow SQLite safe from concurrent writes
CMD ["uvicorn", "smartLoan.api.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "1"]