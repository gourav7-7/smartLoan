"""
============================================================================
api/main.py  —  FastAPI Model Serving + Dashboard UI
============================================================================
Updated for UCI Default of Credit Card Clients dataset.
All form fields, MLflow metric columns, and response schemas reflect the
new pipeline (composite scoring, optimal threshold, calibrated models).
============================================================================
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from contextlib import asynccontextmanager
import time
import os
import mlflow
import mlflow.sklearn

from smartLoan.schemas.request_schema import (
    CreditCardApplication,
    PredictionResponse,
    BatchPredictionResponse,
    HealthResponse,
)
from smartLoan.pipeline.prediction_pipeline import ModelPredictionStage
from smartLoan.utils.logger import logger
from smartLoan.config.settings import settings
from smartLoan.api.ui import DASHBOARD_HTML

# Read from .env via settings — no os.getenv needed
MLFLOW_URI        = settings.MLFLOW_TRACKING_URI
MLFLOW_EXPERIMENT = settings.MLFLOW_EXPERIMENT_NAME

mlflow.set_tracking_uri(MLFLOW_URI)

pipeline: ModelPredictionStage = None
_training_status = {"running": False, "message": "Idle"}


@asynccontextmanager
async def lifespan(app: FastAPI):
    global pipeline
    logger.info("SmartLoan API starting up...")
    try:
        pipeline = ModelPredictionStage()
        logger.info(f"Model ready: {pipeline.predictor._display_name} | "
                    f"Threshold: {pipeline.predictor.threshold:.4f}")
    except Exception as e:
        logger.error(f"Model failed to load: {e}")
        logger.warning("API running without a model. Run the training pipeline first, then restart.")
        pipeline = None
    yield
    logger.info("SmartLoan API shutting down...")


app = FastAPI(
    title="SmartLoan Credit Risk API",
    description=(
        "Production ML API for UCI Credit Card Default prediction. "
        "Uses composite-scored model selection, calibrated probabilities, "
        "and optimal F2-based decision thresholds."
    ),
    version=settings.APP_VERSION,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── UI ─────────────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse, tags=["UI"])
def serve_ui():
    """Serves the SmartLoan dashboard UI."""
    return HTMLResponse(content=DASHBOARD_HTML)


# ── System: Health ─────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["System"])
def health_check():
    """Returns API and model status."""
    if pipeline is None:
        return HealthResponse(
            status="no_model",
            model_loaded=False,
            model_name="none",
            threshold=0.0
        )
    return HealthResponse(
        status="ok",
        model_loaded=True,
        model_name=pipeline.predictor._display_name,
        threshold=round(pipeline.predictor.threshold, 4),
    )


# ── MLflow: Experiment Runs ────────────────────────────────────────────────────

@app.get("/api/mlruns", tags=["MLflow"])
def get_mlruns():
    """
    Fetch all MLflow runs from the UCI Credit Default experiment.
    Returns all metrics logged by the updated ModelTrainer:
    accuracy, precision, recall, f1, f2, roc_auc, pr_auc, composite_score, threshold.
    """
    try:
        experiment = mlflow.get_experiment_by_name(MLFLOW_EXPERIMENT)
        if experiment is None:
            return {"runs": [], "message": "No experiment found. Run the training pipeline first."}

        runs_df = mlflow.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=["metrics.composite_score DESC"],
        )

        if runs_df.empty:
            return {"runs": [], "message": "No runs found. Run the training pipeline first."}

        def safe_get(row, key, default=0.0):
            val = row.get(f"metrics.{key}", default)
            return round(float(val), 4) if val is not None else default

        results = []
        for _, row in runs_df.iterrows():
            results.append({
                "model":           row.get("tags.mlflow.runName", "unknown"),
                "run_id":          row["run_id"],
                "accuracy":        safe_get(row, "accuracy"),
                "precision":       safe_get(row, "precision"),
                "recall":          safe_get(row, "recall"),
                "f1_score":        safe_get(row, "f1_score"),
                "f2_score":        safe_get(row, "f2_score"),
                "roc_auc":         safe_get(row, "roc_auc"),
                "pr_auc":          safe_get(row, "pr_auc"),
                "composite_score": safe_get(row, "composite_score"),
                "threshold":       safe_get(row, "threshold"),
                "cv_roc_auc_mean": safe_get(row, "cv_roc_auc_mean"),
                "cv_roc_auc_std":  safe_get(row, "cv_roc_auc_std"),
                "status":          row.get("status", "unknown"),
            })

        return {"runs": results, "total": len(results)}

    except Exception as e:
        logger.error(f"Failed to fetch MLflow runs: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ── Training ───────────────────────────────────────────────────────────────────

def _run_training():
    global _training_status
    try:
        _training_status = {"running": True, "message": "Training in progress..."}
        logger.info("Starting model training pipeline...")
        from smartLoan.pipeline.training_pipeline import TrainingPipeline
        pipeline = TrainingPipeline()
        pipeline.run_pipeline()
        _training_status = {"running": False, "message": "Training completed successfully!"}
        logger.info("Training pipeline completed.")
    except Exception as e:
        _training_status = {"running": False, "message": f"Training failed: {str(e)}"}
        logger.error(f"Training failed: {e}")


@app.post("/api/train", tags=["Training"])
def train_model(background_tasks: BackgroundTasks):
    """
    Trigger the full training pipeline in the background.
    Returns immediately; poll /api/train/status for progress.
    """
    if _training_status["running"]:
        return {"message": "Training already in progress.", "status": _training_status}
    background_tasks.add_task(_run_training)
    return {"message": "Training started in background.", "status": _training_status}


@app.get("/api/train/status", tags=["Training"])
def training_status():
    """Returns current training pipeline status."""
    return _training_status


# ── Prediction ─────────────────────────────────────────────────────────────────

@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict(application: CreditCardApplication):
    """
    Predict credit card default risk for a single client.

    Uses the best model selected by composite scoring (F2 + PR-AUC + CV-AUC)
    with calibrated probabilities and an optimal F2-maximising threshold.
    """
    if pipeline is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Run the training pipeline first, then restart the API."
        )
    start = time.time()
    try:
        input_data = application.model_dump()
        result = pipeline.main(input_data)
        elapsed = round(time.time() - start, 4)
        logger.info(f"Prediction completed in {elapsed}s → {result}")
        return PredictionResponse(**result)
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["Prediction"])
def predict_batch(applications: list[CreditCardApplication]):
    """
    Predict default risk for a batch of clients (max 100 per request).
    Failed individual records return error details without failing the whole batch.
    """
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")
    if not applications:
        raise HTTPException(status_code=400, detail="Request body cannot be empty.")
    if len(applications) > 100:
        raise HTTPException(
            status_code=400,
            detail=f"Batch limit is 100. You sent {len(applications)}."
        )

    start = time.time()
    results = []
    for i, app in enumerate(applications):
        try:
            results.append(pipeline.main(app.model_dump()))
        except Exception as e:
            logger.error(f"Batch item {i} failed: {e}")
            results.append({
                "prediction": None,
                "probability_of_default": None,
                "risk_label": "Error",
                "model_used": "none",
                "threshold_applied": None,
                "error": str(e),
            })

    return BatchPredictionResponse(
        count=len(results),
        predictions=results,
        elapsed_seconds=round(time.time() - start, 4),
    )


