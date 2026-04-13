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

MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
# MLflow experiment name must match model_training.py
MLFLOW_EXPERIMENT = "SmartLoan_UCI_CreditDefault"

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
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
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
        from smartLoan.pipeline.training_pipeline import ModelTrainingStage
        stage = ModelTrainingStage()
        stage.main()
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


# ── Dashboard HTML ──────────────────────────────────────────────────────────────

DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>SmartLoan Risk Dashboard</title>
  <style>
    *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

    body {
      font-family: 'Segoe UI', system-ui, sans-serif;
      background: #0f1117;
      color: #e2e8f0;
      min-height: 100vh;
    }

    /* ── Header ── */
    header {
      background: linear-gradient(135deg, #1a1f2e 0%, #16213e 100%);
      border-bottom: 1px solid #2d3748;
      padding: 20px 32px;
      display: flex;
      align-items: center;
      gap: 16px;
    }
    .logo {
      width: 44px; height: 44px;
      background: linear-gradient(135deg, #6366f1, #8b5cf6);
      border-radius: 12px;
      display: flex; align-items: center; justify-content: center;
      font-size: 22px;
    }
    header h1 { font-size: 1.5rem; font-weight: 700; color: #f1f5f9; }
    header p  { font-size: 0.8rem; color: #94a3b8; margin-top: 2px; }

    /* ── Main layout ── */
    main { max-width: 1200px; margin: 0 auto; padding: 40px 24px; }

    /* ── Button Row ── */
    .btn-row {
      display: grid;
      grid-template-columns: repeat(2, 1fr);
      gap: 20px;
      margin-bottom: 40px;
    }
    .action-btn {
      border: none; border-radius: 16px; padding: 28px 20px;
      cursor: pointer; transition: all 0.22s ease;
      display: flex; flex-direction: column; align-items: center; gap: 12px;
      font-family: inherit; font-size: 0.95rem; font-weight: 600;
      position: relative; overflow: hidden;
    }
    .action-btn::after {
      content: ''; position: absolute; inset: 0;
      background: rgba(255,255,255,0.08);
      opacity: 0; transition: opacity 0.2s;
    }
    .action-btn:hover::after { opacity: 1; }
    .action-btn:active { transform: scale(0.97); }
    .action-btn .icon {
      width: 52px; height: 52px; border-radius: 14px;
      display: flex; align-items: center; justify-content: center;
      font-size: 26px;
    }
    .action-btn .label { color: #f1f5f9; }
    .action-btn .sub   { font-size: 0.76rem; font-weight: 400; color: rgba(255,255,255,0.6); }
    .btn-runs  { background: linear-gradient(135deg, #1e293b, #0f172a); border: 1px solid #334155; }
    .btn-runs  .icon { background: rgba(99,102,241,0.2); }
    .btn-pred  { background: linear-gradient(135deg, #1e1a2e, #0f0f1f); border: 1px solid #3d2d4a; }
    .btn-pred  .icon { background: rgba(168,85,247,0.2); }

    /* ── Panel ── */
    .panel {
      background: #1a1f2e; border: 1px solid #2d3748;
      border-radius: 16px; padding: 28px; display: none;
    }
    .panel.active { display: block; }
    .panel h2 { font-size: 1.1rem; color: #94a3b8; margin-bottom: 20px; }

    /* ── Loader ── */
    .loader {
      display: flex; flex-direction: column; align-items: center;
      gap: 16px; padding: 40px 0; color: #64748b;
    }
    .spinner {
      width: 36px; height: 36px;
      border: 3px solid #334155; border-top-color: #6366f1;
      border-radius: 50%; animation: spin 0.8s linear infinite;
    }
    @keyframes spin { to { transform: rotate(360deg); } }

    /* ── Table ── */
    .tbl-wrap { overflow-x: auto; }
    table { width: 100%; border-collapse: collapse; font-size: 0.82rem; }
    thead tr { background: #0f1117; }
    th {
      padding: 10px 12px; text-align: left; color: #64748b;
      font-weight: 600; font-size: 0.72rem; text-transform: uppercase;
      letter-spacing: 0.05em; border-bottom: 1px solid #2d3748; white-space: nowrap;
    }
    td { padding: 11px 12px; border-bottom: 1px solid #1e293b; white-space: nowrap; }
    tr:last-child td { border-bottom: none; }
    tr:hover td { background: rgba(255,255,255,0.02); }

    .badge {
      display: inline-block; padding: 3px 10px; border-radius: 99px;
      font-size: 0.72rem; font-weight: 600;
    }
    .badge-best   { background: rgba(34,197,94,0.15); color: #4ade80; }
    .badge-ok     { background: rgba(99,102,241,0.15); color: #818cf8; }

    .metric-pill {
      display: inline-block; padding: 2px 8px; border-radius: 6px;
      font-size: 0.78rem; font-weight: 600; font-variant-numeric: tabular-nums;
    }
    .pill-blue   { background: rgba(99,102,241,0.15);  color: #a5b4fc; }
    .pill-cyan   { background: rgba(6,182,212,0.15);   color: #67e8f9; }
    .pill-green  { background: rgba(16,185,129,0.15);  color: #34d399; }
    .pill-violet { background: rgba(139,92,246,0.15);  color: #c4b5fd; }
    .pill-amber  { background: rgba(245,158,11,0.15);  color: #fcd34d; }
    .pill-rose   { background: rgba(244,63,94,0.15);   color: #fda4af; }

    /* ── Section Divider ── */
    .form-section {
      font-size: 0.72rem; font-weight: 700; text-transform: uppercase;
      letter-spacing: 0.1em; color: #6366f1; margin: 24px 0 12px;
      padding-bottom: 6px; border-bottom: 1px solid #2d3748;
    }

    /* ── Prediction Form ── */
    .form-grid {
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
      gap: 14px;
    }
    .field label {
      display: block; font-size: 0.72rem; font-weight: 600;
      color: #94a3b8; margin-bottom: 6px;
      text-transform: uppercase; letter-spacing: 0.04em;
    }
    .field input, .field select {
      width: 100%; background: #0f1117; border: 1px solid #334155;
      border-radius: 10px; padding: 9px 12px; color: #e2e8f0;
      font-size: 0.88rem; font-family: inherit;
      transition: border-color 0.2s; outline: none;
    }
    .field input:focus, .field select:focus { border-color: #6366f1; }
    .field .hint { font-size: 0.68rem; color: #475569; margin-top: 3px; }

    .field.span2 { grid-column: span 2; }

    /* ── Profile Presets ── */
    .presets {
      display: flex; gap: 10px; margin-bottom: 20px; flex-wrap: wrap;
    }
    .preset-btn {
      border: 1px solid #334155; border-radius: 10px;
      padding: 8px 16px; cursor: pointer; font-family: inherit;
      font-size: 0.8rem; font-weight: 600; transition: all 0.2s;
      background: #0f1117; color: #94a3b8;
    }
    .preset-btn:hover { border-color: #6366f1; color: #a5b4fc; }

    /* ── Submit ── */
    .submit-row { margin-top: 24px; display: flex; gap: 12px; align-items: center; }
    .submit-btn {
      background: linear-gradient(135deg, #6366f1, #8b5cf6);
      color: #fff; border: none; border-radius: 12px;
      padding: 13px 28px; font-size: 0.92rem; font-weight: 600;
      cursor: pointer; font-family: inherit;
      transition: opacity 0.2s, transform 0.15s;
    }
    .submit-btn:hover  { opacity: 0.9; }
    .submit-btn:active { transform: scale(0.97); }
    .submit-btn:disabled { opacity: 0.5; cursor: not-allowed; }

    /* ── Prediction Result ── */
    .pred-result {
      margin-top: 24px; border-radius: 14px; padding: 24px 28px;
      border: 1px solid #2d3748; background: #0f1117;
    }
    .pred-result h3 {
      font-size: 0.75rem; color: #64748b; margin-bottom: 16px;
      text-transform: uppercase; letter-spacing: 0.05em;
    }
    .pred-cards {
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(170px, 1fr));
      gap: 14px;
    }
    .pred-card {
      background: #1a1f2e; border: 1px solid #2d3748;
      border-radius: 12px; padding: 16px 18px;
    }
    .pred-card .card-label {
      font-size: 0.68rem; color: #64748b; font-weight: 600;
      text-transform: uppercase; letter-spacing: 0.05em;
    }
    .pred-card .card-value { font-size: 1.35rem; font-weight: 700; margin-top: 6px; }

    .risk-low    { color: #4ade80; }
    .risk-medium { color: #fbbf24; }
    .risk-high   { color: #f87171; }
    .text-indigo { color: #818cf8; }
    .text-cyan   { color: #22d3ee; }
    .text-slate  { color: #94a3b8; }

    /* ── Probability Bar ── */
    .prob-bar-wrap { margin-top: 20px; }
    .prob-bar-label { font-size: 0.72rem; color: #64748b; margin-bottom: 6px; }
    .prob-bar {
      height: 10px; border-radius: 5px; background: #1e293b;
      overflow: hidden; position: relative;
    }
    .prob-bar-fill {
      height: 100%; border-radius: 5px;
      transition: width 0.8s ease;
    }
    .threshold-line {
      position: absolute; top: 0; bottom: 0; width: 2px;
      background: #f59e0b; opacity: 0.9;
    }

    /* ── Alerts ── */
    .error-box {
      background: rgba(239,68,68,0.1); border: 1px solid rgba(239,68,68,0.3);
      border-radius: 12px; padding: 16px 20px; color: #fca5a5; font-size: 0.88rem;
    }
    .msg-box {
      background: rgba(34,197,94,0.1); border: 1px solid rgba(34,197,94,0.3);
      border-radius: 12px; padding: 16px 20px; color: #86efac; font-size: 0.88rem;
    }

    @media (max-width: 640px) {
      .btn-row { grid-template-columns: 1fr; }
      .field.span2 { grid-column: span 1; }
    }
  </style>
</head>
<body>
  <header>
    <div class="logo">🏦</div>
    <div>
      <h1>SmartLoan Risk Dashboard</h1>
      <p>UCI Credit Card Default · Composite Model Selection · Calibrated Probabilities</p>
    </div>
  </header>

  <main>
    <!-- Action Buttons -->
    <div class="btn-row">
      <button class="action-btn btn-runs" onclick="showPanel('runs')">
        <div class="icon">📊</div>
        <span class="label">MLflow Model Comparison</span>
        <span class="sub">F2, PR-AUC, composite leaderboard</span>
      </button>
      <button class="action-btn btn-pred" onclick="showPanel('predict')">
        <div class="icon">🔮</div>
        <span class="label">Predict Default Risk</span>
        <span class="sub">Score a UCI credit card client</span>
      </button>
    </div>

    <!-- Panel: MLflow Runs -->
    <div class="panel" id="panel-runs">
      <h2>📊 Model Comparison — SmartLoan_UCI_CreditDefault</h2>
      <div id="runs-content">
        <div class="loader">
          <div class="spinner"></div>
          <span>Fetching runs from MLflow...</span>
        </div>
      </div>
    </div>

    <!-- Panel: Predict -->
    <div class="panel" id="panel-predict">
      <h2>🔮 Credit Card Default Risk Prediction</h2>

      <!-- Quick-fill presets -->
      <div class="presets">
        <span style="font-size:0.78rem;color:#64748b;align-self:center;">Quick fill:</span>
        <button class="preset-btn" onclick="fillPreset('low')">✅ Low Risk Profile</button>
        <button class="preset-btn" onclick="fillPreset('high')">⚠️ High Risk Profile</button>
        <button class="preset-btn" onclick="fillPreset('medium')">🟡 Medium Risk Profile</button>
      </div>

      <!-- DEMOGRAPHICS -->
      <div class="form-section">Client Demographics</div>
      <div class="form-grid">
        <div class="field">
          <label>LIMIT_BAL</label>
          <input type="number" id="f-limit" value="50000" min="10000" max="1000000" />
          <div class="hint">Credit limit in NT$ (10K – 1M)</div>
        </div>
        <div class="field">
          <label>SEX</label>
          <select id="f-sex">
            <option value="2">2 – Female</option>
            <option value="1">1 – Male</option>
          </select>
        </div>
        <div class="field">
          <label>EDUCATION</label>
          <select id="f-education">
            <option value="1">1 – Graduate school</option>
            <option value="2" selected>2 – University</option>
            <option value="3">3 – High school</option>
            <option value="4">4 – Others</option>
            <option value="0">0 – Unknown</option>
          </select>
        </div>
        <div class="field">
          <label>MARRIAGE</label>
          <select id="f-marriage">
            <option value="1">1 – Married</option>
            <option value="2" selected>2 – Single</option>
            <option value="3">3 – Others</option>
            <option value="0">0 – Unknown</option>
          </select>
        </div>
        <div class="field">
          <label>AGE</label>
          <input type="number" id="f-age" value="35" min="18" max="100" />
          <div class="hint">Age in years (18 – 100)</div>
        </div>
      </div>

      <!-- REPAYMENT STATUS -->
      <div class="form-section">Repayment Status (most recent 6 months) — -2=no use, -1=paid, 0=revolving, 1..9=months delayed</div>
      <div class="form-grid">
        <div class="field">
          <label>PAY_0 <span style="color:#6366f1;font-size:0.65rem;">SEP (latest)</span></label>
          <input type="number" id="f-pay0" value="-1" min="-2" max="9" />
        </div>
        <div class="field">
          <label>PAY_2 <span style="color:#475569;font-size:0.65rem;">AUG</span></label>
          <input type="number" id="f-pay2" value="-1" min="-2" max="9" />
        </div>
        <div class="field">
          <label>PAY_3 <span style="color:#475569;font-size:0.65rem;">JUL</span></label>
          <input type="number" id="f-pay3" value="-1" min="-2" max="9" />
        </div>
        <div class="field">
          <label>PAY_4 <span style="color:#475569;font-size:0.65rem;">JUN</span></label>
          <input type="number" id="f-pay4" value="-1" min="-2" max="9" />
        </div>
        <div class="field">
          <label>PAY_5 <span style="color:#475569;font-size:0.65rem;">MAY</span></label>
          <input type="number" id="f-pay5" value="-1" min="-2" max="9" />
        </div>
        <div class="field">
          <label>PAY_6 <span style="color:#475569;font-size:0.65rem;">APR (oldest)</span></label>
          <input type="number" id="f-pay6" value="-1" min="-2" max="9" />
        </div>
      </div>

      <!-- BILL AMOUNTS -->
      <div class="form-section">Bill Statement Amounts (NT$) — negative values allowed (credit balance)</div>
      <div class="form-grid">
        <div class="field">
          <label>BILL_AMT1 <span style="color:#6366f1;font-size:0.65rem;">SEP</span></label>
          <input type="number" id="f-bill1" value="20000" />
        </div>
        <div class="field">
          <label>BILL_AMT2 <span style="color:#475569;font-size:0.65rem;">AUG</span></label>
          <input type="number" id="f-bill2" value="19000" />
        </div>
        <div class="field">
          <label>BILL_AMT3 <span style="color:#475569;font-size:0.65rem;">JUL</span></label>
          <input type="number" id="f-bill3" value="18500" />
        </div>
        <div class="field">
          <label>BILL_AMT4 <span style="color:#475569;font-size:0.65rem;">JUN</span></label>
          <input type="number" id="f-bill4" value="17000" />
        </div>
        <div class="field">
          <label>BILL_AMT5 <span style="color:#475569;font-size:0.65rem;">MAY</span></label>
          <input type="number" id="f-bill5" value="16000" />
        </div>
        <div class="field">
          <label>BILL_AMT6 <span style="color:#475569;font-size:0.65rem;">APR</span></label>
          <input type="number" id="f-bill6" value="15500" />
        </div>
      </div>

      <!-- PAYMENT AMOUNTS -->
      <div class="form-section">Payment Amounts (NT$) — actual amount paid each month (≥ 0)</div>
      <div class="form-grid">
        <div class="field">
          <label>PAY_AMT1 <span style="color:#6366f1;font-size:0.65rem;">SEP</span></label>
          <input type="number" id="f-pamt1" value="19000" min="0" />
        </div>
        <div class="field">
          <label>PAY_AMT2 <span style="color:#475569;font-size:0.65rem;">AUG</span></label>
          <input type="number" id="f-pamt2" value="17000" min="0" />
        </div>
        <div class="field">
          <label>PAY_AMT3 <span style="color:#475569;font-size:0.65rem;">JUL</span></label>
          <input type="number" id="f-pamt3" value="18500" min="0" />
        </div>
        <div class="field">
          <label>PAY_AMT4 <span style="color:#475569;font-size:0.65rem;">JUN</span></label>
          <input type="number" id="f-pamt4" value="17000" min="0" />
        </div>
        <div class="field">
          <label>PAY_AMT5 <span style="color:#475569;font-size:0.65rem;">MAY</span></label>
          <input type="number" id="f-pamt5" value="16000" min="0" />
        </div>
        <div class="field">
          <label>PAY_AMT6 <span style="color:#475569;font-size:0.65rem;">APR</span></label>
          <input type="number" id="f-pamt6" value="15500" min="0" />
        </div>
      </div>

      <div class="submit-row">
        <button class="submit-btn" id="pred-btn" onclick="runPrediction()">🔮 Predict Default Risk</button>
      </div>
      <div id="pred-result"></div>
    </div>
  </main>

  <script>
    // ── Panel switching ────────────────────────────────────────────────────────
    function showPanel(name) {
      document.querySelectorAll('.panel').forEach(p => p.classList.remove('active'));
      const panel = document.getElementById('panel-' + name);
      if (panel) panel.classList.add('active');
      if (name === 'runs') fetchRuns();
    }

    // ── Quick-fill presets ────────────────────────────────────────────────────
    const PRESETS = {
      low: {
        limit:50000, sex:2, edu:2, mar:2, age:35,
        pay0:-1, pay2:-1, pay3:-1, pay4:-1, pay5:-1, pay6:-1,
        bill1:20000, bill2:19000, bill3:18500, bill4:17000, bill5:16000, bill6:15500,
        pamt1:19000, pamt2:17000, pamt3:18500, pamt4:17000, pamt5:16000, pamt6:15500,
      },
      high: {
        limit:50000, sex:1, edu:3, mar:2, age:28,
        pay0:3, pay2:2, pay3:2, pay4:1, pay5:1, pay6:0,
        bill1:49000, bill2:47000, bill3:45000, bill4:43000, bill5:40000, bill6:37000,
        pamt1:0, pamt2:0, pamt3:500, pamt4:500, pamt5:1000, pamt6:1500,
      },
      medium: {
        limit:150000, sex:2, edu:2, mar:1, age:42,
        pay0:0, pay2:0, pay3:-1, pay4:-1, pay5:0, pay6:-1,
        bill1:60000, bill2:58000, bill3:55000, bill4:50000, bill5:48000, bill6:45000,
        pamt1:3000, pamt2:2500, pamt3:5000, pamt4:2000, pamt5:3000, pamt6:2500,
      },
    };

    function fillPreset(key) {
      const p = PRESETS[key];
      if (!p) return;
      document.getElementById('f-limit').value   = p.limit;
      document.getElementById('f-sex').value     = p.sex;
      document.getElementById('f-education').value = p.edu;
      document.getElementById('f-marriage').value  = p.mar;
      document.getElementById('f-age').value     = p.age;
      document.getElementById('f-pay0').value    = p.pay0;
      document.getElementById('f-pay2').value    = p.pay2;
      document.getElementById('f-pay3').value    = p.pay3;
      document.getElementById('f-pay4').value    = p.pay4;
      document.getElementById('f-pay5').value    = p.pay5;
      document.getElementById('f-pay6').value    = p.pay6;
      document.getElementById('f-bill1').value   = p.bill1;
      document.getElementById('f-bill2').value   = p.bill2;
      document.getElementById('f-bill3').value   = p.bill3;
      document.getElementById('f-bill4').value   = p.bill4;
      document.getElementById('f-bill5').value   = p.bill5;
      document.getElementById('f-bill6').value   = p.bill6;
      document.getElementById('f-pamt1').value   = p.pamt1;
      document.getElementById('f-pamt2').value   = p.pamt2;
      document.getElementById('f-pamt3').value   = p.pamt3;
      document.getElementById('f-pamt4').value   = p.pamt4;
      document.getElementById('f-pamt5').value   = p.pamt5;
      document.getElementById('f-pamt6').value   = p.pamt6;
    }

    // ── MLflow Runs table ──────────────────────────────────────────────────────
    async function fetchRuns() {
      const el = document.getElementById('runs-content');
      el.innerHTML = '<div class="loader"><div class="spinner"></div><span>Fetching runs...</span></div>';
      try {
        const res  = await fetch('/api/mlruns');
        const data = await res.json();
        if (!data.runs || data.runs.length === 0) {
          el.innerHTML = '<div class="msg-box">' + (data.message || 'No runs found.') + '</div>';
          return;
        }
        renderRunsTable(data.runs, el);
      } catch (e) {
        el.innerHTML = '<div class="error-box">Failed to fetch runs: ' + e.message + '</div>';
      }
    }

    function renderRunsTable(runs, el) {
      const maxComp = Math.max(...runs.map(r => r.composite_score));

      let html = `<div class="tbl-wrap"><table><thead><tr>
        <th>Rank</th><th>Model</th>
        <th>Composite ↓</th><th>F2</th><th>Recall</th><th>PR-AUC</th>
        <th>ROC-AUC</th><th>CV-AUC</th><th>Precision</th><th>F1</th>
        <th>Threshold</th><th>Accuracy</th>
      </tr></thead><tbody>`;

      runs.forEach((r, i) => {
        const isBest = r.composite_score === maxComp;
        html += `<tr>
          <td><span class="badge ${isBest ? 'badge-best' : 'badge-ok'}">${isBest ? '🥇 Best' : '#'+(i+1)}</span></td>
          <td style="font-weight:700;color:#e2e8f0">${r.model}</td>
          <td><span class="metric-pill pill-violet">${r.composite_score}</span></td>
          <td><span class="metric-pill pill-amber">${r.f2_score}</span></td>
          <td><span class="metric-pill pill-rose">${r.recall}</span></td>
          <td><span class="metric-pill pill-green">${r.pr_auc}</span></td>
          <td><span class="metric-pill pill-cyan">${r.roc_auc}</span></td>
          <td style="font-size:0.78rem;color:#94a3b8">${r.cv_roc_auc_mean} ± ${r.cv_roc_auc_std}</td>
          <td><span class="metric-pill pill-blue">${r.precision}</span></td>
          <td style="color:#64748b">${r.f1_score}</td>
          <td style="color:#f59e0b;font-weight:600">${r.threshold}</td>
          <td style="color:#64748b">${(r.accuracy*100).toFixed(1)}%</td>
        </tr>`;
      });

      html += `</tbody></table></div>
        <p style="font-size:0.72rem;color:#475569;margin-top:14px;">
          Selection criterion: Composite = 0.40×F2 + 0.25×PR-AUC + 0.20×CV-AUC + 0.15×ROC-AUC.
          Threshold is F2-optimal (not fixed at 0.5). CV = 5-fold stratified.
        </p>`;
      el.innerHTML = html;
    }

    // ── Prediction ────────────────────────────────────────────────────────────
    function buildPayload() {
      return {
        LIMIT_BAL:  parseFloat(document.getElementById('f-limit').value),
        SEX:        parseInt(document.getElementById('f-sex').value),
        EDUCATION:  parseInt(document.getElementById('f-education').value),
        MARRIAGE:   parseInt(document.getElementById('f-marriage').value),
        AGE:        parseInt(document.getElementById('f-age').value),
        PAY_0:      parseInt(document.getElementById('f-pay0').value),
        PAY_2:      parseInt(document.getElementById('f-pay2').value),
        PAY_3:      parseInt(document.getElementById('f-pay3').value),
        PAY_4:      parseInt(document.getElementById('f-pay4').value),
        PAY_5:      parseInt(document.getElementById('f-pay5').value),
        PAY_6:      parseInt(document.getElementById('f-pay6').value),
        BILL_AMT1:  parseFloat(document.getElementById('f-bill1').value),
        BILL_AMT2:  parseFloat(document.getElementById('f-bill2').value),
        BILL_AMT3:  parseFloat(document.getElementById('f-bill3').value),
        BILL_AMT4:  parseFloat(document.getElementById('f-bill4').value),
        BILL_AMT5:  parseFloat(document.getElementById('f-bill5').value),
        BILL_AMT6:  parseFloat(document.getElementById('f-bill6').value),
        PAY_AMT1:   parseFloat(document.getElementById('f-pamt1').value),
        PAY_AMT2:   parseFloat(document.getElementById('f-pamt2').value),
        PAY_AMT3:   parseFloat(document.getElementById('f-pamt3').value),
        PAY_AMT4:   parseFloat(document.getElementById('f-pamt4').value),
        PAY_AMT5:   parseFloat(document.getElementById('f-pamt5').value),
        PAY_AMT6:   parseFloat(document.getElementById('f-pamt6').value),
      };
    }

    async function runPrediction() {
      const btn      = document.getElementById('pred-btn');
      const resultEl = document.getElementById('pred-result');
      btn.disabled   = true;
      resultEl.innerHTML = '<div class="loader"><div class="spinner"></div><span>Running prediction...</span></div>';

      try {
        const res  = await fetch('/predict', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(buildPayload()),
        });
        const data = await res.json();
        if (!res.ok) {
          resultEl.innerHTML = '<div class="error-box">Error ' + res.status + ': ' + (data.detail || JSON.stringify(data)) + '</div>';
        } else {
          renderPrediction(data, resultEl);
        }
      } catch (e) {
        resultEl.innerHTML = '<div class="error-box">Request failed: ' + e.message + '</div>';
      } finally {
        btn.disabled = false;
      }
    }

    function renderPrediction(d, el) {
      const riskColor  = d.risk_label === 'Low' ? 'risk-low' : d.risk_label === 'Medium' ? 'risk-medium' : 'risk-high';
      const verdict    = d.prediction === 0 ? '✅ No Default' : '⚠️ Will Default';
      const vColor     = d.prediction === 0 ? 'risk-low' : 'risk-high';
      const prob       = d.probability_of_default;
      const threshold  = d.threshold_applied;
      const probPct    = (prob * 100).toFixed(2);
      const fillColor  = d.risk_label === 'Low' ? '#4ade80' : d.risk_label === 'Medium' ? '#fbbf24' : '#f87171';
      const threshPct  = (threshold * 100).toFixed(1);

      el.innerHTML = `
        <div class="pred-result">
          <h3>Prediction Result</h3>
          <div class="pred-cards">
            <div class="pred-card">
              <div class="card-label">Verdict</div>
              <div class="card-value ${vColor}" style="font-size:1rem">${verdict}</div>
            </div>
            <div class="pred-card">
              <div class="card-label">Default Probability</div>
              <div class="card-value text-indigo">${probPct}%</div>
            </div>
            <div class="pred-card">
              <div class="card-label">Risk Band</div>
              <div class="card-value ${riskColor}">${d.risk_label}</div>
            </div>
            <div class="pred-card">
              <div class="card-label">Decision Threshold</div>
              <div class="card-value text-slate" style="font-size:1rem">${threshold}</div>
            </div>
            <div class="pred-card">
              <div class="card-label">Model Used</div>
              <div class="card-value text-cyan" style="font-size:0.95rem">${d.model_used}</div>
            </div>
          </div>

          <!-- Probability bar with threshold marker -->
          <div class="prob-bar-wrap">
            <div class="prob-bar-label">
              Default probability vs decision threshold
              <span style="color:#f59e0b;margin-left:8px;">┃ = threshold (${threshPct}%)</span>
            </div>
            <div class="prob-bar">
              <div class="bar-fill prob-bar-fill"
                   style="width:${probPct}%;background:${fillColor};"></div>
              <div class="threshold-line" style="left:${threshPct}%"></div>
            </div>
            <div style="display:flex;justify-content:space-between;font-size:0.68rem;color:#475569;margin-top:4px;">
              <span>0%</span><span>50%</span><span>100%</span>
            </div>
          </div>
        </div>`;
    }
  </script>
</body>
</html>"""