# SmartLoan — Credit Default Risk Prediction

A production-grade machine learning system that predicts the probability a credit-card client will **default on their next payment**, served through a FastAPI REST API with a built-in dashboard. Built end-to-end on the [UCI "Default of Credit Card Clients"](https://www.kaggle.com/datasets/uciml/default-of-credit-card-clients-dataset) dataset.

The project covers the full MLOps lifecycle: data ingestion, validation + EDA, feature engineering, multi-model training with experiment tracking, calibrated probability output, business-aware threshold selection, model evaluation with a registry, and containerised serving.

---

## Highlights

- **Four models, one winner.** Logistic Regression, Random Forest, LightGBM, and XGBoost are trained and compared, with the best selected automatically by a **composite score** rather than raw accuracy.
- **Business-aware metric.** Selection is driven by **F2** (recall weighted 2× over precision) because a missed defaulter costs a lender far more than a false alarm — accuracy is deliberately excluded from selection.
- **Calibrated probabilities.** Each model is wrapped in isotonic calibration so the predicted probability of default is trustworthy, not just a ranking score.
- **Optimal decision threshold.** Instead of a naive 0.5 cutoff, the threshold is tuned on a held-out validation set by sweeping the precision–recall curve and maximising F2 under a minimum-precision floor.
- **Experiment tracking + registry.** Every run is logged to MLflow; the winner is registered in the MLflow Model Registry.
- **Single source of truth for serving.** The chosen model and its threshold are saved together, so the API applies the exact same threshold the model was tuned with.
- **Containerised.** `docker compose up` builds and runs the API, dashboard, and persistent volumes for artifacts, logs, and the MLflow DB.

---

## Tech stack

| Layer | Tools |
|---|---|
| Language | Python 3.10 |
| ML / data | scikit-learn, LightGBM, XGBoost, imbalanced-learn (SMOTE), pandas, numpy |
| Experiment tracking | MLflow (SQLite backend) |
| API | FastAPI, Uvicorn, Pydantic v2, pydantic-settings |
| Visualisation | matplotlib, seaborn |
| Packaging / ops | Docker, docker-compose, setuptools |

---

## Project structure

```
smartloan/
├── Dockerfile
├── docker-compose.yml
├── pyproject.toml
├── .env.example                 # copy to .env and fill in
├── .gitignore  .dockerignore
└── src/smartLoan/
    ├── api/
    │   ├── main.py              # FastAPI app, routes, lifespan model load
    │   └── ui.py               # embedded dashboard (HTML/CSS/JS)
    ├── components/
    │   ├── data_ingestion.py    # download + unzip dataset (Kaggle CLI)
    │   ├── data_validation.py   # schema/range checks + EDA report
    │   ├── data_transformation.py # feature engineering, log1p, scaling, split
    │   ├── model_training.py    # 4 models, CV, calibration, threshold, MLflow
    │   ├── model_evaluation.py  # leaderboard, confusion matrix, registry
    │   └── model_prediction.py  # serving-time feature build + predict
    ├── pipeline/
    │   ├── training_pipeline.py # ingestion → validation → transform → train → eval
    │   └── prediction_pipeline.py
    ├── config/
    │   ├── settings.py          # pydantic-settings, reads .env
    │   └── paths.py            # all filesystem paths
    ├── schemas/request_schema.py # request/response Pydantic models
    └── utils/
        ├── logger.py
        └── exceptions.py
```

---

## Dataset

[UCI Default of Credit Card Clients](https://www.kaggle.com/datasets/uciml/default-of-credit-card-clients-dataset) — 30,000 Taiwanese credit-card clients, ~22% default rate (imbalanced). Each record has:

- **Demographics:** `LIMIT_BAL`, `SEX`, `EDUCATION`, `MARRIAGE`, `AGE`
- **Repayment status (6 months):** `PAY_0`, `PAY_2`–`PAY_6` (note: there is no `PAY_1` in the dataset — not a typo)
- **Bill amounts (6 months):** `BILL_AMT1`–`BILL_AMT6`
- **Payment amounts (6 months):** `PAY_AMT1`–`PAY_AMT6`
- **Target:** `default.payment.next.month` → renamed to `TARGET`

### Engineered features

| Feature | Definition |
|---|---|
| `UTIL_RATIO` | mean(BILL_AMT1..6) / (LIMIT_BAL + 1) — credit utilisation |
| `PAY_TREND` | PAY_0 − PAY_6 — delinquency trend |
| `AVG_PAY_AMT` | mean(PAY_AMT1..6) — average repayment |
| `AVG_BILL_AMT` | mean(BILL_AMT1..6) — average bill |

`PAY_AMT*` and `AVG_PAY_AMT` are `log1p`-transformed (heavy right skew, always ≥ 0); `BILL_AMT*` are not (they can be negative). The serving path reproduces this transformation exactly so there is no train/serve skew.

---

## ML methodology

1. **Resampling.** Mild rebalancing with SMOTE + random under-sampling (to ~2:1, not 1:1) — over-rebalancing would miscalibrate probabilities on a 22%-base-rate event. Applied to the training fold only.
2. **Cross-validation.** 5-fold stratified CV on the resampled training set to measure stability (ROC-AUC and F2) before trusting any single split.
3. **Calibration.** `CalibratedClassifierCV(method="isotonic")` on each fitted model.
4. **Threshold tuning.** On a clean validation set (split before SMOTE), sweep the PR curve and pick the F2-maximising threshold subject to `precision ≥ MIN_PRECISION_THRESHOLD`.
5. **Composite model selection.**

   ```
   score = 0.40·F2 + 0.25·PR_AUC + 0.20·CV_ROC_AUC + 0.15·ROC_AUC
   ```

   The winner is saved as `best_model.pkl` (model + threshold) and registered in MLflow.

> Trained artifacts (`artifacts/`, `mlflow/`) are gitignored. Run the training pipeline to generate them — see below.

---

## Getting started

### Prerequisites
- Python 3.10+
- Kaggle API credentials (only needed to download the dataset) — [how to get them](https://www.kaggle.com/docs/api)

### 1. Clone and configure

```bash
git clone https://github.com/<your-username>/smartloan.git
cd smartloan
cp .env.example .env          # then add your KAGGLE_USERNAME / KAGGLE_KEY
```

### 2. Run locally (without Docker)

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\Activate.ps1
pip install -e .                   # installs from pyproject.toml

# Train end-to-end (downloads data, trains, evaluates, registers best model)
python -m smartLoan.pipeline.training_pipeline

# Serve the API + dashboard
uvicorn smartLoan.api.main:app --reload
```

Then open **http://localhost:8000** for the dashboard, or **http://localhost:8000/docs** for interactive Swagger docs.

### 3. Run with Docker

```bash
docker compose up --build
```

This starts the API on port 8000 with persistent volumes for `artifacts`, `mlflow`, and `logs`. Trigger training from the dashboard or via `POST /api/train` once the container is up.

---

## API reference

| Method | Endpoint | Description |
|---|---|---|
| `GET`  | `/` | Dashboard UI |
| `GET`  | `/health` | API + model status |
| `GET`  | `/docs` | Swagger / OpenAPI docs |
| `GET`  | `/api/mlruns` | All MLflow runs, ranked by composite score |
| `POST` | `/api/train` | Start the training pipeline in the background |
| `GET`  | `/api/train/status` | Poll training progress |
| `POST` | `/predict` | Predict default risk for one client |
| `POST` | `/predict/batch` | Predict for up to 100 clients |

### Example: single prediction

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "LIMIT_BAL": 50000, "SEX": 2, "EDUCATION": 2, "MARRIAGE": 1, "AGE": 35,
    "PAY_0": -1, "PAY_2": -1, "PAY_3": -1, "PAY_4": -1, "PAY_5": -1, "PAY_6": -1,
    "BILL_AMT1": 20000, "BILL_AMT2": 19000, "BILL_AMT3": 18500,
    "BILL_AMT4": 17000, "BILL_AMT5": 16000, "BILL_AMT6": 15500,
    "PAY_AMT1": 20000, "PAY_AMT2": 17000, "PAY_AMT3": 18500,
    "PAY_AMT4": 17000, "PAY_AMT5": 16000, "PAY_AMT6": 15500
  }'
```

```json
{
  "prediction": 0,
  "probability_of_default": 0.0731,
  "risk_label": "Low",
  "model_used": "LightGBM",
  "threshold_applied": 0.31
}
```

`risk_label` is banded relative to the model's tuned threshold: **Low** (< 0.5×threshold), **Medium** (0.5×threshold–threshold), **High** (≥ threshold).

---

## Configuration

All settings live in `.env` (see `.env.example`) and are loaded by `config/settings.py` via `pydantic-settings`. Key knobs include the MLflow tracking URI, Kaggle credentials, artifact paths, SMOTE ratios, CV folds, the minimum-precision floor, and the composite-score weights — change behaviour without touching code.

---

## Roadmap

- Unit tests (train/serve feature-engineering parity, schema validation)
- CI workflow (lint + tests on push)
- Remote MLflow tracking server for multi-user experiments
- Drift monitoring on incoming prediction traffic

---