# ─────────────────────────────────────────────────────────────────────────────
# config/paths.py
#
# All filesystem paths used across the pipeline.
# Values are driven by settings (which reads from .env) so Docker, local dev,
# and CI all resolve paths consistently without code changes.
# ─────────────────────────────────────────────────────────────────────────────

from pathlib import Path
from smartLoan.config.settings import settings

# Project root — 3 parents up from this file:
#   src/smartLoan/config/paths.py → config → smartLoan → src → ROOT
ROOT_DIR = Path(__file__).resolve().parents[3]

# ── Artifact directories (driven by .env / settings) ─────────────────────────
# Path() accepts relative strings — they resolve relative to CWD at runtime.
# Locally: CWD = project root.  In Docker: CWD = WORKDIR = /app.
ARTIFACTS_DIR      = Path(settings.ARTIFACTS_DIR)
RAW_DATA_DIR       = Path(settings.RAW_DATA_DIR)
PROCESSED_DATA_DIR = Path(settings.PROCESSED_DATA_DIR)
MODEL_DIR          = Path(settings.MODEL_DIR)
EVALUATION_DIR     = Path(settings.EVALUATION_DIR)
LOGS_DIR           = Path(settings.LOGS_DIR)

# ── Dataset ───────────────────────────────────────────────────────────────────
# Kaggle dataset slug — read from .env so it never needs a code change
SOURCE_DATASET = settings.KAGGLE_DATASET   # uciml/default-of-credit-card-clients-dataset

# ── Specific artefact files ───────────────────────────────────────────────────
BEST_MODEL_PATH = MODEL_DIR / f"{settings.MODEL_NAME}.pkl"
BEST_MODEL_INFO = MODEL_DIR / "best_model_info.json"
SCALER_PATH     = PROCESSED_DATA_DIR / "scaler.pkl"
COLUMNS_PATH    = PROCESSED_DATA_DIR / "training_columns.txt"
LOG_FILE        = Path(settings.LOG_FILE)
