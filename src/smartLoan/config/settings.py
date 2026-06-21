# ─────────────────────────────────────────────────────────────────────────────
# config/settings.py
#
# Central settings object — reads every value from .env (or environment).
# Uses pydantic-settings (v2) which ships with the project's fastapi stack.
#
# Usage anywhere in the project:
#   from smartLoan.config.settings import settings
#   print(settings.MLFLOW_TRACKING_URI)
# ─────────────────────────────────────────────────────────────────────────────

from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import List


class Settings(BaseSettings):

    # ── Application ───────────────────────────────────────────────────────────
    APP_NAME:    str  = "SmartLoan"
    APP_VERSION: str  = "1.0.0"
    DEBUG:       bool = False
    ENV:         str  = "local"   # local | development | staging | production

    # ── API Server ────────────────────────────────────────────────────────────
    API_HOST:    str  = "0.0.0.0"
    API_PORT:    int  = 8000
    API_WORKERS: int  = 1
    API_RELOAD:  bool = False

    # ── MLflow ────────────────────────────────────────────────────────────────
    MLFLOW_TRACKING_URI:    str = "sqlite:///mlflow/mlflow.db"
    MLFLOW_EXPERIMENT_NAME: str = "SmartLoan_UCI_CreditDefault"
    MLFLOW_MODEL_NAME:      str = "SmartLoan_BestModel"
    MLFLOW_ARTIFACT_ROOT:   str = ""

    # ── Kaggle ────────────────────────────────────────────────────────────────
    KAGGLE_USERNAME: str = ""
    KAGGLE_KEY:      str = ""
    KAGGLE_DATASET:  str = "uciml/default-of-credit-card-clients-dataset"

    # ── Paths (relative to WORKDIR — /app in Docker, project root locally) ───
    ARTIFACTS_DIR:      str = "artifacts"
    RAW_DATA_DIR:       str = "artifacts/raw"
    PROCESSED_DATA_DIR: str = "artifacts/processed"
    MODEL_DIR:          str = "artifacts/models"
    EVALUATION_DIR:     str = "artifacts/evaluation"
    LOGS_DIR:           str = "logs"

    # ── Model / Training ──────────────────────────────────────────────────────
    MODEL_NAME:              str   = "best_model"
    MIN_PRECISION_THRESHOLD: float = 0.40
    SMOTE_OVER_RATIO:        float = 0.3
    SMOTE_UNDER_RATIO:       float = 0.5
    CV_FOLDS:                int   = 5
    WEIGHT_F2:               float = 0.40
    WEIGHT_PR_AUC:           float = 0.25
    WEIGHT_CV_AUC:           float = 0.20
    WEIGHT_ROC_AUC:          float = 0.15
    RANDOM_STATE:            int   = 42

    # ── CORS ──────────────────────────────────────────────────────────────────
    # Stored as comma-separated string in .env: CORS_ORIGINS=*
    # or: CORS_ORIGINS=https://smartloan.yourdomain.com,https://other.com
    CORS_ORIGINS: str = "*"

    @property
    def cors_origins_list(self) -> List[str]:
        """Return CORS_ORIGINS as a list for CORSMiddleware."""
        if self.CORS_ORIGINS.strip() == "*":
            return ["*"]
        return [o.strip() for o in self.CORS_ORIGINS.split(",") if o.strip()]

    # ── Logging ───────────────────────────────────────────────────────────────
    LOG_LEVEL:  str = "INFO"
    LOG_FORMAT: str = "[%(asctime)s: %(levelname)s: %(module)s: %(message)s]"
    LOG_FILE:   str = "logs/running_logs.log"

    # ── Pydantic-settings v2 config ───────────────────────────────────────────
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,  # env var names are uppercase — match exactly
        extra="ignore",       # silently ignore any extra keys in .env
    )


# Singleton — import this everywhere
settings = Settings()
