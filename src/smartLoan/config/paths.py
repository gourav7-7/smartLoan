from pathlib import Path

# root directory of project
ROOT_DIR = Path(__file__).resolve().parents[3]

# Artifacts
ARTIFACTS_DIR = ROOT_DIR/ "artifacts"

# Data
SOURCE_DATASET = "home-credit-default-risk"
RAW_DATA_DIR = ARTIFACTS_DIR/ "raw"
PROCESSED_DATA_DIR = ARTIFACTS_DIR/ "processed"

# Model
MODEL_DIR = ARTIFACTS_DIR/ "models"
TRAINED_MODEL = MODEL_DIR/ "model.pkl"

# Logs
LOGS_DIR = ROOT_DIR/ "logs"

