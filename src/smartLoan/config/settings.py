from pydantic import BaseSettings
from typing import List

class Settings(BaseSettings):

    # App
    APP_NAME: str = 'smartLoan'
    DEBUG: bool = False

    # MLflow
    MLFLOW_TRACKING_URI: str = "http://localhost:5000"
    MLFLOW_EXPERIMENT_NAME: str = "smartloan-exp"

    # Model
    MODEL_NAME: str = "smartloan_m1"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

settings = Settings()
