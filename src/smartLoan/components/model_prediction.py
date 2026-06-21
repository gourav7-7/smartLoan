import os, sys
import numpy as np
import pandas as pd
import joblib
import json
from pathlib import Path

from smartLoan.utils.logger import logger
from smartLoan.utils.exceptions import CustomException

from smartLoan.config import paths as _paths
MODEL_DIR      = _paths.MODEL_DIR
PROCESSED_DIR  = _paths.PROCESSED_DATA_DIR
SCALER_PATH    = _paths.SCALER_PATH
COLUMNS_PATH   = _paths.COLUMNS_PATH
BEST_INFO_PATH = _paths.BEST_MODEL_INFO


class ModelPrediction:
    def __init__(self, model_name: str = "best_model"):
        """
        model_name: name of the .pkl file in artifacts/models/ without extension.
                    Defaults to 'best_model' — the canonical winner saved by ModelTrainer.
                    Pass a specific name (e.g. 'RandomForest') to force a particular model.
        """
        self.model_name    = model_name
        self.model         = None
        self.threshold     = 0.5
        self.scaler        = None
        self._display_name = model_name
        self.training_columns: list = []
        self._load_artifacts()

    # ─────────────────────────────────────────────────────────────────────────
    def _load_artifacts(self):
        try:
            # ── 1. Load model + threshold from joblib payload ─────────────────
            model_path = MODEL_DIR / f"{self.model_name}.pkl"
            if not model_path.exists():
                raise FileNotFoundError(
                    f"Model file not found: {model_path}. "
                    "Run the training pipeline first."
                )

            payload        = joblib.load(model_path)
            self.model     = payload["model"]
            self.threshold = payload["threshold"]
            logger.info(
                f"Model '{self.model_name}' loaded | "
                f"Optimal threshold: {self.threshold:.4f}"
            )

            # ── 2. Resolve actual model name if using best_model ──────────────
            if self.model_name == "best_model" and BEST_INFO_PATH.exists():
                with open(BEST_INFO_PATH) as f:
                    info = json.load(f)
                self._display_name = info.get("best_model", "best_model")
                logger.info(f"Best model resolved to: {self._display_name}")
            else:
                self._display_name = self.model_name

            # ── 3. Load training column names ─────────────────────────────────
            if COLUMNS_PATH.exists():
                with open(COLUMNS_PATH) as f:
                    self.training_columns = [line.strip() for line in f if line.strip()]
                logger.info(f"Loaded {len(self.training_columns)} feature names")
            else:
                raise FileNotFoundError(
                    f"training_columns.txt not found at {COLUMNS_PATH}. "
                    "Re-run the data transformation stage."
                )

            # ── 4. Load scaler ────────────────────────────────────────────────
            if SCALER_PATH.exists():
                self.scaler = joblib.load(SCALER_PATH)
                logger.info("Scaler loaded successfully")
            else:
                logger.warning(
                    "scaler.pkl not found — predictions will use unscaled input. "
                    "Re-run data transformation to regenerate the scaler."
                )

        except Exception as e:
            raise CustomException(e, sys)

    # ─────────────────────────────────────────────────────────────────────────
    def _engineer_features(self, data: dict) -> pd.DataFrame:
        """
        Prepares a raw UCI input record for prediction.

        This MUST mirror DataTransformation step-for-step, or the model is fed
        features on a different scale than it was trained on (train/serve skew),
        and every probability it returns is meaningless. The order below is the
        same as training:

          1. Engineered features, computed from RAW values:
               UTIL_RATIO   = mean(BILL_AMT1..6) / (LIMIT_BAL + 1)
               PAY_TREND    = PAY_0 - PAY_6
               AVG_PAY_AMT  = mean(PAY_AMT1..6)
               AVG_BILL_AMT = mean(BILL_AMT1..6)
          2. log1p on PAY_AMT1..6 and AVG_PAY_AMT (skewed, always >= 0).
             AVG_BILL_AMT and UTIL_RATIO are NOT log-transformed (training skips
             them — BILL_AMT can be negative).
          3. Align to the saved training schema (training_columns.txt).
          4. Clean inf/NaN, then apply the saved StandardScaler.
        """
        df = pd.DataFrame([data])

        # ── Identify column groups ────────────────────────────────────────────
        bill_cols    = [f"BILL_AMT{i}" for i in range(1, 7) if f"BILL_AMT{i}" in df.columns]
        pay_amt_cols = [f"PAY_AMT{i}"  for i in range(1, 7) if f"PAY_AMT{i}"  in df.columns]

        # ── 1. Engineer features from RAW values (exactly as DataTransformation)
        if bill_cols:
            df["UTIL_RATIO"]   = df[bill_cols].mean(axis=1) / (df["LIMIT_BAL"] + 1)
            df["AVG_BILL_AMT"] = df[bill_cols].mean(axis=1)
        if {"PAY_0", "PAY_6"}.issubset(df.columns):
            df["PAY_TREND"]    = df["PAY_0"] - df["PAY_6"]
        if pay_amt_cols:
            df["AVG_PAY_AMT"]  = df[pay_amt_cols].mean(axis=1)

        # ── 2. log1p on the same skewed, non-negative cols as training ─────────
        #    AVG_PAY_AMT is built from RAW PAY_AMT above, then log1p'd here —
        #    so it equals log1p(mean(raw)), matching training.
        for col in pay_amt_cols + ["AVG_PAY_AMT"]:
            if col in df.columns and (df[col] >= 0).all():
                df[col] = np.log1p(df[col])

        # ── 3. Align to training schema ───────────────────────────────────────
        #    Any column the model expects but we didn't build gets created as 0.
        #    With the engineering above, this no longer silently zeroes out real
        #    features — it's now only a safety net for genuinely absent columns.
        for col in self.training_columns:
            if col not in df.columns:
                df[col] = 0
        df = df[self.training_columns]

        # ── 4. Clean infinities / NaNs ────────────────────────────────────────
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.fillna(0, inplace=True)

        # ── 5. Apply scaler ───────────────────────────────────────────────────
        if self.scaler is not None:
            scaled = self.scaler.transform(df)
            df = pd.DataFrame(scaled, columns=df.columns)

        return df

    # ─────────────────────────────────────────────────────────────────────────
    def _risk_label(self, prob: float) -> str:
        """
        Risk bands relative to the optimal threshold — not hardcoded values.
          < 0.5 × threshold           → Low    (comfortably safe)
          0.5 × threshold to threshold → Medium (borderline, monitor)
          >= threshold                → High   (flag for review)
        """
        half = self.threshold * 0.5
        if prob < half:
            return "Low"
        elif prob < self.threshold:
            return "Medium"
        else:
            return "High"

    # ─────────────────────────────────────────────────────────────────────────
    def predict(self, input_data: dict) -> dict:
        """
        Runs end-to-end prediction on a single customer record.

        Returns:
            prediction              : 0 (no default) or 1 (default)
            probability_of_default  : calibrated probability score [0, 1]
            risk_label              : Low / Medium / High relative to threshold
            model_used              : resolved model name
            threshold_applied       : stored optimal threshold used
        """
        try:
            logger.info(f"Running prediction | Input keys: {list(input_data.keys())}")

            X    = self._engineer_features(input_data)
            prob = float(self.model.predict_proba(X)[0][1])

            # Apply stored optimal threshold — NOT model.predict() which uses 0.5
            prediction = int(prob >= self.threshold)
            risk       = self._risk_label(prob)

            logger.info(
                f"Result → prediction: {prediction} | "
                f"prob: {prob:.4f} | threshold: {self.threshold:.4f} | risk: {risk}"
            )

            return {
                "prediction":             prediction,
                "probability_of_default": round(prob, 4),
                "risk_label":             risk,
                "model_used":             self._display_name,
                "threshold_applied":      round(self.threshold, 4),
            }

        except Exception as e:
            raise CustomException(e, sys)