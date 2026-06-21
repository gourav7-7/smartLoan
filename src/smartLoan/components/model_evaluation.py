import os, sys
import json
import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    roc_auc_score, accuracy_score, f1_score,
    fbeta_score, average_precision_score,
    precision_score, recall_score,
    classification_report, confusion_matrix,
)

from smartLoan.utils.logger import logger
from smartLoan.utils.exceptions import CustomException

MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow/mlflow.db")
mlflow.set_tracking_uri(MLFLOW_URI)

ARTIFACT_DIR = "artifacts/evaluation"
MODEL_DIR    = "artifacts/models"
os.makedirs(ARTIFACT_DIR, exist_ok=True)


class ModelEvaluator:
    def __init__(self):
        self.processed_dir   = "artifacts/processed"
        self.experiment_name = "SmartLoan_UCI_CreditDefault"   # ← matches trainer

    # ─────────────────────────────────────────────────────────────────
    def load_test_data(self):
        x_test = pd.read_csv(f"{self.processed_dir}/x_test.csv")
        y_test = pd.read_csv(f"{self.processed_dir}/y_test.csv").values.ravel()
        return x_test, y_test

    # ─────────────────────────────────────────────────────────────────
    def load_model_with_threshold(self, model_name):
        """
        Load the joblib dict saved by trainer:
            {"model": CalibratedClassifierCV, "threshold": float}
        This is the single source of truth — threshold is baked in.
        """
        path = f"{MODEL_DIR}/{model_name}.pkl"
        payload = joblib.load(path)
        return payload["model"], payload["threshold"]

    # ─────────────────────────────────────────────────────────────────
    def get_all_runs(self) -> pd.DataFrame:
        experiment = mlflow.get_experiment_by_name(self.experiment_name)
        if experiment is None:
            raise RuntimeError(
                f"Experiment '{self.experiment_name}' not found. Run training first."
            )
        runs = mlflow.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=["metrics.composite_score DESC"],   # ← rank by composite, not just ROC
        )
        return runs

    # ─────────────────────────────────────────────────────────────────
    def evaluate_model(self, model, threshold, x_test, y_test):
        """
        Evaluate using the optimal threshold saved during training —
        not the default 0.5. Mirrors the evaluate() method in ModelTrainer.
        """
        y_prob = model.predict_proba(x_test)[:, 1]
        y_pred = (y_prob >= threshold).astype(int)   # ← apply stored threshold

        return {
            "threshold":  round(threshold, 4),
            "accuracy":   round(accuracy_score(y_test, y_pred),                      4),
            "precision":  round(precision_score(y_test, y_pred, zero_division=0),    4),
            "recall":     round(recall_score(y_test, y_pred, zero_division=0),       4),
            "f1_score":   round(f1_score(y_test, y_pred, zero_division=0),           4),
            "f2_score":   round(fbeta_score(y_test, y_pred, beta=2, zero_division=0),4),
            "roc_auc":    round(roc_auc_score(y_test, y_prob),                       4),
            "pr_auc":     round(average_precision_score(y_test, y_prob),             4),
        }

    # ─────────────────────────────────────────────────────────────────
    def plot_confusion_matrix(self, model, threshold, x_test, y_test, model_name):
        y_prob = model.predict_proba(x_test)[:, 1]
        y_pred = (y_prob >= threshold).astype(int)   # ← same threshold, not 0.5
        cm = confusion_matrix(y_test, y_pred)

        plt.figure(figsize=(6, 5))
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Repay", "Default"],
            yticklabels=["Repay", "Default"],
        )
        plt.title(f"Confusion Matrix — {model_name} (threshold={threshold:.4f})")
        plt.ylabel("Actual")
        plt.xlabel("Predicted")
        plt.tight_layout()
        path = f"{ARTIFACT_DIR}/confusion_matrix_{model_name}.png"
        plt.savefig(path)
        plt.close()
        logger.info(f"Confusion matrix saved → {path}")

    # ─────────────────────────────────────────────────────────────────
    def evaluate(self):
        try:
            logger.info("Starting Model Evaluation...")
            x_test, y_test = self.load_test_data()
            runs = self.get_all_runs()

            if runs.empty:
                logger.warning("No MLflow runs found. Run training first.")
                return

            logger.info(f"Found {len(runs)} MLflow runs.")

            summary        = []
            best_metrics   = None
            best_model     = None
            best_threshold = None
            best_name      = None
            best_composite = -1

            for _, row in runs.iterrows():
                run_name = row.get("tags.mlflow.runName", "unknown")

                try:
                    model, threshold = self.load_model_with_threshold(run_name)
                    metrics = self.evaluate_model(model, threshold, x_test, y_test)

                    # Composite score — mirrors trainer logic
                    composite = round(
                        0.40 * metrics["f2_score"] +
                        0.25 * metrics["pr_auc"]   +
                        0.20 * float(row.get("metrics.cv_roc_auc_mean", 0)) +
                        0.15 * metrics["roc_auc"],
                        4
                    )

                    summary.append({"model": run_name, **metrics, "composite": composite})

                    logger.info(
                        f"{run_name} → "
                        f"F2: {metrics['f2_score']} | Recall: {metrics['recall']} | "
                        f"Precision: {metrics['precision']} | ROC-AUC: {metrics['roc_auc']} | "
                        f"Accuracy: {metrics['accuracy']} | Composite: {composite}"
                    )

                    if composite > best_composite:
                        best_composite = composite
                        best_metrics   = metrics
                        best_model     = model
                        best_threshold = threshold
                        best_name      = run_name

                except Exception as load_err:
                    logger.warning(f"Could not evaluate {run_name}: {load_err}")

            # ── Leaderboard ───────────────────────────────────────────
            df_summary = pd.DataFrame(summary).sort_values("composite", ascending=False)
            logger.info(f"\n{'='*60}\n{df_summary.to_string(index=False)}\n{'='*60}")
            df_summary.to_csv(f"{ARTIFACT_DIR}/leaderboard.csv", index=False)

            # ── Best model artifacts ──────────────────────────────────
            if best_model is not None:
                self.plot_confusion_matrix(best_model, best_threshold, x_test, y_test, best_name)

                report = classification_report(
                    y_test,
                    (best_model.predict_proba(x_test)[:, 1] >= best_threshold).astype(int),
                    output_dict=True
                )

                eval_report = {
                    "best_model":              best_name,
                    "composite_score":         best_composite,
                    "metrics":                 best_metrics,
                    "leaderboard":             summary,
                    "classification_report":   report,
                }
                with open(f"{ARTIFACT_DIR}/eval_report.json", "w") as f:
                    json.dump(eval_report, f, indent=4)

                # Register best model in MLflow Model Registry
                runs_df = self.get_all_runs()
                best_run_id = runs_df[
                    runs_df["tags.mlflow.runName"] == best_name
                ]["run_id"].values[0]

                model_uri  = f"runs:/{best_run_id}/{best_name}"
                registered = mlflow.register_model(model_uri, name="SmartLoan_BestModel")
                logger.info(
                    f"Best model '{best_name}' registered as "
                    f"'SmartLoan_BestModel' v{registered.version}"
                )

            logger.info("Model Evaluation Completed ✅")

        except Exception as e:
            raise CustomException(e, sys)