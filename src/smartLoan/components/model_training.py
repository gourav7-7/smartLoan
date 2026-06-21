import pandas as pd
import numpy as np
import os, sys, json
import joblib
import mlflow
import mlflow.sklearn

from smartLoan.config.settings import settings
mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    roc_auc_score, accuracy_score, f1_score,
    recall_score, precision_score, average_precision_score,
    fbeta_score, precision_recall_curve, make_scorer
)


from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline

from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

from smartLoan.utils.logger import logger
from smartLoan.utils.exceptions import CustomException


class ModelTrainer:
    def __init__(self):
        self.artifact_dir = "artifacts/processed"
        self.model_dir    = "artifacts/models"
        os.makedirs(self.model_dir, exist_ok=True)

    # ─────────────────────────────────────────────────────────────────
    def load_data(self):
        x_train = pd.read_csv(f"{self.artifact_dir}/x_train.csv")
        x_test  = pd.read_csv(f"{self.artifact_dir}/x_test.csv")
        y_train = pd.read_csv(f"{self.artifact_dir}/y_train.csv").values.ravel()
        y_test  = pd.read_csv(f"{self.artifact_dir}/y_test.csv").values.ravel()

        # Hold out 15% of train as a clean validation set for threshold tuning.
        # This split happens BEFORE SMOTE so val set reflects real class distribution.
        x_train, x_val, y_train, y_val = train_test_split(
            x_train, y_train,
            test_size=0.15,
            stratify=y_train,
            random_state=42
        )

        logger.info(f"Data loaded → train: {x_train.shape}, val: {x_val.shape}, test: {x_test.shape}")
        return x_train, x_val, x_test, y_train, y_val, y_test
       
    # ─────────────────────────────────────────────────────────────────
    def smote(self, x_train, y_train):
        """
        UCI ratio: 3.52:1 (neg:pos).
        SMOTE + UnderSampler → final ratio ~2:1 (~21K rows).
        Mild rebalancing intentionally — over-rebalancing to 1:1 
        miscalibrates probabilities on a 22% base-rate event.
        """
        pipe = Pipeline([
            ('over',  SMOTE(sampling_strategy=settings.SMOTE_OVER_RATIO, random_state=settings.RANDOM_STATE)),
            ('under', RandomUnderSampler(sampling_strategy=settings.SMOTE_UNDER_RATIO, random_state=settings.RANDOM_STATE)),
        ])
        x_res, y_res = pipe.fit_resample(x_train, y_train)
        counts = dict(zip(*np.unique(y_res, return_counts=True)))
        logger.info(f"After resampling → class counts: {counts}")
        return x_res, y_res

    # ─────────────────────────────────────────────────────────────────
    def find_optimal_threshold(self, model, x, y, min_precision=0.40):
        """
        Industry practice: do NOT use 0.5 as the decision threshold for
        imbalanced credit-risk data. Instead, sweep all precision-recall
        thresholds and pick the one that maximises F2 score.

        F2 (beta=2) weights recall twice as heavily as precision because
        in credit default prediction a missed defaulter (false negative)
        costs the bank significantly more than a wrongly flagged
        non-defaulter (false positive).
        """
        y_prob = model.predict_proba(x)[:, 1]
        precisions, recalls, thresholds = precision_recall_curve(y, y_prob)

        f2_scores = (5 * precisions * recalls) / (4 * precisions + recalls + 1e-9)

        # Mask out thresholds where precision is too low to be operationally useful
        valid_mask = precisions[:-1] >= min_precision
        f2_valid   = np.where(valid_mask, f2_scores[:-1], 0)

        if f2_valid.max() == 0:
            # Fallback: no threshold meets precision floor — relax and warn
            logger.warning(f"No threshold achieves precision >= {min_precision}. Falling back to best F2.")
            best_idx = np.argmax(f2_scores[:-1])
        else:
            best_idx = np.argmax(f2_valid)

        best_threshold = float(thresholds[best_idx])
        logger.info(
            f"Optimal threshold → {best_threshold:.4f} "
            f"(F2={f2_scores[best_idx]:.4f}, "
            f"P={precisions[best_idx]:.4f}, R={recalls[best_idx]:.4f})"
        )
        return best_threshold

    # ─────────────────────────────────────────────────────────────────
    def evaluate(self, model, x_test, y_test, threshold=0.5):
        """
        Evaluate at a custom threshold, not blindly at 0.5.
        Returns both threshold-dependent metrics and threshold-independent
        ranking metrics (ROC-AUC, PR-AUC).
        """
        y_prob = model.predict_proba(x_test)[:, 1]
        y_pred = (y_prob >= threshold).astype(int)   # ← apply optimal threshold

        return {
            "threshold":  round(threshold, 4),
            "accuracy":   round(accuracy_score(y_test, y_pred),             4),
            "precision":  round(precision_score(y_test, y_pred,             zero_division=0), 4),
            "recall":     round(recall_score(y_test, y_pred,                zero_division=0), 4),
            "f1_score":   round(f1_score(y_test, y_pred,                    zero_division=0), 4),
            # PRIMARY SELECTION METRIC — F2 rewards recall 2x over precision
            "f2_score":   round(fbeta_score(y_test, y_pred, beta=2,         zero_division=0), 4),
            "roc_auc":    round(roc_auc_score(y_test, y_prob),              4),
            # PR-AUC is threshold-independent and stricter than ROC-AUC for imbalanced data
            "pr_auc":     round(average_precision_score(y_test, y_prob),    4),
        }

    # ─────────────────────────────────────────────────────────────────
    def cross_validate_model(self, model, x, y, name):
        """
        5-fold stratified CV on the resampled training data.
        CV scores tell us whether a model is stable or just got lucky
        on a single train/test split — critical for reliable model selection.
        """
        cv = StratifiedKFold(n_splits=settings.CV_FOLDS, shuffle=True, random_state=settings.RANDOM_STATE)
        scores = cross_validate(
            model, x, y, cv=cv,
            scoring={"roc_auc": "roc_auc", "f2": make_scorer(fbeta_score, beta=2)},
            return_train_score=False,
            n_jobs=-1
        )
        mean_auc = round(float(np.mean(scores["test_roc_auc"])), 4)
        std_auc  = round(float(np.std(scores["test_roc_auc"])),  4)
        mean_f2  = round(float(np.mean(scores["test_f2"])),      4)

        logger.info(f"{name} CV → ROC-AUC: {mean_auc} ± {std_auc}  |  F2: {mean_f2}")
        return {"cv_roc_auc_mean": mean_auc, "cv_roc_auc_std": std_auc, "cv_f2_mean": mean_f2}

    # ─────────────────────────────────────────────────────────────────
    def composite_score(self, metrics, cv_metrics):
        """
        Industry-standard composite scoring for credit default selection.
        Weights rationale:
          - F2 (0.40): primary business metric — penalises missed defaulters
          - PR-AUC (0.25): threshold-independent, honest on imbalanced data
          - CV ROC-AUC (0.20): generalisation stability across folds
          - ROC-AUC (0.15): overall ranking quality on held-out test set

        Accuracy and precision are deliberately excluded from selection —
        a model that achieves 78% accuracy by ignoring defaulters is useless
        in a credit risk context.
        """
        return (
            settings.WEIGHT_F2 * metrics["f2_score"]       +
            settings.WEIGHT_PR_AUC * metrics["pr_auc"]         +
            settings.WEIGHT_CV_AUC * cv_metrics["cv_roc_auc_mean"] +
            settings.WEIGHT_ROC_AUC * metrics["roc_auc"]
        )

    # ─────────────────────────────────────────────────────────────────
    def train(self):
            try:
                logger.info("Starting Model Training")
                x_train, x_val, x_test, y_train, y_val, y_test = self.load_data()  # ← unpack val

                x_train_res, y_train_res = self.smote(x_train, y_train)  # ← SMOTE on train only

                spw = float((y_train_res == 0).sum() / max((y_train_res == 1).sum(), 1))
                logger.info(f"scale_pos_weight (post-resample) = {spw:.2f}")
                # ── Model definitions ─────────────────────────────────────
                models = {
                    "LogisticRegression": LogisticRegression(
                        max_iter=1000, class_weight="balanced",
                        solver="lbfgs", C=0.1,      # L2 regularisation; 0.1 prevents overfit on correlated PAY cols
                        random_state=42
                    ),
                    "RandomForest": RandomForestClassifier(
                        n_estimators=300,
                        class_weight="balanced",
                        max_depth=10,
                        min_samples_leaf=10,         # smooths decision boundary on 21K rows
                        random_state=42, n_jobs=-1   
                    ),
                    "LightGBM": LGBMClassifier(
                        n_estimators=500, learning_rate=0.05,
                        num_leaves=31, min_child_samples=30,
                        reg_alpha=0.1, reg_lambda=0.1,   # L1+L2 regularisation
                        is_unbalance=False,              # already resampled — don't double-count
                        verbose=-1, random_state=42
                    ),
                    "XGBoost": XGBClassifier(
                        n_estimators=500, learning_rate=0.05,
                        max_depth=6, min_child_weight=5,
                        subsample=0.8, colsample_bytree=0.8,  # row/col sampling = implicit regularisation
                        scale_pos_weight=spw,
                        eval_metric="aucpr",
                        verbosity=0, random_state=42
                    ),
                }

                mlflow.set_experiment(settings.MLFLOW_EXPERIMENT_NAME)
                all_results = {}

                for name, model in models.items():
                    with mlflow.start_run(run_name=name):
                        logger.info(f"Training {name}...")

                        cv_metrics = self.cross_validate_model(model, x_train_res, y_train_res, name)

                        model.fit(x_train_res, y_train_res)

                        calibrated = CalibratedClassifierCV(model, method="isotonic", cv="prefit")
                        calibrated.fit(x_train_res, y_train_res)

                        # ↓ threshold tuned on VAL set — never seen during training or SMOTE
                        threshold = self.find_optimal_threshold(calibrated, x_val, y_val)

                        # ↓ final metrics on TEST set — completely untouched until this line
                        metrics = self.evaluate(calibrated, x_test, y_test, threshold)

                        score = round(self.composite_score(metrics, cv_metrics), 4)

                        mlflow.log_params(model.get_params())
                        for k, v in {**metrics, **cv_metrics, "composite_score": score}.items():
                            mlflow.log_metric(k, v)
                        # serialization_format="cloudpickle": MLflow 3.x defaults to
                        # skops, which refuses to serialize CalibratedClassifierCV's
                        # internals ("untrusted types"). cloudpickle handles it and is
                        # backward-compatible with the older MLflow pinned in Docker.
                        mlflow.sklearn.log_model(calibrated, name, serialization_format="cloudpickle")

                        model_path = f"{self.model_dir}/{name}.pkl"
                        joblib.dump({"model": calibrated, "threshold": threshold}, model_path)

                        all_results[name] = {
                            "metrics":    metrics,
                            "cv_metrics": cv_metrics,
                            "composite":  score,
                            "model_path": model_path,
                            "threshold":  threshold,
                        }

                        logger.info(
                            f"{name} → F2: {metrics['f2_score']} | "
                            f"Recall: {metrics['recall']} | "
                            f"ROC-AUC: {metrics['roc_auc']} | "
                            f"CV-AUC: {cv_metrics['cv_roc_auc_mean']} ± {cv_metrics['cv_roc_auc_std']} | "
                            f"Composite: {score}"
                        )

                # ── Best model selection by COMPOSITE score ───────────────
                best_name = max(all_results, key=lambda n: all_results[n]["composite"])
                best      = all_results[best_name]

                logger.info(
                    f"Best model: {best_name} | "
                    f"Composite: {best['composite']} | "
                    f"F2: {best['metrics']['f2_score']} | "
                    f"Recall: {best['metrics']['recall']} | "
                    f"Threshold: {best['threshold']:.4f}"
                )

                # Save best model as canonical file for prediction pipeline
                best_payload = joblib.load(best["model_path"])
                joblib.dump(best_payload, f"{self.model_dir}/best_model.pkl")

                # Save full report for auditing
                report = {
                    "best_model":     best_name,
                    "composite_score": best["composite"],
                    "threshold":      best["threshold"],
                    "metrics":        best["metrics"],
                    "cv_metrics":     best["cv_metrics"],
                    "scoring_weights": {
                        "f2_score":        0.40,
                        "pr_auc":          0.25,
                        "cv_roc_auc_mean": 0.20,
                        "roc_auc":         0.15,
                    },
                    "all_results": {
                        n: {"metrics": v["metrics"], "cv": v["cv_metrics"], "composite": v["composite"]}
                        for n, v in all_results.items()
                    },
                }
                with open(f"{self.model_dir}/best_model_info.json", "w") as f:
                    json.dump(report, f, indent=4)

                logger.info("All models trained, calibrated, and saved ✅")

            except Exception as e:
                raise CustomException(e, sys)