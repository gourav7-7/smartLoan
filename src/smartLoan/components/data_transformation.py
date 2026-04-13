import pandas as pd
import numpy as np
import os, sys
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from smartLoan.config import paths
from smartLoan.utils.logger import logger
from smartLoan.utils.exceptions import CustomException


class DataTransformation:
    def __init__(self):
        self.data_path = paths.RAW_DATA_DIR / "UCI_Credit_Card.csv"   
        self.artifact_dir = "artifacts/processed"
        os.makedirs(self.artifact_dir, exist_ok=True)

    def transform(self):
        try:
            logger.info("Starting Data Transformation")
            df = pd.read_csv(self.data_path)

            # Rename target — dot in name causes issues downstream
            df.rename(columns={"default.payment.next.month": "TARGET"}, inplace=True)

            # =========================
            # 1. DROP IRRELEVANT COLUMNS
            # =========================
            # ID is a row index, not a feature
            df.drop(columns=["ID"], inplace=True)
            logger.info("Dropped column: ID")

            

            # =========================
            # 2. FEATURE ENGINEERING
            # =========================
            bill_cols   = ["BILL_AMT1", "BILL_AMT2", "BILL_AMT3",
                           "BILL_AMT4", "BILL_AMT5", "BILL_AMT6"]
            pay_cols    = ["PAY_AMT1",  "PAY_AMT2",  "PAY_AMT3",
                           "PAY_AMT4",  "PAY_AMT5",  "PAY_AMT6"]

            # Credit utilisation — how much of the limit is being billed on average
            df["UTIL_RATIO"] = df[bill_cols].mean(axis=1) / (df["LIMIT_BAL"] + 1)

            # Payment behaviour trend — positive = improving, negative = worsening
            df["PAY_TREND"] = df["PAY_0"] - df["PAY_6"]

            # Average monthly repayment amount
            df["AVG_PAY_AMT"] = df[pay_cols].mean(axis=1)

            # Average monthly bill amount (kept separate from UTIL_RATIO)
            df["AVG_BILL_AMT"] = df[bill_cols].mean(axis=1)

            logger.info("Feature engineering completed: UTIL_RATIO, PAY_TREND, AVG_PAY_AMT, AVG_BILL_AMT")

            # =========================
            # 3. HANDLE INF VALUES
            # =========================
            df.replace([np.inf, -np.inf], np.nan, inplace=True)

            # =========================
            # 4. HANDLE MISSING VALUES (safety net — dataset is clean but
            #    engineered features could introduce NaN via edge cases)
            # =========================
            num_cols = df.select_dtypes(include=np.number).columns
            for col in num_cols:
                if df[col].isnull().any():
                    df[col] = df[col].fillna(df[col].median())

            # =========================
            # 5. HANDLE SKEWNESS (selective log1p)
            # =========================
            # PAY_AMT cols are heavily skewed (skew > 10) AND always >= 0 → safe for log1p
            # BILL_AMT cols are skewed but contain NEGATIVE values → skip log, scale only
            skew_safe_cols = pay_cols + ["AVG_PAY_AMT"]

            for col in skew_safe_cols:
                if (df[col] >= 0).all():
                    df[col] = np.log1p(df[col])

            logger.info(f"Log1p applied to: {skew_safe_cols}")

            # Clean again after log transformation
            df.replace([np.inf, -np.inf], np.nan, inplace=True)
            for col in num_cols:
                if df[col].isnull().any():
                    df[col] = df[col].fillna(df[col].median())

            # =========================
            # 6. SPLIT DATA
            # =========================
            assert "TARGET" in df.columns, "TARGET column missing!"

            x = df.drop("TARGET", axis=1)
            y = df["TARGET"]

            # Stratify preserves the 78/22 class split across train and test
            x_train, x_test, y_train, y_test = train_test_split(
                x, y,
                test_size=0.2,
                random_state=42,
                stratify=y
            )

            logger.info(f"Train size: {x_train.shape}, Test size: {x_test.shape}")
            logger.info(f"Default rate — train: {y_train.mean():.3f}, test: {y_test.mean():.3f}")

            # =========================
            # 7. FINAL SAFETY CHECK
            # =========================
            assert not np.isinf(x_train.values).any(), "Infinity found in X_train!"
            assert not np.isnan(x_train.values).any(), "NaN found in X_train!"

            # =========================
            # 8. SCALING
            # =========================
            scaler = StandardScaler()

            x_train_scaled = scaler.fit_transform(x_train)
            x_test_scaled  = scaler.transform(x_test)

            # =========================
            # 9. SAVE FILES
            # =========================
            feature_columns = x_train.columns.tolist()

            pd.DataFrame(x_train_scaled, columns=feature_columns).to_csv(
                f"{self.artifact_dir}/x_train.csv", index=False
            )
            pd.DataFrame(x_test_scaled, columns=feature_columns).to_csv(
                f"{self.artifact_dir}/x_test.csv", index=False
            )
            y_train.to_csv(f"{self.artifact_dir}/y_train.csv", index=False)
            y_test.to_csv(f"{self.artifact_dir}/y_test.csv", index=False)

            # Save scaler for prediction pipeline
            scaler_path = f"{self.artifact_dir}/scaler.pkl"
            joblib.dump(scaler, scaler_path)
            logger.info(f"Scaler saved → {scaler_path}")

            # Save column names for prediction pipeline alignment
            col_path = f"{self.artifact_dir}/training_columns.txt"
            with open(col_path, "w") as f:
                f.write("\n".join(feature_columns))
            logger.info(f"Training columns saved → {col_path} ({len(feature_columns)} features)")

            logger.info("Data Transformation Completed ✅")

        except Exception as e:
            raise CustomException(e, sys)