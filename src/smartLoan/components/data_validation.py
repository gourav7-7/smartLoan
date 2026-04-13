import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os, sys, json

from smartLoan.config import paths
from smartLoan.utils.logger import logger
from smartLoan.utils.exceptions import CustomException


class DataValidation:
    def __init__(self):
        self.data_path = paths.RAW_DATA_DIR / "UCI_Credit_Card.csv"   # updated filename
        self.artifact_dir = "artifacts/eda"
        os.makedirs(self.artifact_dir, exist_ok=True)

    def run(self):
        try:
            logger.info("Starting Data Validation + EDA")

            # =========================
            # 1. LOAD DATA
            # =========================
            df = pd.read_csv(self.data_path)

            # Rename target column — dot in name causes issues downstream
            df.rename(columns={"default.payment.next.month": "TARGET"}, inplace=True)

            logger.info(f"Data loaded with shape: {df.shape}")

            errors = []
            insights = {}

            # =========================
            # 2. DATA VALIDATION
            # =========================

            # Required Columns
            required_columns = [
                "LIMIT_BAL",
                "AGE",
                "SEX",
                "EDUCATION",
                "TARGET"
            ]

            for col in required_columns:
                if col not in df.columns:
                    errors.append(f"Missing column: {col}")

            # Null Check
            null_counts = df[required_columns].isnull().sum().to_dict()
            insights["null_counts"] = null_counts

            # Range Check — credit limit: 10,000 to 1,000,000 NT dollars
            if not df["LIMIT_BAL"].between(10_000, 1_000_000).all():
                errors.append("LIMIT_BAL out of expected range (10,000 – 1,000,000)")

            # Range Check — age: 18 to 100
            if not df["AGE"].between(18, 100).all():
                errors.append("AGE out of expected range (18 – 100)")

            # Category Check — SEX: 1 = male, 2 = female
            if not df["SEX"].isin([1, 2]).all():
                errors.append("Invalid values in SEX (expected 1 or 2)")

            # Category Check — EDUCATION: 1=grad, 2=university, 3=high school, 4=others (0,5,6 are undocumented)
            if not df["EDUCATION"].isin([0, 1, 2, 3, 4, 5, 6]).all():
                errors.append("Invalid values in EDUCATION")

            # Category Check — MARRIAGE: 0=unknown, 1=married, 2=single, 3=others
            if not df["MARRIAGE"].isin([0, 1, 2, 3]).all():
                errors.append("Invalid values in MARRIAGE")

            # Target Check — must be binary
            if not df["TARGET"].isin([0, 1]).all():
                errors.append("TARGET contains values other than 0 and 1")

            # =========================
            # 3. BASIC EDA
            # =========================

            insights["shape"] = df.shape
            insights["columns"] = list(df.columns)

            # ---- Missing Values ----
            missing = df.isnull().sum().sort_values(ascending=False)
            missing = missing[missing > 0]
            insights["missing_top"] = missing.head(20).to_dict()

            if len(missing) > 0:
                plt.figure(figsize=(10, 5))
                missing.head(20).plot(kind="bar")
                plt.title("Top Missing Values")
                plt.tight_layout()
                plt.savefig(f"{self.artifact_dir}/missing_values.png")
                plt.close()

            # =========================
            # 4. UNIVARIATE ANALYSIS
            # =========================

            # ---- LIMIT_BAL Distribution ----
            plt.figure(figsize=(6, 4))
            sns.histplot(df["LIMIT_BAL"], bins=50, kde=True)
            plt.title("Credit Limit (LIMIT_BAL) Distribution")
            plt.savefig(f"{self.artifact_dir}/limit_bal_dist.png")
            plt.close()

            # ---- LIMIT_BAL Boxplot (Outliers) ----
            plt.figure(figsize=(6, 4))
            sns.boxplot(x=df["LIMIT_BAL"])
            plt.title("Credit Limit (LIMIT_BAL) Outliers")
            plt.savefig(f"{self.artifact_dir}/limit_bal_box.png")
            plt.close()

            # ---- AGE Distribution ----
            plt.figure(figsize=(6, 4))
            sns.histplot(df["AGE"], bins=30, kde=True)
            plt.title("Age Distribution")
            plt.savefig(f"{self.artifact_dir}/age_dist.png")
            plt.close()

            # ---- SEX Count ----
            plt.figure(figsize=(5, 4))
            sns.countplot(x=df["SEX"])
            plt.xticks([0, 1], ["Male (1)", "Female (2)"])
            plt.title("Gender Distribution")
            plt.savefig(f"{self.artifact_dir}/sex_count.png")
            plt.close()

            # ---- EDUCATION Count ----
            plt.figure(figsize=(6, 4))
            sns.countplot(x=df["EDUCATION"])
            plt.title("Education Level Distribution")
            plt.savefig(f"{self.artifact_dir}/education_count.png")
            plt.close()

            # ---- Target Class Balance ----
            plt.figure(figsize=(5, 4))
            sns.countplot(x=df["TARGET"])
            plt.xticks([0, 1], ["No Default (0)", "Default (1)"])
            plt.title("Target Class Balance")
            plt.savefig(f"{self.artifact_dir}/target_balance.png")
            plt.close()

            # =========================
            # 5. BIVARIATE ANALYSIS (WITH TARGET)
            # =========================

            # ---- LIMIT_BAL vs Target ----
            plt.figure(figsize=(6, 4))
            sns.boxplot(x="TARGET", y="LIMIT_BAL", data=df)
            plt.title("Credit Limit vs Default")
            plt.savefig(f"{self.artifact_dir}/limit_bal_vs_target.png")
            plt.close()

            # ---- AGE vs Target ----
            plt.figure(figsize=(6, 4))
            sns.boxplot(x="TARGET", y="AGE", data=df)
            plt.title("Age vs Default")
            plt.savefig(f"{self.artifact_dir}/age_vs_target.png")
            plt.close()

            # ---- SEX vs Target ----
            plt.figure(figsize=(6, 4))
            sns.countplot(x="SEX", hue="TARGET", data=df)
            plt.title("Gender vs Default")
            plt.savefig(f"{self.artifact_dir}/sex_vs_target.png")
            plt.close()

            # ---- EDUCATION vs Target ----
            plt.figure(figsize=(6, 4))
            sns.countplot(x="EDUCATION", hue="TARGET", data=df)
            plt.title("Education vs Default")
            plt.savefig(f"{self.artifact_dir}/education_vs_target.png")
            plt.close()

            # ---- MARRIAGE vs Target ----
            plt.figure(figsize=(6, 4))
            sns.countplot(x="MARRIAGE", hue="TARGET", data=df)
            plt.title("Marital Status vs Default")
            plt.savefig(f"{self.artifact_dir}/marriage_vs_target.png")
            plt.close()

            # =========================
            # 6. CORRELATION ANALYSIS
            # =========================

            corr = df.corr(numeric_only=True)

            plt.figure(figsize=(14, 12))
            sns.heatmap(corr, cmap="coolwarm", annot=False)
            plt.title("Correlation Heatmap")
            plt.tight_layout()
            plt.savefig(f"{self.artifact_dir}/correlation.png")
            plt.close()

            # =========================
            # 7. INSIGHT GENERATION
            # =========================

            # Skewness check on LIMIT_BAL
            skewness = df["LIMIT_BAL"].skew()
            insights["limit_bal_skewness"] = float(skewness)
            if skewness > 1:
                insights["recommendation_limit_bal"] = "Apply log transformation on LIMIT_BAL"

            # Class imbalance check
            default_rate = df["TARGET"].mean()
            insights["default_rate"] = float(default_rate)
            if default_rate < 0.3:
                insights["recommendation_imbalance"] = (
                    "Dataset is imbalanced — consider SMOTE or class_weight='balanced'"
                )

            # Missing percent
            missing_percent = (df.isnull().mean() * 100).sort_values(ascending=False)
            insights["missing_percent"] = missing_percent.head(10).to_dict()

            # Correlation with TARGET
            target_corr = corr["TARGET"].drop("TARGET").abs().sort_values(ascending=False)
            insights["top_correlated_features"] = target_corr.head(10).to_dict()

            # Feature engineering suggestions
            insights["feature_suggestions"] = [
                "UTIL_RATIO = mean(BILL_AMT1..6) / LIMIT_BAL  — credit utilisation",
                "PAY_TREND = PAY_0 - PAY_6                     — payment behaviour trend",
                "AVG_PAY_AMT = mean(PAY_AMT1..6)               — average repayment amount"
            ]

            # =========================
            # 8. SAVE REPORT
            # =========================

            report = {
                "validation_success": len(errors) == 0,
                "errors": errors,
                "insights": insights
            }

            with open("artifacts/eda_report.json", "w") as f:
                json.dump(report, f, indent=4)

            logger.info("EDA report saved")

            # =========================
            # 9. HANDLE FAILURE
            # =========================

            if errors:
                logger.error(f"Validation FAILED ❌: {errors}")
                raise Exception("Data validation failed")

            logger.info("Validation + EDA completed successfully ✅")

        except Exception as e:
            raise CustomException(e, sys)