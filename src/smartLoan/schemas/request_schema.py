# ─────────────────────────────────────────────────────────────────────────────
# schemas/request_schema.py
# Updated for UCI Default of Credit Card Clients dataset.
# ─────────────────────────────────────────────────────────────────────────────

from pydantic import BaseModel, Field
from typing import Literal


class CreditCardApplication(BaseModel):
    """
    Input schema for a single credit card client record.
    All fields map directly to UCI Credit Card dataset columns.
    """

    # ── Credit Limit ───────────────────────────────────────────────────────────
    LIMIT_BAL: float = Field(..., ge=10_000, le=1_000_000,
        description="Total credit limit in NT dollars (10,000 – 1,000,000)")

    # ── Demographics ───────────────────────────────────────────────────────────
    SEX:       Literal[1, 2]          = Field(..., description="1=Male, 2=Female")
    EDUCATION: Literal[0,1,2,3,4,5,6] = Field(..., description="1=Grad,2=Uni,3=HS,4=Other,0/5/6=Unknown")
    MARRIAGE:  Literal[0,1,2,3]       = Field(..., description="1=Married,2=Single,3=Other,0=Unknown")
    AGE:       int                    = Field(..., ge=18, le=100, description="Age in years")

    # ── Repayment Status ────────────────────────────────────────────────────────
    # -2=no use, -1=paid in full, 0=revolving credit, 1–9=months delayed
    # Note: there is no PAY_1 in the UCI dataset — this is not a typo.
    PAY_0: int = Field(..., ge=-2, le=9, description="Repayment status Sep 2005 (most recent)")
    PAY_2: int = Field(..., ge=-2, le=9, description="Repayment status Aug 2005")
    PAY_3: int = Field(..., ge=-2, le=9, description="Repayment status Jul 2005")
    PAY_4: int = Field(..., ge=-2, le=9, description="Repayment status Jun 2005")
    PAY_5: int = Field(..., ge=-2, le=9, description="Repayment status May 2005")
    PAY_6: int = Field(..., ge=-2, le=9, description="Repayment status Apr 2005 (oldest)")

    # ── Bill Statement Amounts (NT$) ────────────────────────────────────────────
    # Negative values valid — represent credit balance owed back to client
    BILL_AMT1: float = Field(..., description="Bill amount Sep 2005")
    BILL_AMT2: float = Field(..., description="Bill amount Aug 2005")
    BILL_AMT3: float = Field(..., description="Bill amount Jul 2005")
    BILL_AMT4: float = Field(..., description="Bill amount Jun 2005")
    BILL_AMT5: float = Field(..., description="Bill amount May 2005")
    BILL_AMT6: float = Field(..., description="Bill amount Apr 2005")

    # ── Payment Amounts (NT$) ───────────────────────────────────────────────────
    # Always >= 0 (cannot make a negative payment)
    PAY_AMT1: float = Field(..., ge=0, description="Amount paid Sep 2005")
    PAY_AMT2: float = Field(..., ge=0, description="Amount paid Aug 2005")
    PAY_AMT3: float = Field(..., ge=0, description="Amount paid Jul 2005")
    PAY_AMT4: float = Field(..., ge=0, description="Amount paid Jun 2005")
    PAY_AMT5: float = Field(..., ge=0, description="Amount paid May 2005")
    PAY_AMT6: float = Field(..., ge=0, description="Amount paid Apr 2005")

    model_config = {
        "json_schema_extra": {
            "example": {
                "LIMIT_BAL": 50000,
                "SEX": 2, "EDUCATION": 2, "MARRIAGE": 1, "AGE": 35,
                "PAY_0": -1, "PAY_2": -1, "PAY_3": -1,
                "PAY_4": -1, "PAY_5": -1, "PAY_6": -1,
                "BILL_AMT1": 20000, "BILL_AMT2": 19000, "BILL_AMT3": 18500,
                "BILL_AMT4": 17000, "BILL_AMT5": 16000, "BILL_AMT6": 15500,
                "PAY_AMT1": 19000, "PAY_AMT2": 17000, "PAY_AMT3": 18500,
                "PAY_AMT4": 17000, "PAY_AMT5": 16000, "PAY_AMT6": 15500,
            }
        }
    }


class PredictionResponse(BaseModel):
    prediction:             int   = Field(..., description="0=no default, 1=default")
    probability_of_default: float = Field(..., description="Calibrated default probability (0.0–1.0)")
    risk_label:             str   = Field(..., description="Low / Medium / High")
    model_used:             str   = Field(..., description="Model name")
    threshold_applied:      float = Field(..., description="Optimal decision threshold used")


class BatchPredictionResponse(BaseModel):
    count:           int
    predictions:     list
    elapsed_seconds: float


class HealthResponse(BaseModel):
    status:       str
    model_loaded: bool
    model_name:   str
    threshold:    float