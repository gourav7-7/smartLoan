from smartLoan.components.model_prediction import ModelPrediction
from smartLoan.utils.logger import logger


class ModelPredictionStage:
    def __init__(self):
        self.predictor = ModelPrediction()

    def main(self, input_data: dict) -> dict:
        result = self.predictor.predict(input_data)
        return result


if __name__ == "__main__":
    # ── LOW RISK sample — healthy customer ────────────────────────────────────
    # All payments on time, paying nearly full bill each month,
    # declining balances. Expected: prediction=0, risk=Low
    low_risk_input = {
        # ── Credit Limit ──────────────────────────────────────────────────────
        # Total credit limit given to this customer by the bank, in NT dollars.
        # Typical range: 10,000 – 1,000,000
        # Higher limit generally means the bank trusts this customer more.
        # Example: 50000 means NT$50,000 credit limit
        "LIMIT_BAL": 50000,

        # ── Gender ────────────────────────────────────────────────────────────
        # 1 = Male
        # 2 = Female
        "SEX": 2,

        # ── Education Level ───────────────────────────────────────────────────
        # Highest education level completed by the customer.
        # 1 = Graduate school (postgraduate)
        # 2 = University (undergraduate)
        # 3 = High school
        # 4 = Others (vocational, primary, etc.)
        "EDUCATION": 2,

        # ── Marital Status ────────────────────────────────────────────────────
        # 1 = Married
        # 2 = Single
        # 3 = Others (divorced, widowed, etc.)
        "MARRIAGE": 1,

        # ── Age ───────────────────────────────────────────────────────────────
        # Customer's age in years at the time of the record.
        # Typical range: 21 – 79
        "AGE": 35,

         # ── Repayment Status (Most Recent 6 Months) ───────────────────────────
        # These are the most important features for default prediction.
        # Each PAY column represents the repayment status for a given month.
        # PAY_0 = September (most recent), PAY_2 = August, PAY_3 = July,
        # PAY_4 = June, PAY_5 = May, PAY_6 = April (oldest)
        # NOTE: There is no PAY_1 in the UCI dataset — this is not a typo.
        #
        # Allowed values:
        # -2 = No consumption (no bill that month)
        # -1 = Paid in full on time
        #  0 = Use of revolving credit (minimum payment made)
        #  1 = Payment delayed by 1 month
        #  2 = Payment delayed by 2 months
        #  3 = Payment delayed by 3 months
        #  ... and so on up to 9 months delay
        #
        # Higher positive values = more serious delinquency = higher default risk
        "PAY_0": -1,   # September: paid in full on time
        "PAY_2": -1,   # August:    paid in full on time
        "PAY_3": -1,   # July:      paid in full on time
        "PAY_4": -1,   # June:      paid in full on time
        "PAY_5": -1,   # May:       paid in full on time
        "PAY_6": -1,   # April:     paid in full on time

        # ── Bill Statement Amounts ────────────────────────────────────────────
        # The outstanding bill amount on the customer's credit card statement
        # for each month, in NT dollars. This is how much they OWED, not paid.
        # BILL_AMT1 = September (most recent), BILL_AMT6 = April (oldest)
        #
        # Typical range: 0 – 1,000,000 NT dollars
        # Negative values are possible (overpayment / credit balance)
        # Rising bill amounts over time can signal growing financial stress.
        "BILL_AMT1": 20000,   # September: NT$20,000 owed
        "BILL_AMT2": 19000,   # August:    NT$19,000 owed
        "BILL_AMT3": 18500,   # July:      NT$18,500 owed
        "BILL_AMT4": 17000,   # June:      NT$17,000 owed
        "BILL_AMT5": 16000,   # May:       NT$16,000 owed
        "BILL_AMT6": 15500,   # April:     NT$15,500 owed

        # ── Previous Payment Amounts ──────────────────────────────────────────
        # The actual amount the customer PAID toward their bill each month,
        # in NT dollars. PAY_AMT1 = payment made in September for August's bill,
        # PAY_AMT2 = payment made in August for July's bill, and so on.
        #
        # Typical range: 0 – 1,000,000 NT dollars (always 0 or positive)
        # 0 = no payment made that month (high risk signal)
        # Very low payments relative to BILL_AMT = customer is barely keeping up
        # Payments close to or exceeding BILL_AMT = financially healthy customer
        "PAY_AMT2": 17000,   # Paid NT$17,000 in August
        "PAY_AMT3": 18500,   # Paid NT$18,500 in July (full bill)
        "PAY_AMT4": 17000,   # Paid NT$17,000 in June
        "PAY_AMT5": 16000,   # Paid NT$16,000 in May (full bill)
        "PAY_AMT6": 15500,   # Paid NT$15,500 in April (full bill)
    }

    # ── HIGH RISK sample — delinquent customer ────────────────────────────────
    # Multiple months overdue, zero payments, maxed out credit.
    # Expected: prediction=1, risk=High
    high_risk_input = {
        "LIMIT_BAL": 50000,
        "SEX": 1,
        "EDUCATION": 3,
        "MARRIAGE": 2,
        "AGE": 28,

        "PAY_0":  3,   # September: 3 months overdue
        "PAY_2":  2,   # August:    2 months overdue
        "PAY_3":  2,   # July:      2 months overdue
        "PAY_4":  1,   # June:      1 month overdue
        "PAY_5":  1,   # May:       1 month overdue
        "PAY_6":  0,   # April:     revolving (started slipping)

        "BILL_AMT1": 49000,   # Nearly maxed out credit limit
        "BILL_AMT2": 47000,
        "BILL_AMT3": 45000,
        "BILL_AMT4": 43000,
        "BILL_AMT5": 40000,
        "BILL_AMT6": 37000,

        "PAY_AMT1":    0,   # No payment made
        "PAY_AMT2":    0,   # No payment made
        "PAY_AMT3":  500,   # Token payment only
        "PAY_AMT4":  500,
        "PAY_AMT5": 1000,
        "PAY_AMT6": 1500,
    }

    try:
        logger.info(">>>>>>>>>>>>>>>> Prediction started <<<<<<<<<<<<<<<")
        stage = ModelPredictionStage()

        logger.info("--- Low Risk Customer ---")
        result_low = stage.main(low_risk_input)
        logger.info(f"Prediction result: {result_low}")

        logger.info("--- High Risk Customer ---")
        result_high = stage.main(high_risk_input)
        logger.info(f"Prediction result: {result_high}")

        logger.info(">>>>>>>>>>>>>>>> Prediction completed <<<<<<<<<<<<<<<")
    except Exception as e:
        logger.exception(e)
        raise e