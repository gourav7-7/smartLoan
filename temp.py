import mlflow
import pandas as pd

# 🔹 Set your MLflow tracking URI (local server)
mlflow.set_tracking_uri("http://127.0.0.1:5000")

# 🔹 Define your experiment name
EXPERIMENT_NAME = "SmartLoan_UCI_CreditDefault"

# 🔹 Fetch all runs as a DataFrame
df = mlflow.search_runs(
    experiment_names=[EXPERIMENT_NAME]
)

# 🔹 Optional: Keep only useful columns (clean table)
# This filters only metrics + params + run_id
cols = [col for col in df.columns if "metrics." in col or "params." in col or col == "run_id"]
df_clean = df[cols]

# 🔹 Save to CSV
output_file = "mlflow_metrics.csv"
df_clean.to_csv(output_file, index=False)

print(f"✅ Metrics table saved to {output_file}")