
from pathlib import Path
import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
n = 700

df = pd.DataFrame({
    "Dependents": rng.integers(0, 5, n),
    "Education": rng.integers(0, 2, n),
    "Self_Employed": rng.integers(0, 2, n),
    "Applicant_Income": rng.integers(18000, 180000, n),
    "Loan_Amount": rng.integers(50000, 900000, n),
    "Loan_Term_Months": rng.choice([12, 24, 36, 48, 60], n),
    "CIBIL_Score": rng.integers(350, 851, n),
    "Residential_Assets": rng.integers(0, 1500000, n),
    "Commercial_Assets": rng.integers(0, 900000, n),
    "Luxury_Assets": rng.integers(0, 700000, n),
    "Bank_Assets": rng.integers(5000, 800000, n),
})

# Synthetic educational target with a clear, explainable relationship.
score = (
    0.018 * (df["CIBIL_Score"] - 600)
    + 0.000010 * df["Applicant_Income"]
    - 0.0000022 * df["Loan_Amount"]
    - 0.00035 * df["Dependents"]
    + 0.25 * df["Education"]
    - 0.22 * df["Self_Employed"]
    + 0.00000035 * df["Residential_Assets"]
    + 0.00000025 * df["Bank_Assets"]
    - 0.003 * (df["Loan_Term_Months"] - 36)
    + rng.normal(0, 1.15, n)
)
df["Loan_Approved"] = (score > 1.0).astype(int)

# Ensure both classes are present and balanced enough for training.
if df["Loan_Approved"].mean() < 0.2 or df["Loan_Approved"].mean() > 0.8:
    threshold = score.median()
    df["Loan_Approved"] = (score >= threshold).astype(int)

# A few duplicate rows make cleaning demonstrable.
df = pd.concat([df, df.iloc[:5]], ignore_index=True)
df.to_csv(Path(__file__).resolve().parent / "loan_prediction_dataset.csv", index=False)
print("Created loan_prediction_dataset.csv:", len(df), "rows")
