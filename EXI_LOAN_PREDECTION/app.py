
from pathlib import Path
import json
import os

import numpy as np
import pandas as pd
import shap
from flask import Flask, render_template, request, redirect, url_for
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

BASE = Path(__file__).resolve().parent
TRAINING_FILE = BASE / "loan_prediction_dataset.csv"
APPLICATION_FILE = BASE / "customer_applications.csv"

app = Flask(__name__)

FEATURES = [
    "Dependents",
    "Education",
    "Self_Employed",
    "Applicant_Income",
    "Loan_Amount",
    "Loan_Term_Months",
    "CIBIL_Score",
    "Residential_Assets",
    "Commercial_Assets",
    "Luxury_Assets",
    "Bank_Assets",
]

LABELS = {
    "Dependents": "Dependents",
    "Education": "Education",
    "Self_Employed": "Employment Status",
    "Applicant_Income": "Applicant Income",
    "Loan_Amount": "Loan Amount",
    "Loan_Term_Months": "Loan Term",
    "CIBIL_Score": "CIBIL Score",
    "Residential_Assets": "Residential Assets",
    "Commercial_Assets": "Commercial Assets",
    "Luxury_Assets": "Luxury Assets",
    "Bank_Assets": "Bank Assets",
}

def load_training_data():
    df = pd.read_csv(TRAINING_FILE)
    df = df.drop_duplicates().copy()
    for col in FEATURES + ["Loan_Approved"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=FEATURES + ["Loan_Approved"]).reset_index(drop=True)
    return df

df = load_training_data()
X = df[FEATURES]
y = df["Loan_Approved"].astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

model = RandomForestClassifier(
    n_estimators=250,
    max_depth=9,
    min_samples_leaf=2,
    random_state=42
)
model.fit(X_train, y_train)
accuracy = accuracy_score(y_test, model.predict(X_test))
explainer = shap.TreeExplainer(model)

def ensure_application_file():
    if not APPLICATION_FILE.exists():
        cols = [
            "Full_Name", "Age", "Gender", "Phone", "Email", "City", "Address",
            *FEATURES, "Prediction", "Probability"
        ]
        pd.DataFrame(columns=cols).to_csv(APPLICATION_FILE, index=False)

ensure_application_file()

def shap_contributions(row):
    raw = explainer.shap_values(row)
    if isinstance(raw, list):
        arr = np.asarray(raw[1] if len(raw) > 1 else raw[0])
        return arr.reshape(-1)[:len(FEATURES)].astype(float)

    arr = np.asarray(raw)
    if arr.ndim == 3:
        return arr[0, :, 1].astype(float)
    if arr.ndim == 2:
        if arr.shape == (1, len(FEATURES)):
            return arr[0].astype(float)
        if arr.shape[0] == len(FEATURES):
            return arr[:, 1 if arr.shape[1] > 1 else 0].astype(float)
        return arr.reshape(-1)[:len(FEATURES)].astype(float)

    return arr.reshape(-1)[:len(FEATURES)].astype(float)

def friendly(feature, value):
    if feature == "Education":
        return "Graduate" if int(value) == 1 else "Not Graduate"
    if feature == "Self_Employed":
        return "Self-employed" if int(value) == 1 else "Salaried / Not self-employed"
    if feature == "Loan_Term_Months":
        return f"{int(value)} months"
    if feature in {
        "Applicant_Income", "Loan_Amount", "Residential_Assets",
        "Commercial_Assets", "Luxury_Assets", "Bank_Assets"
    }:
        return f"₹{float(value):,.0f}"
    return f"{float(value):.0f}"

def application_counts():
    ensure_application_file()
    apps = pd.read_csv(APPLICATION_FILE)
    if apps.empty:
        return {"approved": 0, "rejected": 0, "total": 0, "approved_pct": 0, "rejected_pct": 0}
    approved = int((apps["Prediction"] == "Approved").sum())
    rejected = int((apps["Prediction"] == "Rejected").sum())
    total = approved + rejected
    return {
        "approved": approved,
        "rejected": rejected,
        "total": total,
        "approved_pct": round(approved / total * 100, 1) if total else 0,
        "rejected_pct": round(rejected / total * 100, 1) if total else 0,
    }

def blank_form_values():
    return {
        "Dependents": "2",
        "Education": "1",
        "Self_Employed": "0",
        "Applicant_Income": "85000",
        "Loan_Amount": "250000",
        "Loan_Term_Value": "3",
        "Loan_Term_Unit": "years",
        "CIBIL_Score": "720",
        "Residential_Assets": "300000",
        "Commercial_Assets": "100000",
        "Luxury_Assets": "80000",
        "Bank_Assets": "150000",
    }

@app.route("/")
def welcome():
    return render_template("welcome.html")

@app.route("/predict", methods=["GET", "POST"])
def predict():
    if request.method == "GET":
        return render_template("predict.html", values=blank_form_values())

    try:
        customer = {
            "Full_Name": request.form.get("Full_Name", "").strip(),
            "Age": request.form.get("Age", "").strip(),
            "Gender": request.form.get("Gender", "Not specified").strip(),
            "Phone": request.form.get("Phone", "").strip(),
            "Email": request.form.get("Email", "").strip(),
            "City": request.form.get("City", "").strip(),
            "Address": request.form.get("Address", "").strip(),
        }

        if not customer["Full_Name"]:
            raise ValueError("Please enter the customer's full name.")
        age = int(customer["Age"])
        if not 18 <= age <= 100:
            raise ValueError("Age must be between 18 and 100.")
        if not customer["Phone"]:
            raise ValueError("Please enter a phone number.")

        values = {}
        for name in [
            "Dependents", "Applicant_Income", "Loan_Amount", "CIBIL_Score",
            "Residential_Assets", "Commercial_Assets", "Luxury_Assets", "Bank_Assets"
        ]:
            raw = request.form.get(name, "").strip()
            if raw == "":
                raise ValueError("Please complete all loan and financial details.")
            values[name] = float(raw)

        values["Education"] = float(request.form.get("Education", ""))
        values["Self_Employed"] = float(request.form.get("Self_Employed", ""))

        term_value = float(request.form.get("Loan_Term_Value", ""))
        term_unit = request.form.get("Loan_Term_Unit", "years")
        if term_value <= 0:
            raise ValueError("Loan term must be greater than zero.")
        term_months = term_value * 12 if term_unit == "years" else term_value
        values["Loan_Term_Months"] = float(term_months)

        if not 300 <= values["CIBIL_Score"] <= 900:
            raise ValueError("CIBIL score must be between 300 and 900.")
        if values["Loan_Amount"] <= 0 or values["Applicant_Income"] <= 0:
            raise ValueError("Income and loan amount must be greater than zero.")

        row = pd.DataFrame([values], columns=FEATURES)
        prediction_num = int(model.predict(row)[0])
        probability = float(model.predict_proba(row)[0, 1] * 100)

        impacts = shap_contributions(row)
        explanation = []
        for feature, value, impact in zip(FEATURES, row.iloc[0].values, impacts):
            explanation.append({
                "feature": LABELS[feature],
                "value": friendly(feature, value),
                "impact": round(float(impact), 5),
                "direction": "positive" if impact >= 0 else "negative",
            })
        explanation.sort(key=lambda x: abs(x["impact"]), reverse=True)

        if prediction_num == 1:
            reasons = [x for x in explanation if x["impact"] >= 0][:4]
            prediction = "Approved"
        else:
            reasons = [x for x in explanation if x["impact"] < 0][:4]
            prediction = "Rejected"

        if not reasons:
            reasons = explanation[:4]

        # Save this customer's prediction in a separate application log.
        # The original training dataset is never modified.
        ensure_application_file()
        applications = pd.read_csv(APPLICATION_FILE)
        new_row = {
            **customer,
            **values,
            "Prediction": prediction,
            "Probability": round(probability, 2),
        }
        applications = pd.concat([applications, pd.DataFrame([new_row])], ignore_index=True)
        applications.to_csv(APPLICATION_FILE, index=False)

        counts = application_counts()

        chart_max = max(
            [abs(x["impact"]) for x in explanation[:6]] + [0.001]
        )

        return render_template(
            "result.html",
            customer=customer,
            prediction=prediction,
            probability=round(probability, 2),
            reasons=reasons,
            explanation=explanation[:6],
            counts=counts,
            chart_max=chart_max,
            model_accuracy=round(accuracy * 100, 2),
        )

    except Exception as exc:
        return render_template(
            "predict.html",
            values=request.form.to_dict(),
            error=str(exc)
        )

@app.route("/thank-you")
def thank_you():
    return render_template("thank_you.html")

if __name__ == "__main__":
    print("LoanXAI PBL-2")
    print(f"Training rows: {len(df)}")
    print(f"Model test accuracy: {accuracy * 100:.2f}%")
    print("Open: http://127.0.0.1:5000")
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=False)
