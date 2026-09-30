
# EXI_LOAN_PREDECTION

PBL-2: Explainable AI for Loan Prediction

## Simple application flow

Welcome
→ Get Started
→ Customer + Loan Details
→ AI Prediction
→ Approved / Rejected
→ SHAP explanation
→ Documents (if approved)
→ Try Again / Thank You

The original training dataset is kept separate from `customer_applications.csv`.
Every submitted customer prediction is stored in that application log. The result
page shows only the approved/rejected ratio of submitted applications.

## Run locally

```bash
pip install -r requirements.txt
python app.py
```

Open http://127.0.0.1:5000

## Optional: regenerate the training dataset

```bash
python generate_dataset.py
```

## Public deployment

Start command for Render or another Gunicorn host:

```bash
gunicorn app:app
```

Build command:

```bash
pip install -r requirements.txt
```

Educational PBL demonstration only. This is not a real bank underwriting system.
