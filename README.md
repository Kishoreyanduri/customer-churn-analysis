# Customer Churn Early Warning System

A machine learning pipeline that predicts customer churn risk on telecom data, designed to give a retention team actionable, prioritised output without requiring them to interpret model internals.

---

## Problem Statement

Telecom companies lose significant revenue to customer churn. Identifying at-risk customers early — before they cancel — allows retention teams to intervene with targeted offers. The challenge is building a model that is both accurate and interpretable enough for a non-technical team to act on.

---

## Dataset

- 10,000+ customer records from a telecom dataset
- Features include contract type, tenure, monthly charges, service subscriptions, payment method, and support ticket history
- Target variable: Churn (binary — Yes/No)
- Class imbalance handled during preprocessing

---

## Tech Stack

| Layer | Tools |
|---|---|
| Language | Python 3.x |
| Data Processing | Pandas, NumPy |
| Visualisation | Matplotlib, Seaborn |
| Modelling | Scikit-learn |
| Environment | Jupyter Notebook |

---

## Feature Engineering

12 features were engineered beyond the raw columns:

- Tenure buckets — grouped tenure into Early (0–12 months), Mid (13–36 months), Long-term (37+ months)
- Service usage ratio — number of active services divided by total available services
- Contract-type flags — one-hot encoded month-to-month, one-year, two-year contracts
- Charge per service — monthly charges normalised by number of active services
- Support interaction flag — binary flag for customers with 3+ support tickets
- Additional interaction terms between tenure and monthly charges

---

## Models Trained

| Model | Accuracy | Notes |
|---|---|---|
| Logistic Regression | 84% | Baseline, high interpretability |
| Random Forest | 88% | Best overall performance |

Both models were evaluated using:

- Accuracy
- Precision, Recall, F1-score
- Confusion Matrix
- ROC-AUC Curve

---

## Pipeline

Raw CSV
--  └── Data Cleaning (nulls, type fixes, duplicates)
  --      └── Exploratory Data Analysis
    --          └── Feature Engineering
      --              └── Train/Test Split (80/20, stratified)
        --                  └── Model Training
          --                      └── Evaluation
            --                          └── Risk-Segmented Output Report

---

## Output

The final output is a risk-segmented CSV with three customer tiers:

- High Risk — churn probability > 0.70
- Medium Risk — churn probability 0.40–0.70
- Low Risk — churn probability < 0.40

This format lets a retention team prioritise outreach without reading model code or interpreting probability scores directly.

---

## Key Findings

- Month-to-month contract customers churn at 3x the rate of annual contract customers
- Customers in the first 12 months (Early tenure bucket) are the highest-risk segment
- High monthly charges combined with low service usage is the strongest churn signal

---

## Project Structure

customer-churn-analysis/
│
├── data/
│   └── telco_churn.csv
├── notebooks/
│   └── churn_analysis.ipynb
├── output/
│   └── churn_risk_segments.csv
└── README.md

---

## How to Run

-- git clone https://github.com/Kishoreyanduri/customer-churn-analysis
-- cd customer-churn-analysis
-- pip install pandas numpy matplotlib seaborn scikit-learn
-- jupyter notebook notebooks/churn_analysis.ipynb
