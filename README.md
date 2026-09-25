# Customer Churn Analysis & Risk Scoring

An end-to-end machine learning project designed to analyze customer behavior, predict churn risk, and calculate retention campaign ROI using Python, scikit-learn, and pandas.

---

## 📌 Project Overview

Customer churn is a critical metric for subscription-based businesses. This project builds a machine learning pipeline that identifies high-risk customers, assigns churn probability scores, and categorizes users into actionable risk tiers (Low, Medium, High). 

By optimizing the decision threshold for recall, the model maximizes the capture of at-risk users, allowing business teams to implement targeted retention strategies efficiently.

---

## 🛠️ Key Features

* **Data Simulation & Feature Engineering:** Generates a realistic customer dataset containing demographics, tenure, contract types, support interaction, and payment delays.
* **Exploratory Data Analysis (EDA):** Visualizes key relationships between customer attributes and churn behavior using `seaborn` and `matplotlib`.
* **Model Training & Comparison:** Evaluates Logistic Regression and Random Forest algorithms using scaled features and class-weight balancing.
* **Custom Threshold Optimization:** Adjusts decision thresholds (e.g., set to `0.40`) to prioritize **Recall (86.2%)**, catching the majority of churning customers.
* **Risk Scoring Framework:** Categorizes customers into Low Risk, Medium Risk, and High Risk tiers based on predicted probabilities.
* **Financial ROI Analysis:** Quantifies the net profit and campaign ROI ($100/user outreach cost vs. $1,200 Customer Lifetime Value saved).
* **Production-Ready Artifacts:** Exports trained models (`churn_model.pkl`), scalers (`scaler.pkl`), and high-risk customer lists (`high_risk_customers.csv`).

---

## 📊 Business ROI Highlights

| Metric | Output / Value |
| :--- | :--- |
| **Model Recall (Sensitivity)** | **86.2%** (50 of 58 churners caught in test set) |
| **ROC-AUC Score** | **0.7742** |
| **Targeted Outreach List** | 120 customers (TP + FP) |
| **Campaign Cost** | $12,000.00 ($100 / user) |
| **Estimated Saved Revenue** | $24,000.00 (20 saved accounts @ $1,200 LTV) |
| **Net Campaign Profit** | **$12,000.00** |
| **Projected ROI** | **100.00%** |

---

## 💡 Key Business Drivers & Insights

1. **Contract Type:** Month-to-month contracts are the primary indicator of churn risk.
   * *Strategy:* Target month-to-month users with promotional incentives to migrate them to 1-year contracts.
2. **Support Interactions & Payment Delays:** Customers with >3 support tickets or payment delays exceeding 15 days exhibit higher churn probabilities.
   * *Strategy:* Automatically flag and route accounts with high ticket volumes and payment delays to priority customer support.

---

## 🚀 Getting Started

### Prerequisites

Ensure you have Python 3.8+ installed along with the required libraries:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn joblib
