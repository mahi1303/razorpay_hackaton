# razorpay_hackaton# 🛡️ RiskShield — AI-Powered Return Risk Manager

> **AI Risk Manager | A defense-only ML system for detecting high-risk product returns and reducing preventable business losses.**

## 🚨 Problem

Product returns quietly reduce e-commerce margins through return shipping, restocking, operational overhead, and lost revenue.

A simple ML classifier that predicts whether an order will be returned is not enough. Businesses also need to determine **when intervention is financially justified**.

* Missing a genuine high-risk return can be expensive.
* Incorrectly flagging a legitimate customer can create unnecessary friction.

**RiskShield** combines machine learning with **cost-sensitive decision threshold optimization** to identify risky transactions while considering the financial impact of model errors.

---

## 🎯 Project Objectives

* Predict the probability that an incoming transaction will result in a return.
* Identify high-risk transactions before potential losses occur.
* Reduce unnecessary intervention against legitimate customers.
* Optimize the classification threshold based on false-positive and false-negative costs.
* Provide a real-time return-risk score.
* Recommend defensive actions for high-risk transactions.
* Measure performance using precision, recall, false-positive rate, false-negative rate, and business cost.

---

## 💡 Solution

RiskShield uses an **XGBoost classification model** to analyze customer, product, transaction, and behavioral data.

```text
Customer / Transaction Data
            │
            ▼
      Data Processing
            │
            ▼
      XGBoost Classifier
            │
            ▼
     Return Risk Score
            │
            ▼
   Cost-Optimized Threshold
            │
      ┌─────┴─────┐
      ▼           ▼
   LOW RISK    HIGH RISK
      │           │
      ▼           ▼
   Approve    Step-up Verification /
              Return Policy Warning
```

Instead of relying on the default `0.50` classification threshold, RiskShield selects a threshold according to the estimated financial impact of incorrect decisions.

---

## 🧠 Machine Learning Features

The model uses customer, product, behavioral, and transaction attributes:

| Feature                  | Description                     |
| ------------------------ | ------------------------------- |
| `customer_age`           | Customer age                    |
| `product_price`          | Product price                   |
| `discount_percent`       | Applied discount                |
| `product_rating`         | Product rating                  |
| `past_purchase_count`    | Number of previous purchases    |
| `past_return_rate`       | Historical customer return rate |
| `delivery_delay_days`    | Delivery delay                  |
| `session_length_minutes` | Shopping session duration       |
| `num_product_views`      | Number of product views         |
| `device_type`            | Customer device                 |
| `product_category`       | Product category                |
| `shipping_method`        | Shipping method                 |
| `payment_method`         | Payment method                  |
| `used_coupon`            | Whether a coupon was used       |

`order_id` is excluded because it is an identifier and does not provide meaningful predictive information.

---

## 🤖 Machine Learning Model

RiskShield uses **XGBoost** for binary classification.

```text
0 → No Return
1 → Return
```

The model outputs a probability representing the estimated return risk.

Example:

```text
Return Risk Score: 78%
Decision: HIGH RETURN RISK
```

---

## 💰 Cost-Sensitive Risk Management

RiskShield considers two major types of classification errors.

### False Negative — Missed Return

The system predicts:

```text
LOW RISK
```

but the order is actually returned.

This can result in return-related business losses.

### False Positive — Incorrectly Flagged Customer

The system predicts:

```text
HIGH RISK
```

but the order would not have been returned.

This creates unnecessary customer friction.

The application allows the business to configure:

* **Estimated Loss per Missed Return**
* **Estimated Customer Friction Cost**

The system evaluates different thresholds and selects the threshold with the lowest estimated business cost using the **validation dataset**.

---

## 🔬 Honest Evaluation Methodology

RiskShield uses separate training, validation, and test sets:

```text
              100% Dataset
                   │
        ┌──────────┼──────────┐
        ▼          ▼          ▼
      60%         20%        20%
    Training   Validation    Test
        │          │          │
        ▼          ▼          │
     Train      Optimize      │
     Model      Threshold     │
        │          │          │
        └──────────┴──────────┘
                   │
                   ▼
          Final Test Evaluation
```

### Why this matters

The **test set remains untouched** during threshold optimization.

* **Training set:** Used to train the XGBoost model.
* **Validation set:** Used to select the cost-optimal decision threshold.
* **Test set:** Used only for final performance evaluation.

This prevents test-set leakage and provides a more honest estimate of real-world performance.

---

## 📊 Evaluation Metrics

### Precision

Of the transactions classified as high-risk, how many actually resulted in returns?

### Recall

Of all actual returns, how many were correctly identified?

### False Positive Rate

What percentage of legitimate transactions were incorrectly flagged?

### False Negative Rate

What percentage of actual returns were missed?

### Business Cost

RiskShield estimates the cost of model errors:

```text
Total Business Cost
=
(False Negatives × FN Cost)
+
(False Positives × FP Cost)
```

This allows the model's threshold to be evaluated from a **business perspective**, not just a statistical one.

---

## 🖥️ Application Features

### 📊 Model Evaluation & Cost Trade-offs

The dashboard provides:

* Precision
* Recall
* False Positive Rate
* False Negative Rate
* Confusion Matrix
* Cost optimization curve
* Decision threshold
* Estimated business cost
* Validation/test methodology

### 🛒 Live Transaction Simulator

Users can enter transaction attributes and receive a real-time risk assessment.

```text
Transaction Details
        │
        ▼
RiskShield Model
        │
        ▼
Return Risk Score
        │
        ▼
Compare With Threshold
        │
   ┌────┴────┐
   ▼         ▼
LOW RISK   HIGH RISK
   │         │
   ▼         ▼
Approve    Verify / Warn
```

---

## 🛠️ Tech Stack

### Machine Learning

* Python
* Pandas
* NumPy
* Scikit-learn
* XGBoost

### Application

* Streamlit

### Visualization

* Plotly

### Dataset

* Kaggle

---

## 📁 Project Structure

```text
RiskShield/
│
├── app.py
├── train.csv
├── test.csv
├── requirements.txt
├── README.md
└── .gitignore
```

> If the Kaggle dataset license does not permit redistribution, keep the CSV files out of the repository and provide instructions for downloading the dataset from its original Kaggle source.

---

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd RiskShield
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Add the dataset

Place the required dataset file in the project directory:

```text
train.csv
```

### 4. Run the application

```bash
streamlit run app.py
```

The RiskShield dashboard will open in your browser.

---

## 📦 Requirements

```text
streamlit
pandas
numpy
scikit-learn
xgboost
plotly
```

---

## 🛡️ Defense-Only Design

RiskShield is strictly designed for **defensive risk management and loss prevention**.

The system:

* Detects potentially high-risk returns.
* Estimates financial impact.
* Recommends defensive interventions.
* Helps reduce unnecessary customer friction.

It does **not** generate fraudulent activity, exploit payment systems, bypass security controls, or facilitate financial abuse.

---

## 🚀 Future Improvements

* Real-time transaction/event streaming.
* SHAP-based model explainability.
* Customer-level risk history.
* Model probability calibration.
* Automated monitoring for return-rate distribution shifts.
* Integration with e-commerce checkout systems.
* Fraud and chargeback risk modules.
* Continuous model retraining using newly labeled transactions.

---

## 🏆 Hackathon Track Alignment

| Requirement             | RiskShield                      |
| ----------------------- | ------------------------------- |
| Risk category           | Product Returns                 |
| Solution type           | AI Return-Risk Scorer           |
| ML model                | XGBoost                         |
| Decision mechanism      | Cost-sensitive threshold        |
| Key metrics             | Precision & Recall              |
| Additional metrics      | FPR & FNR                       |
| Financial consideration | FP/FN Cost Matrix               |
| Evaluation              | Untouched Held-Out Test Set     |
| Real-time demo          | Streamlit Transaction Simulator |
| Security approach       | Defense-only                    |

---

## 👥 Project

### **RiskShield — AI-Powered Return Risk Manager**

> **Turning return prediction into cost-aware risk decisions.**
