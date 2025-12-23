# ✈️ Airline Delay Prediction — Machine Learning Summary

A production-inspired machine learning pipeline that predicts airline arrival delays using a two-stage modeling approach to handle the heavy-tailed nature of flight delay distributions.

## Overview

This project implements a two-stage machine learning pipeline to predict airline arrival delays using large-scale real-world flight data. Instead of relying on a single regression model, the system explicitly handles the heavy-tailed nature of flight delays by separating normal delays from extreme delays, resulting in more stable and interpretable predictions.

---

## Modeling Approach

### 1. Heavy-Delay Classification
- **Model:** HistGradientBoostingClassifier
- **Objective:** Classify flights into:
  - Normal delay (≤ 30 minutes)
  - Heavy delay (> 30 minutes)
- **Reasoning:** Heavy delays behave differently from routine operational delays and require specialized modeling.

### 2. Delay Regression (Two Models)
- **Model:** XGBoostRegressor
- **Target:** `log1p(ArrivalDelay)` (log-transformed to handle skewed distribution)
- Separate regressors trained for:
  - Normal delays
  - Heavy delays
- **Benefit:** This specialization improves accuracy and robustness compared to a single global regressor.

---

## Feature Engineering

The model uses a combination of operational, temporal, categorical, and historical features:

### Operational Features
- Taxi-out time, taxi-in time, air time
- Flight distance
- Scheduled and actual departure/arrival hours

### Temporal Features
- Month
- Day of week
- Weekend indicator

### Categorical Features
- Airline
- Origin airport
- Destination airport
- Route
*(High-cardinality categories reduced using top-N encoding with an OTHER bucket.)*

### Historical Rolling Features (Leakage-Safe)
- 7-day and 14-day rolling mean arrival delay per Airline and Route
- Rolling flight counts as congestion indicators
*All rolling features are computed only on training data and safely mapped to test data.*

---

## Evaluation Results (Test Set)

### Overall Performance
- **RMSE:** ~39.5 minutes
- **MAE:** ~16.5 minutes
- **R²:** ~0.58

### Heavy-Delay Classifier
- **Accuracy:** ~92.6%
- **F1-score (heavy delay class):** ~0.69
- **Note:** High precision ensures reliable heavy-delay alerts

### Segmented Regression Performance
- **Normal delays (≤30 min):** MAE ~12.5 minutes
- **Heavy delays (>30 min):** MAE ~40 minutes

> **Note:** These results are strong for airline delay prediction without external data such as weather or air traffic control constraints.

---

## Prediction Workflow

At inference time, the system works as follows:

1. **Input:** Flight details available before departure
2. **Classification:** Classifier predicts whether the flight is likely to experience a heavy delay
3. **Regression:** Based on the classification:
   - Normal-delay regressor predicts expected delay, **OR**
   - Heavy-delay regressor predicts expected delay
4. **Output:** Final predicted arrival delay in minutes, with an optional heavy-delay risk flag

---

## Key Takeaway

This project demonstrates a realistic, leakage-safe, and production-inspired ML pipeline for airline delay prediction, combining classification and regression to handle skewed outcomes effectively. The approach is well-suited for operational forecasting, dashboards, and decision-support systems.
