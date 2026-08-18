# Data-Driven Electricity Load Forecasting & Renewable Reliability Analysis

A data-driven framework for **short-term electricity load forecasting** and **renewable reliability analysis** using high-resolution German electricity data.

## Overview

The project combines machine-learning-based electricity demand forecasting with renewable-generation analysis to identify **potentially critical periods where high electricity demand coincides with comparatively low renewable support**.

The system processes historical electricity data, engineers temporal and lag-based features, evaluates multiple forecasting models, and analyzes demand–renewable conditions.

## Key Features

- High-resolution electricity-load analysis
- Short-term demand forecasting
- Temporal and historical feature engineering
- Renewable generation analysis
- Renewable Coverage Ratio and Generation Gap calculation
- Demand–renewable condition classification
- Potentially critical reliability-window identification
- Chronological model evaluation
- Persistence baseline comparison

## Models

| Model | MAE (MW) | RMSE (MW) | R² |
|---|---:|---:|---:|
| Persistence | 506.90 | 670.42 | 0.994445 |
| Linear Regression | 289.44 | 398.98 | 0.998033 |
| Random Forest | 272.41 | 364.91 | 0.998354 |
| **XGBoost** | **254.36** | **341.28** | **0.998561** |

**XGBoost achieved the best overall forecasting performance.**

## Dataset

- **Observations:** 105,216
- **Variables:** 17
- **Temporal resolution:** 15 minutes
- **Coverage:** Approximately 3 years
- **Region:** Germany

## Methodology

```text
Data Validation
      ↓
Preprocessing
      ↓
Feature Engineering
      ↓
Load Forecasting
      ↓
Renewable Analysis
      ↓
Reliability Analysis
      ↓
Model Evaluation
