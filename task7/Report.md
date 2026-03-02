# Walmart Sales Forecasting — Project Report

## Overview

End-to-end weekly sales forecasting pipeline built on the Walmart Sales Forecast dataset (Kaggle).
The project covers data preprocessing, exploratory analysis, feature engineering, and two forecasting models — Prophet and XGBoost — with a final robust validation.

---

## Dataset

| File | Rows | Description |
|------|------|-------------|
| `train.csv` | 421,570 | Weekly sales per store/department |
| `features.csv` | 8,190 | Store-level features (temperature, fuel price, CPI, etc.) |
| `stores.csv` | 45 | Store type (A/B/C) and size |
| `test.csv` | 115,064 | Holdout set for submission |

**MarkDown columns dropped** — over 60% missing values, negligible predictive value.

---

## Methodology

### Feature Engineering
- **Time features:** Year, Month, Week, Quarter, DayOfYear
- **Holiday flags:** IsHoliday, IsBlackFriday, IsChristmas, IsThanksgiving
- **Store encoding:** Type_A, Type_B, Type_C (one-hot)
- **Lag features:** Sales at 1, 4, 8, and 52 weeks back (per Store/Dept)
- **Rolling statistics:** 4, 8, 12-week moving average and standard deviation

Rows with NaN from lag features (first ~52 weeks per Store/Dept) were removed before training.

### Train / Validation / Test Split (chronological — no data leakage)

| Split | Period | Samples |
|-------|--------|---------|
| Training | Feb 2011 → May 2012 | ~320k |
| Validation | Jun 2012 → Jul 2012 | ~40k |
| Test | Aug 2012 → Oct 2012 | ~60k |

---

## Models

### Prophet
Trained on aggregated total weekly sales. Captures trend and yearly seasonality with a custom holiday calendar (Black Friday, Christmas).
Best suited for high-level trend visualization — struggles at store/department granularity.

### XGBoost
Trained at store/department level using the full feature set.
Hyperparameters tuned via grid search on the validation set.

**Best hyperparameters:**
- Learning rate: `0.07`
- Max depth: `7`
- N estimators: `500` | Subsample: `0.8` | ColSample: `0.8`

---

## Results

### Model Comparison (Validation Set)

| Model | RMSE | MAE | R² |
|-------|------|-----|----|
| Prophet | $1,413,464 | $1,207,509 | 0.0780 |
| XGBoost | $2,157 | $1,010 | 0.9904 |

XGBoost outperforms Prophet by **99.8% in RMSE** on store/department-level prediction.

### Final XGBoost — All Splits

| Split | RMSE | MAE | R² |
|-------|------|-----|----|
| Training | $1,355 | $798 | 0.9965 |
| Validation | $2,206 | $1,006 | 0.9902 |
| Test | $2,137 | $971 | 0.9906 |

- R² drop (train → test): **0.0059** — very low
- RMSE gap (train → test): **57.6%** — driven by the model seeing harder unseen periods, acceptable given the high absolute R²

> **Note on MAPE:** The dataset contains negative and near-zero weekly sales (returns/corrections), which inflate MAPE artificially. R² and RMSE are the reliable metrics here.

---

## Key Findings

1. **Holiday seasonality is the dominant driver** — weeks 47–52 (Black Friday through Christmas) consistently show the highest sales spikes.
2. **Lag features are the strongest predictors** — particularly `Sales_Lag_1` (last week) and `Sales_Lag_52` (same week last year).
3. **Store characteristics matter** — size and type (A/B/C) have significant influence on sales levels.
4. **XGBoost generalizes well** — test R² of **99.06%** with minimal drop from training confirms the model is not overfitting.
5. **Prophet is complementary** — useful for communicating trends and seasonality visually, but not competitive for granular forecasting.

---

## Notebook Structure

The project is organized into 5 notebooks:

| Notebook | Content |
|----------|---------|
| `01_data_loading_eda.ipynb` | Data loading, merging, and full exploratory analysis |
| `02_feature_engineering.ipynb` | Feature creation, lag/rolling features, train/val split |
| `03_prophet_model.ipynb` | Prophet training, holiday calendar, forecast visualization |
| `04_xgboost_model.ipynb` | XGBoost training, feature importance, residual analysis, model comparison |
| `05_final_validation_summary.ipynb` | 3-way split, hyperparameter tuning, final evaluation, results summary |