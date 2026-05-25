# Project Specification

## Research Question

Does the TFT improve out-of-sample RMSE over ARIMA and VAR baselines for 1Q and 4Q forecasts of US GDP growth, CPI inflation, and unemployment on FRED-MD over 2020-2024?

## Data

FRED-MD: 1990–2017 for training, 2018-2019 validation, and 2020-2024 for testing.

Features from FRED-MD used at native monthly frequency for the CPI and UNRATE models. For the GDP model, monthly features are aggregated to quarterly frequency by within-quarter mean to match GDPC1's native frequency, it uses only observations that were actually available in real time, unlike interpolating quarterly GDP into monthly slots which would inject future information.

Approximately 100 of the 127 series after dropping those with >5% missing values.

## Targets

Target transformations follow McCracken & Ng (2016) where standard:

- **GDPC1**: code 5 (log first difference) → quarterly real GDP growth rate
- **CPIAUCSL**: code 5 (log first difference) → monthly inflation rate. This deviates from McCracken & Ng's recommended code 6 (log second difference, i.e. change in inflation) for interpretability — the forecast target is directly the inflation rate, comparable across the macro forecasting literature.
- **UNRATE**: code 2 (first difference) → monthly change in unemployment rate

## Horizon

For the GDP model (quarterly), h=1 means one quarter ahead and h=4 means four quarters ahead. For the CPI and UNRATE models (monthly), h=1Q means three months ahead and h=4Q means twelve months ahead.

## Models

**Baselines:**

- **RW**: Random walk, forecast(t+h) = y(t).
- **VAR(p)**: a small system of 5-7 variables — the three targets plus FEDFUNDS, T10Y3M, and INDPRO — with lag order p selected by BIC on the training set.
- **ARIMA(p,d,q)**: orders chosen per target by `auto_arima` minimizing AIC on the training set.

**Model:** TFT (Lim et al. 2021), implemented via `pytorch-forecasting`. Encoder length 24 months (monthly models) or 8 quarters (GDP model). Hyperparameters (hidden size, dropout, learning rate, attention heads) selected by random search on the validation set. Quantile outputs at 0.1, 0.5, 0.9; point forecasts are the 0.5 quantile.

## Test-Set Hygiene

The 2020-2024 test set is touched exactly once, after all hyperparameter selection and model tuning is complete. All tuning — TFT hyperparameters, ARIMA order selection, VAR lag length — uses only the training and validation sets (1990-2019). No iterative refinement based on test-set performance is permitted.

## Evaluation

RMSE, MAE, and Diebold-Mariano test against the strongest classical baseline, plus plotting predicted vs. actual.

Statistical significance reported with conventional stars (\*p<0.10, \*\*p<0.05, \*\*\*p<0.01). Given the short test sample (~20 quarterly observations), DM tests have limited statistical power; non-significant differences are interpreted accordingly.

**Robustness:** results re-run on the subsample train 1990-2007 / test 2008-2010 (Financial Crisis) to assess regime stability.

As a supplementary diagnostic, TFT's variable importance scores and temporal attention weights are reported to characterize which inputs and lags drive forecasts.
