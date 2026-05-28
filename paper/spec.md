# Project Specification

## Research Question

Does the TFT improve out-of-sample RMSE over ARIMA and VAR baselines for 1Q and 4Q forecasts of US CPI inflation, unemployment and industrial production growth on FRED-MD over 2020-2024?

## Data

FRED-MD: 1990–2017 for training, 2018-2019 validation, and 2020-2024 for testing.

Features from FRED-MD used at native monthly frequency for the CPI, UNRATE and INDPRO models.

Approximately 100 of the 127 series after dropping those with >5% missing values.

## Targets

Target transformations follow McCracken & Ng (2016) where standard:

- **INDPRO**: code 5 (log first difference) → monthly industrial production growth rate. INDPRO is the standard monthly real-activity indicator used in Federal Reserve nowcasting models (Higgins 2014, Atlanta Fed GDPNow).
- **CPIAUCSL**: code 5 (log first difference) → monthly inflation rate. This deviates from McCracken & Ng's recommended code 6 (log second difference, i.e. change in inflation) for interpretability — the forecast target is directly the inflation rate, comparable across the macro forecasting literature.
- **UNRATE**: code 2 (first difference) → monthly change in unemployment rate

## Horizon

For the CPI, UNRATE and INDPRO models (monthly), h=1Q means three months ahead and h=4Q means twelve months ahead.

## Models

**Baselines:**

- **RW**: Random walk, forecast(t+h) = y(t).
- **VAR(p)**: a small system of 5 variables — the three targets plus FEDFUNDS and T10YFFM — with lag order p selected by BIC on the training set.
- **ARIMA(p,d,q)**: orders chosen per target by `auto_arima` minimizing AIC on the training set.

**Model:** TFT (Lim et al. 2021), implemented via `pytorch-forecasting`. Encoder length 24 months (monthly models). Hyperparameters (hidden size, dropout, learning rate, attention heads) selected by random search on the validation set. Quantile outputs at 0.1, 0.5, 0.9; point forecasts are the 0.5 quantile.

## Test-Set Hygiene

The 2020-2024 test set is touched exactly once, after all hyperparameter selection and model tuning is complete. All tuning — TFT hyperparameters, ARIMA order selection, VAR lag length — uses only the training and validation sets (1990-2019). No iterative refinement based on test-set performance is permitted.

## Evaluation

RMSE, MAE, and Diebold-Mariano test against the strongest classical baseline, plus plotting predicted vs. actual.

Statistical significance reported with conventional stars (\*p<0.10, \*\*p<0.05, \*\*\*p<0.01). Given the short test sample (~60 monthly observations), DM tests have limited statistical power; non-significant differences are interpreted accordingly.

**Robustness:** results re-run on the subsample train 1990-2007 / test 2008-2010 (Financial Crisis) to assess regime stability.

As a supplementary diagnostic, TFT's variable importance scores and temporal attention weights are reported to characterize which inputs and lags drive forecasts.
