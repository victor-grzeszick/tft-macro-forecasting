# Project Specification

## Research Question

Does the TFT improve out-of-sample RMSE over ARIMA and VAR baselines for 3- and 12-month-ahead forecasts of US CPI inflation, unemployment, and industrial production growth on FRED-MD over the 2020–2024 test period?

## Data

FRED-MD (revised vintage, 2026-05), at native monthly frequency: 1990–2017 for training, 2018–2019 for validation, 2020–2024 for testing.

~125 of the 127 series retained after dropping columns with >5% missing values (only ACOGNO removed; see notes_data).

## Targets

Transformations follow McCracken & Ng (2016) where standard:

- **INDPRO**: code 5 (log first difference) → monthly industrial production growth. The standard monthly real-activity indicator in Federal Reserve nowcasting models (Higgins 2014, Atlanta Fed GDPNow).
- **CPIAUCSL**: code 5 (log first difference) → monthly inflation rate. Deliberate deviation from McCracken & Ng's code 6 (log second difference) for interpretability — the target is the inflation rate directly, comparable across the macro forecasting literature.
- **UNRATE**: code 2 (first difference) → monthly change in the unemployment rate.

GDPC1 (real GDP) was dropped as a target: quarterly frequency yields too few training observations, and interpolation to monthly would introduce look-ahead bias. INDPRO serves as the monthly real-activity proxy in its place.

## Horizons

h = 3 and h = 12 months ahead, for all three targets.

## Models

**Baselines:**

- **RW**: random walk, forecast(t+h) = y(t).
- **ARIMA(p,d,q)**: orders chosen per target by `auto_arima` (AIC) on the training set.
- **VAR(p)**: a 5-variable system — the three targets plus FEDFUNDS and T10YFFM (the native 10yr-minus-fed-funds term spread; FRED-MD has no T10Y3M column) — lag order p selected by BIC on the training set.

**TFT** (Lim et al. 2021), via `pytorch-forecasting`. A separate model is trained per target and horizon, mirroring the per-series ARIMA baselines, so any RMSE gap is attributable to architecture rather than joint-vs-separate modelling; joint multi-target modelling is left to Stage 2. Encoder length fixed at 24 months a priori (≈2 years, the standard lookback in the macro forecasting literature) — not tuned, to limit researcher degrees of freedom. Hyperparameters (hidden size, dropout, learning rate, attention heads) selected by random search on the validation set. Quantile outputs at 0.1, 0.5, 0.9; point forecasts are the 0.5 quantile.

## Test-Set Hygiene

The 2020–2024 test set is touched exactly once, after all model selection is complete. All tuning — TFT hyperparameters, ARIMA orders, VAR lag length — uses only training and validation data (1990–2019). No iterative refinement on test-set performance is permitted.

## Evaluation

RMSE and MAE per model × target × horizon, plus predicted-vs-actual plots. Diebold-Mariano test against the strongest classical baseline per target, with conventional significance stars (\*p<0.10, \*\*p<0.05, \*\*\*p<0.01). Given the short test sample (~60 monthly observations), DM tests have limited power; non-significant differences are interpreted accordingly.

As a supplementary diagnostic, TFT variable-importance scores and temporal attention weights are reported to characterize which inputs and lags drive the forecasts.

**Robustness (scope-dependent):** if time permits, the main comparison is re-run on a crisis subsample (train 1990–2007, test 2008–2010) to assess regime stability. This requires retraining all TFT models on a second split; it is the first item to defer to a 4-page note or Stage 2 if the July 5 deadline tightens.