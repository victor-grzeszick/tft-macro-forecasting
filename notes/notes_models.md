# Model notes — TFT macro forecasting

## Baselines

Random Walk — forecast(t+h) = y(t).
ARIMA — pmdarima.auto_arima fit per target. Both evaluated on the same
validation window with the same RMSE/MAE function.

## Forecasting setup

The h-step-ahead definition: for each origin t, forecast y_{t+h} using
data up to t. Horizons h=3 and h=12. used a rolling-update
scheme — fit auto_arima once to pick orders, then walk through validation
calling .update() with each new actual and re-forecasting. this
deviates from the calendar's "frozen" suggestion; rolling-update is more
faithful to true fixed-h forecasting, at the cost of more compute.

## ARIMA configuration

d=0 — the series are already stationary from the transformation codes,
so no further differencing. seasonal=False. auto_arima orders
(p,d,q) chosen per target — 
CPIAUCSL: (0,0,3), 
UNRATE: (5,0,1), 
INDPRO: (1,0,3)

## VAR configuration

- System: 5 variables — CPIAUCSL, UNRATE, INDPRO, FEDFUNDS, T10YFFM. Note T10YFFM stands in for the spec's T10Y3M (not a FRED-MD column; T10YFFM = 10yr minus fed funds, the native term spread).
- Why those two extras: FEDFUNDS = monetary policy stance, T10YFFM = term spread, both standard activity/inflation predictors in small monetary VARs.
- Lag order: p=1, selected by BIC on the training set (AIC picked 3, HQIC 2 — BIC's heavier penalty gives the smaller, parsimonious order; sensible given 25 coefficients per lag).
- Forecasting scheme: expanding-window refit at each origin, lag frozen at 1 — scheme B, chosen for consistency with ARIMA's rolling update rather than freezing coefficients.
- Stationarity: all 5 inputs pass ADF (details in notes_data) — required since VAR assumes stationary inputs.

## Validation vs test

All metrics reported so far are on the validation set, 2018-2019. The
test set, 2020-2024, is reserved and touched exactly once at the very end
after all model selection — per the spec's test-set hygiene rule.

## Results

data/processed/Baseline_Results.csv.
ARIMA and VAR both beat RW on every target-horizon combo (AR-on-stationary-series sanity check passes).
VAR≈ARIMA with no clear winner, so the strongest classical baseline is chosen per-target for the DM test against TFT.

## Change log

2026-05-26: Completed W3 RW + ARIMA baselines. Rolling-update ARIMA, validation metrics, saved Baseline_Results.csv.
2026-05-28: Completed W3 S2. VAR baseline (5-var, p=1 BIC, scheme B), validation metrics appended to Baseline_Results.csv.