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
so no further differencing. seasonal=False. List the auto_arima orders
(p,d,q) chosen per target — 
CPIAUCSL: (0,0,3), 
UNRATE: (5,0,1), 
INDPRO: (1,0,3)

## Validation vs test

All metrics reported so far are on the validation set, 2018-2019. The
test set, 2020-2024, is reserved and touched exactly once at the very end
after all model selection — per the spec's test-set hygiene rule.

## Results

data/processed/Baseline_Results.csv. 
ARIMA beats RW on every target-horizon combo, so the AR-on-stationary-series
sanity check passes.

## Change log

2026-05-26: Completed W3 RW + ARIMA baselines. Rolling-update ARIMA, validation metrics, saved Baseline_Results.csv.