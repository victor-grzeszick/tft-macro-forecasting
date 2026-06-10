# Model notes — TFT macro forecasting

## Code layout

- baseline.py — generates forecasts, pickles to predictions_unalligned.pkl
- evaluation_utils.py — shared align() and evaluate()
- align_results.py — aligned RMSE/MAE → Baseline_Results.csv
- dm_test.py — Diebold-Mariano tests
- plot_baselines.py — predicted-vs-actual figures → figures/

## Baselines

Random Walk — forecast(t+h) = y(t).
ARIMA — pmdarima.auto_arima fit per target. 
VAR — single 5-variable system, forecasts read per target (see VAR configuration)

All three baselines are aligned to a common date window before scoring (see Alignment), then evaluated with the same RMSE/MAE function.

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
- Stationarity: all 5 inputs pass ADF at the 5% level (CPIAUCSL fails at 1%, T10YFFM marginal — details in notes_data)

## Validation vs test

All metrics reported so far are on the validation set, 2018-2019. The
test set, 2020-2024, is reserved and touched exactly once at the very end
after all model selection — per the spec's test-set hygiene rule.

## Results

data/processed/Baseline_Results.csv.
ARIMA and VAR both beat RW on every target-horizon combo (AR-on-stationary-series sanity check passes).
VAR≈ARIMA with no clear winner, so the strongest classical baseline is chosen per-target for the DM test against TFT.

## Alignment

The Random Walk, unlike ARIMA and VAR, can reach back into the training
data, so it produces a forecast for every validation month. The rolling
models can't: an h-step forecast can't be placed until h months into the
window. So the three models cover different date ranges. Before comparing
them, all are trimmed to the dates they share for each (target, horizon) —
h=3 → 22 months, h=12 → 13 — and RMSE/MAE are recomputed on that aligned
window. Without this the errors are measured over different periods and
aren't comparable. Done once in align() (evaluation_utils.py), reused by
the scoring and DM steps.

## DM test

Checks whether one model's lower error is a genuine edge or just luck on
this sample. Takes the per-month difference of the two models' squared
errors and tests whether its mean is distinguishable from zero; |DM| > ~1.96
(p < 5%) means a real difference, otherwise noise. Variance corrected for
multi-step overlap via autocovariances out to lag h−1. Sign convention:
models passed (better, worse), so negative DM = the better model won.
Validation set 2018–2019; implemented in dm_test.py.

Results (h=3): ARIMA ≈ VAR, each winning one target; both beat RW
significantly on UNRATE; CPI borderline; INDPRO indistinguishable from RW.
h=12 dropped — 13 months is too short for the lag-11 correction (variance
estimate degenerates), deferred to the test set. Validation window is short
(low power) and 9 simultaneous tests raise multiple-comparison risk, so all
results are preliminary, to be confirmed on the test set.

## Figures

figures/ holds one predicted-vs-actual plot per target × horizon (6 total):
Actual, RW, ARIMA, VAR on shared axes. Visual sanity check — confirms the
rolling models track smoothly while RW chases noise, and that all models
struggle at these short horizons.

## Change log

2026-05-26: Completed W3 RW + ARIMA baselines. Rolling-update ARIMA, validation metrics, saved Baseline_Results.csv.
2026-05-28: Completed W3 S2. VAR baseline (5-var, p=1 BIC, scheme B), validation metrics appended to Baseline_Results.csv.
2026-06-10: W3 final. DM test (lag-(h−1) corrected) vs ARIMA/VAR/RW; split baseline.py into generation + evaluation_utils + align_results + dm_test; shared align(); predicted-vs-actual plots → figures/.