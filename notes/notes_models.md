# Model notes — TFT macro forecasting

## Code layout

- baseline.py — generates forecasts, pickles to predictions_unalligned.pkl
- evaluation_utils.py — shared align() and evaluate()
- align_results.py — aligned RMSE/MAE → Baseline_Results.csv
- dm_test.py — Diebold-Mariano tests
- plot_baselines.py — predicted-vs-actual figures → figures/
- pytorchDummy.py — W4 toy: TFT on synthetic sine, full pipeline test
- build_data_sets.py — wide→long prep, build_one_dataset(df_train, df_val, h, target) → (train_ds, val_ds)
- trainer.py — trains/loads one TFT per target×horizon; checkpoint → models/, plot → figures/

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

TFT CPIAUCSL h=3 ≈ 0.0014 RMSE / 0.0012 MAE, sitting right on tuned ARIMA (0.00137), beating VAR and RW

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

## Toy run 

Sine + noise (σ=0.1), 200 points, encoder 24 / h=3, hidden_size 16 (15.5k params).
Findings: 10 epochs underfit (val 0.32); ~300 epochs near the noise floor
(val 0.049); 400 epochs overfit onset — train falls, val rises to 0.059 →
EarlyStopping mandatory for FRED runs. Forecasts track flanks well, miss
turning points — expect the same on macro series. Checkpoint workflow
established: auto-save to lightning_logs/, reload via load_from_checkpoint;
explicit saves to models/ for final runs.

## TFT data setup 

Wide X_/Y_ parquets → long format: integer time_idx replaces the date index,
single dummy group_id ("series" = "macro"), one TimeSeriesDataSet per
target × horizon (6 total), per-target modelling per the Stage-1 decision.

Encoder length 24 (fixed a priori, ~2yr context, standard in the macro
literature — not tuned, to limit researcher degrees of freedom). All 125
FRED columns are time_varying_unknown_reals (incl. the target's own past);
time_idx is the only known real for now (calendar dummies pending).

Val datasets via from_dataset(train_ds, df_full, min_prediction_idx=336):
inherits train config + train-fit normalization (no leakage), encoders
reach back into late-2017. Train examples 310/301 (h=3/12), val 22/13 —
val counts match the baseline aligned windows.

Verified: all 125 inputs StandardScaler-scaled, target EncoderNormalizer, scalers train-fit and inherited by val via from_dataset (no leakage)

## First FRED TFT run

CPIAUCSL h=3, default config (hidden_size 16, lr 0.03, QuantileLoss), seed 42.
EarlyStopping(patience=15) → stopped epoch 22. Trains clean, no NaN.
Forecast is flat near the series mean with wide quantile bands covering the
actuals — consistent with short-horizon monthly inflation being hard to
forecast (Stock-Watson 2007), and with the baselines (CPI was already
borderline ARIMA-vs-RW). Not undertraining: converged by ~epoch 7.
val_loss is pinball, not RMSE — not comparable to Baseline_Results.csv;
Checkpoint: models/tft_CPIAUCSL_h3.ckpt.

TFT hits ARIMA-level error via the flat forecast → short-horizon CPI is near-unforecastable, all methods collapse to the mean; not a genuine match.

VSN collapsed ~all weight (21.9997/22) onto HOUSTNE, which shows, that although its not a leakage, the tft chooses something almost meaningless, which is just mostly a constant.

## Tuning

Architecture tuned once on UNRATE h=3, then frozen across all six models; learning rate found per-model via a manual sweep. That design limits researcher degrees of freedom (same logic as fixing encoder length a priori), and only let lr vary because it's optimization-specific, not capacity.
Why the manual lr sweep instead of lr_find: lr_find kept crashing on the torch 2.6 weights_only checkpoint-restore (EncoderNormalizer/numpy whack-a-mole), so I picked lr by manual trial, narrowing to ~0.023.

The architecture result: grid over hidden_size {8,16,32} × dropout {0.1,0.15,0.2}, all at fixed lr on UNRATE h=3, selected by val_loss → (8, 0.1). 

The model will be trained with those parameters and not with those from the first run.

!Critically! It won probably because val_loss was nearly flat across the whole grid (~0.048–0.052), so the smallest model won on parsimony, not because it meaningfully beat the others.
The UNRATE h=3 number (MAE ~0.097 / RMSE ~0.113) and the third single-variable VSN collapse (MANEMP) — three targets, three collapses now.
It's persisting exactly as predicted for the flat regime.

## To Do

-Perhaps see if a new training of hyperparameters for each variable is feasable if it only takes more compute/time

## Change log

2026-05-26  W3S1 — RW + ARIMA baselines: rolling-update ARIMA, validation metrics, Baseline_Results.csv saved.
2026-05-28  W3S2 — VAR baseline (5-var, p=1 BIC, scheme B); validation metrics appended to Baseline_Results.csv.
2026-06-10  W3S3 — DM test (lag-(h−1) corrected) vs ARIMA/VAR/RW; split baseline.py into generation + evaluation_utils + align_results + dm_test; shared align(); predicted-vs-actual plots → figures/.
2026-06-11  W4   — pytorch-forecasting toy on sine; dataset→loader→train→checkpoint→reload→predict→plot verified end-to-end; under/overfit boundary located; .gitignore + README replication section added.
2026-06-13  W5S1 — FRED-MD wide→long; 6 per-target×horizon TimeSeriesDataSets (train + val via from_dataset, min_prediction_idx=336); prepare() helper.
2026-06-13  W5S2 — First FRED TFT (CPIAUCSL h=3, default config) end-to-end; trainer.py + build_data_sets.py; save/load checkpoint workflow; W5 gate passed.
2026-06-20  W5S3 — TFT RMSE/MAE vs baselines (CPIAUCSL h=3 ≈ ARIMA); variable-importance smoke test (VSN collapsed onto HOUSTNE — degenerate, not leakage); refactored trainer.py into load_or_train / eval_vs_baseline / top_k_importance / save_plot with target/horizon-keyed checkpoint + figure paths.
2026-06-21  W6S1 — hyperparameter tuning: architecture frozen at (hidden_size 8, dropout 0.1) via UNRATE h=3 grid; per-model lr sweep (~0.0234); lr_find abandoned (torch 2.6 weights_only); grid near-flat, smallest model won on parsimony.