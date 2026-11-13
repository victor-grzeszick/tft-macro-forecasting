# Does a Temporal Fusion Transformer forecast US macroeconomic series better than classical benchmarks?

**Victor Grzeszick** · Felix-Hausdorff-Gymnasium Bonn · November 2026

## Abstract

ABSTRACT_PLACEHOLDER

## 1. Introduction

Forecasts of inflation, unemployment and output growth are inputs to monetary policy and to most
investment decisions. For decades the standard tools have been small linear time-series models: univariate
ARIMA models and vector autoregressions (VARs). These models are hard to beat. Stock and Watson (2007) show
that for US inflation even sophisticated models rarely improve on simple univariate benchmarks, and
Atkeson and Ohanian (2001) find that a naive forecast matches Phillips-curve models.

Deep learning offers an alternative. The Temporal Fusion Transformer (TFT; Lim et al., 2021) combines
recurrent layers, an attention mechanism and a learned variable-selection step. It can use many predictors
at once, produces forecast intervals, and reports which inputs it relied on. FRED-MD (McCracken and Ng,
2016), a monthly database of about 126 US series, is a natural test bed: it has many predictors but only a
few hundred observations per series, which is small by deep-learning standards.

This report asks one question: **does a TFT that sees all FRED-MD series produce more accurate 3- and
12-month-ahead forecasts of US CPI inflation, the unemployment rate and industrial production than a random
walk, an ARIMA model and a small VAR?** The emphasis is on making the comparison fair. Every forecast uses
only information available at the time it is made, all model choices are fixed on data up to 2019, and the
2020–2024 test period was evaluated once, after the pipeline had been frozen.

## 2. Data

### 2.1 Source

The data are the FRED-MD file `current.csv`, downloaded on 10 May 2026 from the St. Louis Fed
(observations 1959-01 to 2026-03, 126 series). FRED-MD is re-issued every month with revised values, so the
file is stored in the repository (`data/raw/fred_md.csv`) and all results refer to this vintage.

### 2.2 Transformations and targets

Most macroeconomic series trend upwards and are therefore non-stationary: their mean changes over time, and
regressions between such series can find relationships that do not exist. FRED-MD assigns each series a
transformation code that removes the trend (for example 2 = first difference, 5 = first difference of the
logarithm). I apply these codes with one exception, which gives the three targets:

| Target | Series | Transformation | Meaning |
|---|---|---|---|
| Inflation | CPIAUCSL | code 5: Δ log | monthly CPI inflation rate |
| Unemployment | UNRATE | code 2: Δ | monthly change in the unemployment rate (percentage points) |
| Output | INDPRO | code 5: Δ log | monthly growth of industrial production |

McCracken and Ng use code 6 (the *change* in inflation) for CPI. I use code 5 so that the target is the
inflation rate itself, which is easier to interpret. Industrial production stands in for GDP, which is
only available quarterly.

Augmented Dickey-Fuller tests on the training sample reject a unit root for all five variables used in the
VAR (Table 1), although only at the 5% level for the term spread.

**Table 1. ADF tests, 1990-01 to 2017-12** (`results/adf_tests.csv`)

| Series | ADF statistic | p-value |
|---|---|---|
ADF_ROWS

### 2.3 Sample

The estimation panel runs from 1990-01 to 2024-12 (420 months). One series (ACOGNO) is dropped because more
than 5% of its training observations are missing, leaving 125 series. Three missing cells in 2020 (CP3Mx in
April and May, COMPAPFFx in April) are filled with the previous month's value. The sample is split by time:

| Period | Months | Use |
|---|---|---|
| Training | 1990-01 to 2017-12 (336) | estimation |
| Validation | 2018-01 to 2019-12 (24) | early stopping, hyperparameters, choice of best baseline |
| Test | 2020-01 to 2024-12 (60) | final evaluation, used once |

A random split would be invalid for time series: neighbouring months are correlated, so a model trained on
randomly chosen months would effectively have seen the months it is tested on.

## 3. Forecasting design

### 3.1 Forecast definition

At the end of month *t* (the forecast origin) each model forecasts the value of the transformed target in
month *t + h*, for *h* = 3 and *h* = 12, using only data up to and including *t*. The target is the monthly
value *h* months ahead, not the cumulative change over the next *h* months.

Forecasts are produced walk-forward (rolling origin): the origin moves forward one month at a time. A
forecast is counted in a period only if its origin is no earlier than the last month before that period.
For the test period this means origins from 2019-12 on, giving 58 forecasts at *h* = 3 (targets 2020-03 to
2024-12) and 49 at *h* = 12 (targets 2020-12 to 2024-12); for validation, 22 and 13. No forecast is
therefore made from inside a sample that was used to fit or select the model.

Data are final revised values, not the real-time vintages a forecaster would have had (see Section 6).

### 3.2 Benchmarks

- **Random walk (RW):** the forecast is the last observed value, ŷ(t+h) = y(t).
- **Historical mean (Mean):** the forecast is the average of all observations up to *t*. For a stationary
  series this is the forecast one would make if the past contained no usable information beyond the mean.
- **ARIMA(p, 0, q):** a univariate model in which the series depends on its own *p* past values and *q*
  past forecast errors. Orders are chosen per target by stepwise AIC search on the training sample
  (`pmdarima.auto_arima`); no differencing is applied because the targets are already stationary.
- **VAR(p):** each of five variables (the three targets, the federal funds rate and the 10-year-minus-
  federal-funds term spread) is regressed on *p* lags of all five. The lag length is chosen by BIC on the
  training sample.

Selected specifications (`results/baseline_specs.json`): BASELINE_SPECS

Orders and lag length are fixed after the training sample. The parameters of ARIMA and VAR are re-estimated
at every origin on all data up to that origin (expanding window), which is standard practice in
econometric forecast evaluation.

### 3.3 Temporal Fusion Transformer

The TFT is implemented with `pytorch-forecasting`. One model is trained per target and horizon (six in
total), mirroring the per-series benchmarks. Its main components are:

1. **Variable selection network.** For each month, a small network assigns a weight to each of the 126
   inputs (125 series plus the position in the window), and the model works with the weighted combination.
   The weights sum to one and can be read as variable importances.
2. **LSTM encoder–decoder.** A recurrent network reads the 24 months before the origin (encoder) and then
   steps through the *h* forecast months (decoder), carrying a summary of the past.
3. **Multi-head attention.** For each forecast month the model computes weights over all past months and
   takes a weighted average of their representations, so it can draw directly on any earlier month.
4. **Gated residual networks.** Skip connections with learned gates let the model bypass components it
   does not need, which matters when data are scarce.
5. **Quantile outputs.** The model outputs the 10%, 50% and 90% quantiles and is trained with the quantile
   (pinball) loss. The median is used as the point forecast.

All 125 series enter as past-observed inputs, standardised with means and variances from the training
sample. The target is standardised within each 24-month encoder window. The encoder length of 24 months was
fixed in advance and not tuned.

**Training.** Models are trained on 1990–2017 (310 samples at *h* = 3, 301 at *h* = 12) with the Adam
optimiser, batch size 64 and seed 42. Training stops when the validation loss has not improved for 15
epochs, and the weights of the best validation epoch are kept. Weights are then frozen: the same model
produces all validation and test forecasts.

**Tuning.** Hidden size {8, 16, 32}, dropout {0.1, 0.2} and learning rate {0.01, 0.03} were searched in a
12-point grid on one model (UNRATE, *h* = 3), and the configuration with the lowest validation loss was
applied to all six models. TUNING_SENTENCE

### 3.4 Evaluation

Accuracy is measured by the root mean squared error (RMSE) and the mean absolute error (MAE). RMSE is also
reported relative to the random walk, so that values below one mean the model is more accurate.

A lower RMSE in one sample can be due to chance. The **Diebold-Mariano (DM) test** asks whether the
difference in accuracy between two models is statistically significant. For each month it takes the
difference between the two models' squared errors, d(t) = e₁(t)² − e₂(t)², and tests whether the mean of
d is zero. Because *h*-step forecast errors overlap and are therefore correlated up to lag *h* − 1, the
variance of the mean is estimated with autocovariances up to that lag. I apply the small-sample correction
of Harvey, Leybourne and Newbold (1997) and compare the statistic with a Student-*t* distribution with
*n* − 1 degrees of freedom. A negative statistic means the first-named model is more accurate.

Three robustness checks were specified before the test run: (i) ARIMA and VAR with parameters frozen at
the end of the training sample, which is the like-for-like comparison with the TFT's frozen weights; (ii)
four further random seeds for the TFT and the average (ensemble) of all five; (iii) the test period
without 2020.

RESULTS_PLACEHOLDER
