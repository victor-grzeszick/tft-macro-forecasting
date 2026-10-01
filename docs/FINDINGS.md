# Findings

**Every error and DM number below is a validation result.** Nothing has been evaluated on the 2020–2024 test period, so the research question is not answered yet. <!-- src: paper/spec.md:5; git grep -E "X_test|Y_test" -- src scripts (only the writes at src/data_pipeline.py:72,76) -->

**Windows.** Validation months are 2018-01 to 2019-12. RW, ARIMA and VAR are scored on the target dates they share: h=3 → 2018-03 to 2019-12 (n=22); h=12 → 2018-12 to 2019-12 (n=13). <!-- src: notes/notes_models.md:64-71; src/evaluation_utils.py:18-24; data/processed/predictions_unalligned.pkl (ARIMA and VAR date index; RW covers 2018-01 to 2019-12) --> The ADF tests (section 4) use other samples.

## Limitations (read first)

- **L1 ARIMA saw the validation data.** `baseline.py` hands train + validation (1990–2019) to `auto_arima` as the training series, then appends the validation months again and places each forecast by position. So the models share target dates but not information: every ARIMA forecast for 2018–2019 comes from a model fitted on data that include 2018–2019. The ARIMA errors are not out-of-sample; the size and direction of the bias were not measured. ARIMA is the lowest-RMSE baseline in 4 of 6 cells and the DM baseline for CPIAUCSL and INDPRO. The ARIMA orders in the notes come from this fit, not from "the training set" as the spec says. <!-- src: src/baseline.py:27,29-37,79,93; data/processed/Baseline_Results.csv:2-19; data/processed/dm_test_results.parquet (Compare1); paper/spec.md:32; notes/notes_models.md:34-38 -->
- **L2 Validation was also used for selection.** TFT early stopping, the tuning grid, the learning rate and the best-baseline choice all used 2018–2019. <!-- src: src/trainer.py:18-20; notes/notes_models.md:146-149; src/dm_test.py:46-49 -->
- **L3 Small samples, weak tests.** Each cell has 22 or 13 dates. DM is a statistic only: no p-value, no small-sample correction, no h=12 test, and 12 tests so far (9 baseline pairs, 3 TFT comparisons). The spec expects ~60 test observations; reusing the validation design would give 58 (h=3) and 49 (h=12) target dates. <!-- src: src/dm_test.py:36,52-57; 770bc4d:data/processed/dm_test_results.parquet (9 rows); data/processed/dm_test_results.parquet (3 rows); paper/spec.md:43; src/baseline.py:35-37 (first forecast lands h−1 months into the window: 60−h+1) -->
- **L4 Revised data.** "Revised FRED data, not real-time vintages — limits real-time forecasting claims." <!-- src: notes/notes_data.md:73 -->
- **L5 COVID.** "COVID outliers (2020–2021) fall inside the test set — test RMSE will be dominated by a few extreme months". <!-- src: notes/notes_data.md:75-76 -->
- **L6 Saved TFT weights are from the last epoch, not the best one.** Training stops 15 epochs after the best `val_loss` and saves then; the early-stopping state in all six checkpoints records this. <!-- src: src/trainer.py:18-22; models/tft_*_h*.ckpt (EarlyStopping wait_count 15, patience 15) -->
- **L7 TFT setup differs from the spec.** Seven default quantiles instead of 0.1/0.5/0.9; a grid instead of random search; 4 attention heads, never tuned; dropout never passed to the model (library default 0.1); learning rate 0.0234 for all six, although the notes describe a per-model sweep. TODO(Victor): confirm the per-model sweep. <!-- src: src/trainer.py:17,61-84,117; paper/spec.md:35; notes/notes_models.md:146; models/tft_*_h*.ckpt hyper_parameters (quantiles 0.02-0.98, attention_head_size 4, dropout 0.1, learning_rate 0.0234) -->
- **L8 VAR lag p=1 is hard-coded.** The BIC comparison in the notes is not in any committed code. <!-- src: src/baseline.py:57; notes/notes_models.md:44; git grep -i -E "select_order|bic|aic|hqic" over all revisions, *.py (no hits) -->
- **L9 Some preprocessing looked at test years.** The missing-value filter and the target ADF tests use data after 2019. For this vintage the filter keeps the same series either way ([DATA.md](DATA.md#sample-filter-and-split-srcdata_pipelinepy)); the ADF output feeds no model. <!-- src: src/data_pipeline.py:23-26; src/validate.py:20-22 --> For the future test run, `X_test` has 3 missing cells. <!-- src: data/processed/X_test.parquet (3 NaN cells) -->

## 1. Baselines (validation)

Source: `data/processed/Baseline_Results.csv`, rounded. Units: monthly log growth (CPIAUCSL, INDPRO); monthly change in the unemployment rate (UNRATE). Bold = lowest RMSE, the "best baseline". <!-- src: data/processed/Baseline_Results.csv:2-19; paper/spec.md:17-19; src/dm_test.py:46-49 -->

| Series | h | n | RW RMSE | ARIMA* RMSE | VAR RMSE | RW MAE | ARIMA* MAE | VAR MAE |
|---|---|---|---|---|---|---|---|---|
| CPIAUCSL | 3 | 22 | 0.002263 | **0.001370** | 0.001510 | 0.001941 | 0.001142 | 0.001197 |
| CPIAUCSL | 12 | 13 | 0.002292 | **0.001494** | 0.001497 | 0.001838 | 0.001318 | 0.001309 |
| UNRATE | 3 | 22 | 0.1732 | 0.1179 | **0.1120** | 0.1364 | 0.09684 | 0.09172 |
| UNRATE | 12 | 13 | 0.1641 | **0.09460** | 0.09524 | 0.1308 | 0.07629 | 0.07794 |
| INDPRO | 3 | 22 | 0.006274 | **0.005304** | 0.005355 | 0.004791 | 0.004528 | 0.004357 |
| INDPRO | 12 | 13 | 0.007376 | 0.005443 | **0.005290** | 0.006165 | 0.004554 | 0.004386 |

<!-- src: data/processed/Baseline_Results.csv:2-19; data/processed/predictions_unalligned.pkl (n per horizon) -->
\* Look-ahead (L1).

In words (validation, windows above; percentages computed from the CSV):
- ARIMA and VAR have lower RMSE and MAE than RW in all six cells. Their RMSE is 14.6% (VAR, INDPRO h=3) to 42.3% (ARIMA, UNRATE h=12) lower than RW's. <!-- src: data/processed/Baseline_Results.csv:14,16 (INDPRO h=3 RW, VAR) and :11-12 (UNRATE h=12 RW, ARIMA) -->
- ARIMA and VAR are close: VAR RMSE ranges from 5.0% lower than ARIMA (UNRATE h=3) to 10.3% higher (CPIAUCSL h=3). <!-- src: data/processed/Baseline_Results.csv:9-10 (UNRATE h=3 ARIMA, VAR) and :3-4 (CPIAUCSL h=3 ARIMA, VAR) --> By MAE, VAR would be lowest for CPIAUCSL h=12 and INDPRO h=3. <!-- src: data/processed/Baseline_Results.csv:5-7,14-16 -->

Plots of actual vs RW/ARIMA/VAR on the same windows: `figures/{CPIAUCSL,UNRATE,INDPRO}_h{3,12}.png`. <!-- src: src/plot_baselines.py:17-30; git ls-files figures/ -->

## 2. TFT (validation)

- **No TFT metrics file is committed.** `src/trainer.py` writes TFT MAE/RMSE, predictions and variable importances only to gitignored files. Load mode regenerates them ([REPRODUCE.md](REPRODUCE.md), step 7), but no stored copy exists to check them against. <!-- src: src/trainer.py:135-139; .gitignore:226; git ls-files data/processed -->
- **The only summary** is in Victor's notes: "Six-model results — TFT vs best baseline (RMSE): 3 wins, 3 losses, all margins ≤10%. Competitive, doesn't systematically beat tuned ARIMA/VAR." <!-- src: notes/notes_models.md:158 --> TODO(Victor): confirm this count — a load-mode re-run of the committed checkpoints (REPRODUCE step 7) gave a different count; commit the TFT metrics file so the count can be checked.
- TFT error values at `notes/notes_models.md` lines 60 and 154 come from earlier checkpoints, replaced in commit 6e01320 (for UNRATE the exact model cannot be identified). They are not results for the committed models. <!-- src: git blame notes/notes_models.md -L60 (c8b2f49), -L154 (b16fcbe); git log -- models/tft_CPIAUCSL_h3.ckpt (1dbac3c, 6e01320) models/tft_UNRATE_h3.ckpt (b16fcbe, 6e01320) -->

## 3. Diebold-Mariano statistics (validation, h=3, n=22, 2018-03 to 2019-12)

How it is computed: d_t = (baseline error)² − (TFT error)²; variance = the lag-0 autocovariance plus twice the autocovariances at lags 1 to h−1, unweighted; DM = mean(d) / √(variance / n). <!-- src: src/dm_test.py:9-20,52-57 --> A negative DM means the baseline had the lower mean squared error on these dates. <!-- src: src/dm_test.py:10-13,52,57 --> No p-value is computed (L3).

| Series | Baseline (lowest RMSE) | Compared with | DM (rounded) | Lower sample MSE |
|---|---|---|---|---|
| CPIAUCSL | ARIMA* | TFT | −2.139 | ARIMA |
| UNRATE | VAR | TFT | 0.592 | TFT |
| INDPRO | ARIMA* | TFT | −2.152 | ARIMA |

<!-- src: data/processed/dm_test_results.parquet (rows 0-2: -2.139284, 0.592323, -2.152136) -->
Victor's notes call the two values near −2.1 "artifacts, not real significance": the variance estimate is "~0 (2e-13 to 6e-6) because TFT and baselines produce nearly identical flat forecasts". <!-- src: notes/notes_models.md:91 --> TODO(Victor): this explanation is unconfirmed; please re-check it. By the formula, DM does not change when all errors are multiplied by the same constant, so a variance that is small only because the errors are small does not by itself inflate DM. <!-- src: src/dm_test.py:9-20,52-57 -->

**Baseline vs baseline** (same window; overwritten later, kept only in git history: `git show 770bc4d:data/processed/dm_test_results.parquet`):

| Series | ARIMA* vs VAR | ARIMA* vs RW | VAR vs RW |
|---|---|---|---|
| CPIAUCSL | −2.600 | −1.883 | −1.722 |
| UNRATE | +2.086 | −2.294 | −2.194 |
| INDPRO | −0.304 | −0.790 | −0.717 |

<!-- src: 770bc4d:data/processed/dm_test_results.parquet rows 0-8 -->
Negative means the first-named model had the lower mean squared error. <!-- src: 770bc4d:src/dm_test.py:6-11,33-39 --> Victor's notes call all nine "preliminary, to be confirmed on the test set", because the window is short and "9 simultaneous tests raise multiple-comparison risk". <!-- src: notes/notes_models.md:84-89 -->

## 4. Diagnostics

**Stationarity (ADF).** Printed by `src/validate.py`, not saved; values as recorded in `notes/notes_data.md`. <!-- src: src/validate.py:6-13; notes/notes_data.md:46-69 -->

| Series | Sample tested | ADF statistic | p-value (as printed) |
|---|---|---|---|
| CPIAUCSL | all months 1959–2026-03 | −3.1210 | 0.025037 |
| INDPRO | all months 1959–2026-03 | −7.3170 | 0.000000 |
| UNRATE | all months 1959–2026-03 | −15.5832 | 0.000000 |
| FEDFUNDS | 1990-01 to 2017-12 | −4.5576 | 0.000155 |
| T10YFFM | 1990-01 to 2017-12 | −2.8747 | 0.048367 |

<!-- src: notes/notes_data.md:46-69; src/validate.py:20-27 -->
In Victor's words: "CPIAUCSL passes at 5% but fails at 1% — the accepted code-5 trade-off; T10YFFM passes 5% marginally". <!-- src: notes/notes_data.md:42-44 -->

**Variable importance and attention.** No numeric output is committed. <!-- src: src/trainer.py:126,139; .gitignore:226 --> The notes report (validation windows): "VSN collapse universal — all six models put weight = window-count (22 at h=3, 13 at h=12) on a single variable, second place ~10,000× smaller. Arbitrary variables (DMANEMP for CPI, etc.), no economic coherence." <!-- src: notes/notes_models.md:159 --> Attention appears only as a line in `figures/tft_*.png`, for the first validation window (forecasts from 2018-01). <!-- src: src/trainer.py:54-59; src/build_data_sets.py:22 --> The planned importance and attention analysis is not done. <!-- src: paper/spec.md:45 -->

## 5. Robustness

Not done. The crisis subsample in the spec (train 1990–2007, test 2008–2010, "if time permits") has no code, data or results. <!-- src: paper/spec.md:47; git grep -E "2008|2007" -- src scripts (only recession shading, src/data_pipeline.py:14) -->

## 6. What these results do not show

- Anything about 2020–2024, crises, or real-time forecasting.
- Whether the TFT is more or less accurate than the classical models. ARIMA had lower sample MSE than the TFT for CPIAUCSL and INDPRO (DM −2.14, −2.15), but see L1, L3 and Victor's "artifacts" note (section 3). <!-- src: data/processed/dm_test_results.parquet; notes/notes_models.md:91 -->
- That ARIMA is the strongest baseline: part of its edge may come from look-ahead (L1).
- Which inputs drive the TFT (section 4).
