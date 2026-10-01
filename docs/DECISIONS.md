# Decisions

One entry per design choice. Reasons are quoted from Victor's notes, spec, commit messages and the README before this rewrite; nothing is added. A date is when the choice first appears in his notes or git history. Each Source line is the entry's provenance; hidden comments add only extra evidence. Groups: data (D1–D5), baselines (D6–D11), evaluation (D12–D14), TFT (D15–D21), scope (D22–D23). Choices with no recorded reason are listed at the end.

**D1 · 2026-05-10 · Commit the downloaded FRED-MD file (frozen vintage).**
Why: "It is committed deliberately: FRED-MD is re-issued monthly with revised values, so replication on a later download would not reproduce these numbers exactly."
Source: previous README (`README.md@80e837f`, lines 28-29); `notes/notes_data.md:5-6`; commit cde9e0c.

**D2 · 2026-05-25 · Replace real GDP (GDPC1) with industrial production (INDPRO) as the third target.**
Why: "GDPC1 (real GDP) was dropped as a target: quarterly frequency yields too few training observations, and interpolation to monthly would introduce look-ahead bias. INDPRO serves as the monthly real-activity proxy in its place." In the change log: "Dropped GDP for INDPRO, marginal value but lots of more work". INDPRO is "The standard monthly real-activity indicator in Federal Reserve nowcasting models (Higgins 2014, Atlanta Fed GDPNow)."
Alternative considered earlier: "for the GDP model, aggregate monthly features to quarterly by within-quarter mean to match GDPC1's frequency. Trade-off: roughly 120 quarterly training observations is tight for TFT".
Source: `paper/spec.md:17,21`; `notes/notes_data.md:89`; commit 9775b99; earlier notes at `9775b99^:notes/notes_data.md:53`.

**D3 · 2026-05-25 · CPI as log first difference (code 5), not McCracken–Ng's code 6.**
Why: "Deliberate deviation from McCracken & Ng's code 6 (log second difference) for interpretability — the target is the inflation rate directly, comparable across the macro forecasting literature." Accepted cost: "CPIAUCSL passes at 5% but fails at 1% — the accepted code-5 trade-off".
Source: `paper/spec.md:18` (first in commit 889a25d); `notes/notes_data.md:42-44`; `src/build_dataset.py:40`.

**D4 · 2026-05-25 · Drop series with more than 5% missing values (only ACOGNO).**
Victor: "Dropped ACOGNO due to lack of data"; "only column above the threshold". TODO(Victor): reason for the 5% threshold.
Source: `notes/notes_data.md:77,90`; `src/data_pipeline.py:25-26`.

**D5 · 2026-05-25 · Split: train 1990–2017, validation 2018–2019, test 2020–2024, test touched once.**
Victor's rule: "The 2020–2024 test set is touched exactly once, after all model selection is complete. All tuning — TFT hyperparameters, ARIMA orders, VAR lag length — uses only training and validation data (1990–2019). No iterative refinement on test-set performance is permitted."
`paper/spec.md` has given validation as 2018–2019 since its first commit. TODO(Victor): reason for the split dates and the 1990 start.
Source: `paper/spec.md:9,39`; commit 889a25d. <!-- src: git show 889a25d:paper/spec.md (line 9) -->

**D6 · 2026-05-28 · Term spread T10YFFM instead of T10Y3M.**
Why: "T10Y3M (spec's term spread) is not a FRED-MD column. T10YFFM (10yr minus fed funds, the native term spread) is used instead".
Source: `notes/notes_data.md:15-17`; commit dba8318.

**D7 · 2026-05-26 · ARIMA uses a rolling update, not a frozen model.**
Why: "fit auto_arima once to pick orders, then walk through validation calling .update() with each new actual and re-forecasting. this deviates from the calendar's "frozen" suggestion; rolling-update is more faithful to true fixed-h forecasting, at the cost of more compute."
Alternative: the "frozen" scheme. As implemented, the first fit uses 1990–2019, not training data only ([FINDINGS, L1](FINDINGS.md#limitations-read-first)). <!-- src: src/baseline.py:79,93 -->
Source: `notes/notes_models.md:24-29` (date: line 167); `src/baseline.py:21-43`.

**D8 · 2026-05-26 · ARIMA without differencing or seasonality.**
Why: "d=0 — the series are already stationary from the transformation codes, so no further differencing. seasonal=False."
Source: `notes/notes_models.md:33-34`; `src/baseline.py:27`.

**D9 · 2026-05-28 · A five-variable VAR: the three targets plus FEDFUNDS and T10YFFM.**
Why: "FEDFUNDS = monetary policy stance, T10YFFM = term spread, both standard activity/inflation predictors in small monetary VARs."
Source: `notes/notes_models.md:42-43`; `src/baseline.py:73-74`.

**D10 · 2026-05-28 · VAR lag length p=1.**
Victor's notes: "Lag order: p=1, selected by BIC on the training set (AIC picked 3, HQIC 2 — BIC's heavier penalty gives the smaller, parsimonious order; sensible given 25 coefficients per lag)."
Alternatives (same note): p=3 (AIC), p=2 (HQIC). The code hard-codes p=1, and no committed code computes these criteria. <!-- src: git grep -i -E "select_order|bic|aic|hqic" over all revisions, *.py (no hits) --> TODO(Victor): commit the lag-selection code or its output.
Source: `notes/notes_models.md:44`; `src/baseline.py:57`.

**D11 · 2026-05-28 · Re-estimate the VAR at every forecast origin.**
Why: "expanding-window refit at each origin, lag frozen at 1 — scheme B, chosen for consistency with ARIMA's rolling update rather than freezing coefficients."
Alternative: freezing coefficients.
Source: `notes/notes_models.md:45`; `src/baseline.py:45-66`.

**D12 · 2026-06-10 · Score every model on the dates all of them cover.**
Why: "Without this the errors are measured over different periods and aren't comparable."
Alternative used before: RW scored on all 24 validation months, ARIMA and VAR on fewer. <!-- src: f9ded32:src/baseline.py:12-17,54 (RW on 2018-01 to 2019-12) and :38-40 (ARIMA on its own dates); dba8318:src/baseline.py:59-61 (VAR on its own dates) -->
Source: `notes/notes_models.md:64-72`; `src/evaluation_utils.py:18-24`; commit 770bc4d.

**D13 · 2026-05-28 · Test the TFT against the strongest baseline per target.**
Why: "VAR≈ARIMA with no clear winner, so the strongest classical baseline is chosen per-target for the DM test against TFT." In code, "strongest" means lowest validation RMSE.
Source: `notes/notes_models.md:58`; `src/dm_test.py:46-49`. <!-- src: date = git blame notes/notes_models.md -L58 (dba8318) -->

**D14 · 2026-06-10 · No DM test at h=12 on validation.**
Why: "h=12 dropped — 13 months is too short for the lag-11 correction (variance estimate degenerates), deferred to the test set."
Source: `notes/notes_models.md:86-87`; `src/dm_test.py:36`. <!-- src: date = 770bc4d:src/dm_test.py:23 -->

**D15 · 2026-06-11 · Early stopping for every FRED TFT run.**
Why, from the toy run on synthetic sine data: "400 epochs overfit onset — train falls, val rises to 0.059 → EarlyStopping mandatory for FRED runs."
Source: `notes/notes_models.md:103-106`; `src/trainer.py:18`; commit fa57351.

**D16 · 2026-06-13 · One TFT per target and horizon (six models).**
Why: "A separate model is trained per target and horizon, mirroring the per-series ARIMA baselines, so any RMSE gap is attributable to architecture rather than joint-vs-separate modelling; joint multi-target modelling is left to Stage 2."
Alternative: one joint model, left to Stage 2.
Source: `paper/spec.md:35`; commit 248d8b3.

**D17 · 2026-05-25 · Encoder length 24 months, not tuned.**
Why: "Encoder length fixed at 24 months a priori (≈2 years, the standard lookback in the macro forecasting literature) — not tuned, to limit researcher degrees of freedom." (The reason was written down on 2026-06-13.)
Source: `paper/spec.md:35` (encoder 24 since commit 889a25d; reason added in 248d8b3); `notes/notes_models.md:117-118`; `src/build_data_sets.py:21`.

**D18 · 2026-06-13 · Build validation data from the training dataset's settings.**
Why: "Val datasets via from_dataset(train_ds, df_full, min_prediction_idx=336): inherits train config + train-fit normalization (no leakage)".
Source: `notes/notes_models.md:122-123` (date: line 171); `src/build_data_sets.py:22`.

**D19 · 2026-06-21 · Tune the TFT architecture once, on UNRATE h=3, then freeze it.**
Why: "Architecture tuned once on UNRATE h=3, then frozen across all six models; learning rate found per-model via a manual sweep. That design limits researcher degrees of freedom (same logic as fixing encoder length a priori), and only let lr vary because it's optimization-specific, not capacity."
Alternative in the spec: random search over hidden size, dropout, learning rate and attention heads. Where code and checkpoints differ from this note: [FINDINGS, L7](FINDINGS.md#limitations-read-first).
Source: `notes/notes_models.md:146` (date: line 174); `paper/spec.md:35`.

**D20 · 2026-06-21 · Pick the learning rate by hand instead of `lr_find`.**
Why: "lr_find kept crashing on the torch 2.6 weights_only checkpoint-restore (EncoderNormalizer/numpy whack-a-mole), so I picked lr by manual trial, narrowing to ~0.023."
Source: `notes/notes_models.md:147`; commit b16fcbe.

**D21 · 2026-06-21 · Use the smallest model: hidden size 8, dropout 0.1.**
Victor: "It won probably because val_loss was nearly flat across the whole grid (~0.048–0.052), so the smallest model won on parsimony, not because it meaningfully beat the others." (Validation pinball loss, UNRATE h=3 grid; recorded only in the notes.)
Source: `notes/notes_models.md:149,153`; `src/trainer.py:117`.

**D22 · 2026-06-13 · Crisis robustness check (train 1990–2007, test 2008–2010) is optional.**
Victor: "if time permits … This requires retraining all TFT models on a second split; it is the first item to defer to a 4-page note or Stage 2 if the July 5 deadline tightens." Status: not done.
Source: `paper/spec.md:47`; commit 248d8b3. <!-- src: git log -S "if time permits" -- paper/spec.md = 248d8b3 (2026-06-13); no robustness code: git grep -E "2008|2007" -- src scripts (only recession shading) -->

**D23 · 2026-05-24 · Do not redistribute copyrighted papers.**
Victor: "Copyrighted material — read locally, don't redistribute".
Source: `.gitignore:221` (added in commit 4de9d0a); commit 2b3927b. <!-- src: git blame .gitignore line 221 = 4de9d0a; git show --stat 2b3927b -->

## Choices with no recorded reason

TODO(Victor): add a reason for each, or state that it was a default.

| Choice | Where |
|---|---|
| Horizons h=3 and h=12 | `paper/spec.md:25` |
| Point forecast = median; quantiles (spec 0.1/0.5/0.9, code uses the library default of 7) | `paper/spec.md:35`; `src/trainer.py:17` |
| All 125 series as TFT inputs | `src/build_data_sets.py:21` |
| Seed 42 | `src/trainer.py:92` |
| Batch size 64 | `src/trainer.py:114-115` |
| Early-stopping patience 15; at most 500 epochs | `src/trainer.py:18-19` |
| 4 attention heads, not tuned | checkpoint settings |

<!-- src: models/tft_*_h*.ckpt hyper_parameters (attention_head_size 4); no reason for these in notes/, paper/spec.md or commit messages -->
