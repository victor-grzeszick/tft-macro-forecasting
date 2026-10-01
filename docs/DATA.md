# Data

## Source and vintage

- **FRED-MD** (McCracken & Ng 2016) is a large monthly database of US macroeconomic series, maintained at the St. Louis Fed. <!-- src: notes/notes_data.md:9-10 -->
- The input is McCracken's `current.csv` from https://research.stlouisfed.org/econ/mccracken/fred-databases/, downloaded 2026-05-10 and committed as `data/raw/fred_md.csv`. Victor calls it a "frozen vintage". <!-- src: notes/notes_data.md:5-6 -->
- Shape: 808 rows × 127 columns, i.e. one row of transformation codes, then 807 months from 1959-01 to 2026-03; 126 series plus the date column `sasdate`. <!-- src: notes/notes_data.md:8,12-13; data/raw/fred_md.csv -->
- `paper/spec.md` calls it the "revised vintage, 2026-05". <!-- src: paper/spec.md:9 --> TODO(Victor): name the matching file in McCracken's vintage archive, so others can cross-check without this repo.
- These are revised data, not what forecasters saw in real time ([FINDINGS, L4](FINDINGS.md#limitations-read-first)). SHA-256 check: [REPRODUCE.md](REPRODUCE.md#2-check-the-data-file).

## Transformations (`src/build_dataset.py`)

Row 0 of the CSV gives each series a McCracken–Ng code, and the script applies it. These six codes are the only ones in the file. <!-- src: src/build_dataset.py:38,52,62; data/raw/fred_md.csv row 0 --> Output: `fred_md_transformed.parquet`, 807 months × 126 series. <!-- src: src/build_dataset.py:60; data/processed/fred_md_transformed.parquet -->

| Code | Transformation |
|---|---|
| 1 | none (level) |
| 2 | first difference |
| 4 | natural log |
| 5 | first difference of the log |
| 6 | second difference of the log |
| 7 | first difference of the growth rate x_t / x_{t−1} − 1 |

<!-- src: src/build_dataset.py:8-30 -->

| Target | Code used | Meaning |
|---|---|---|
| CPIAUCSL | 5 (the file says 6; overridden) | monthly inflation rate |
| UNRATE | 2 | monthly change in the unemployment rate |
| INDPRO | 5 | monthly industrial-production growth |

<!-- src: src/build_dataset.py:40; data/raw/fred_md.csv row 0; paper/spec.md:17-19 -->
Why CPI uses code 5: [DECISIONS.md](DECISIONS.md), D3.

## Sample, filter and split (`src/data_pipeline.py`)

1. Keep 1990-01 to 2024-12 (420 months). <!-- src: src/data_pipeline.py:23,28 -->
2. Drop every column with more than 5% missing values in that range. Only ACOGNO goes (6.2% missing), leaving 125 of 126 series. <!-- src: src/data_pipeline.py:25-26,30; notes/notes_data.md:77; data/processed/fred_md_transformed.parquet -->
   The filter window includes the test years. Computed on 1990–2017 alone, it drops the same single column, so the kept set does not change for this vintage. <!-- src: data/processed/fred_md_transformed.parquet (share missing over 1990-01..2017-12: only ACOGNO above 5%) -->
3. Split by date:

| Split | Months | Rows | Files in `data/processed/` |
|---|---|---|---|
| train | 1990-01 to 2017-12 | 336 | `X_train`, `Y_train` |
| validation | 2018-01 to 2019-12 | 24 | `X_val`, `Y_val` |
| test | 2020-01 to 2024-12 | 60 | `X_test`, `Y_test` |

<!-- src: src/data_pipeline.py:32-38,70-76; data/processed/X_{train,val,test}.parquet shapes -->
`X_*` hold all 125 kept series, targets included; `Y_*` hold the three targets. `X_test` has 3 missing cells (CP3Mx, COMPAPFFx); `X_train` and `X_val` have none. No code reads `X_test` or `Y_test` yet. <!-- src: src/data_pipeline.py:65-76; data/processed/X_*.parquet (NaN counts); git grep -E "X_test|Y_test" -- src scripts (only the writes at src/data_pipeline.py:72,76) -->
The script also draws target plots with recession shading and a correlation heatmap; these are shown, not saved. <!-- src: src/data_pipeline.py:48-63 -->

## What each model sees

- **RW and ARIMA:** the target's own history (`Y_*`). <!-- src: src/baseline.py:71-72,79,88,93 -->
- **VAR:** CPIAUCSL, UNRATE, INDPRO, FEDFUNDS and T10YFFM, as transformed (from `X_*`). <!-- src: src/baseline.py:73-74 -->
- **TFT:** all 125 series as past-only inputs (the target's own past included), plus an integer time index as the only input known in advance; encoder length 24 months. <!-- src: src/build_data_sets.py:9,21; notes/notes_models.md:117-120 --> Victor's notes: "scalers train-fit and inherited by val via from_dataset (no leakage)". <!-- src: notes/notes_models.md:127 -->

## Artifact flow

Run from the repo root. Files are in `data/processed/` unless a path is given. † = gitignored, not committed. "unalligned" is the spelling used in the code.

| Script (`src/`) | Reads | Writes |
|---|---|---|
| `build_dataset.py` | `data/raw/fred_md.csv` | `fred_md_transformed.parquet` |
| `validate.py` | `fred_md_transformed.parquet` | screen only |
| `data_pipeline.py` | `fred_md_transformed.parquet` | `X_`/`Y_` `{train,val,test}.parquet` |
| `baseline.py` | `X_train`, `X_val`, `Y_train`, `Y_val` | `predictions_unalligned.pkl` |
| `align_results.py` (crashes at the current commit) | `Y_val`, `predictions_unalligned.pkl` | `Baseline_Results.csv` |
| `plot_baselines.py` (crashes at the current commit) | `Y_val`, `predictions_unalligned.pkl` | `figures/{target}_h{h}.png` |
| `build_data_sets.py` | `X_train`, `X_val` | nothing; `trainer.py` imports it |
| `trainer.py` (load) | `X_train`, `X_val`, `predictions_unalligned.pkl` (dates only), `models/tft_*.ckpt`, `Baseline_Results.csv` | `figures/tft_*.png`, `predictions_unalligned_tft.pkl`†, `Baseline_Results_with_tft.csv`†, `Full_results.csv`† |
| `dm_test.py` | `predictions_unalligned.pkl`, `predictions_unalligned_tft.pkl`†, `Baseline_Results.csv`, `Y_val` | `dm_test_results.parquet` |
| `pytorchDummy.py` | `models/tft_toy.ckpt` (synthetic data) | `figures/toy_idx{0,10,20}.png` |

<!-- src: src/build_dataset.py:36,60; src/validate.py:18; src/data_pipeline.py:22,70-76; src/baseline.py:71-74,104; src/align_results.py:11-12,38; src/plot_baselines.py:9-10,30; src/evaluation_utils.py:16; src/build_data_sets.py:43-52; src/trainer.py:8,93-94,107,135-139; src/dm_test.py:26-32,64; src/pytorchDummy.py:38,47 -->
Crashes and workaround: [REPRODUCE.md](REPRODUCE.md#4-rebuild-baseline_resultscsv-and-the-baseline-figures). `.gitignore` lists `data/processed/`, so new outputs there (the † files) are not committed, while the ten files already tracked stay tracked. <!-- src: .gitignore:226; git ls-files data/processed --> TODO(Victor): decide whether results should live in a separate tracked folder.
