# tft-macro-forecasting

**The question.** Does a deep-learning forecaster, the Temporal Fusion Transformer (TFT; Lim et al. 2021), predict US inflation, unemployment and industrial-production growth more accurately than classical time-series models, 3 and 12 months ahead, over 2020–2024? <!-- src: paper/spec.md:5,25 --> The project trains one TFT per series and horizon and compares it with three baselines on the monthly FRED-MD database (McCracken & Ng 2016). <!-- src: paper/spec.md:9,29-35 -->

Baselines:
- **Random walk (RW):** the forecast is the last observed value. <!-- src: paper/spec.md:31 -->
- **ARIMA:** one model per series, forecasting from its own past values and past errors. <!-- src: src/baseline.py:21-27 -->
- **VAR:** one linear model of five series together: the three targets, the federal funds rate and a term spread (T10YFFM). <!-- src: src/baseline.py:73; paper/spec.md:33 -->

## Status (2026-10-01)

- **Done:** data pipeline; three baselines; six TFTs (3 series × 2 horizons), checkpoints in `models/`; Diebold-Mariano (DM) statistics at h=3. All scored on the validation period 2018–2019 only. <!-- src: git ls-files models/; data/processed/Baseline_Results.csv; data/processed/dm_test_results.parquet --> DM, per Victor's notes, "checks whether one model's lower error is a genuine edge or just luck on this sample". <!-- src: notes/notes_models.md:76-77 -->
- **Not evaluated:** the test period 2020–2024; no test result exists. <!-- src: src/data_pipeline.py:72,76 -->
- **Not done:** DM p-values, the crisis robustness check, attention and importance tables. <!-- src: src/dm_test.py:52-59; paper/spec.md:45,47; src/trainer.py:139; .gitignore:226 -->

TODO(Victor): current timeline for the test-set run and the preprint.

**Quickstart:** install `requirements.txt` (Python 3.11), then run `bash scripts/reproduce.sh`. Details, including a workaround for two scripts that crash at the current commit: [docs/REPRODUCE.md](docs/REPRODUCE.md). <!-- src: scripts/reproduce.sh:4,23-25; src/evaluation_utils.py:16 -->

## Headline results (validation)

Root-mean-squared error (RMSE), lower is better, rounded from `data/processed/Baseline_Results.csv`. Units: monthly log growth for CPIAUCSL and INDPRO, monthly change in the unemployment rate for UNRATE. <!-- src: paper/spec.md:17-19; src/build_dataset.py:20-22,40 --> The three baselines share the same target dates: h=3 → 2018-03 to 2019-12 (n=22); h=12 → 2018-12 to 2019-12 (n=13). <!-- src: data/processed/predictions_unalligned.pkl (ARIMA and VAR date index); src/evaluation_utils.py:18-24 -->

| Series | h (months ahead) | RW | ARIMA* | VAR |
|---|---|---|---|---|
| CPIAUCSL | 3 | 0.002263 | 0.001370 | 0.001510 |
| CPIAUCSL | 12 | 0.002292 | 0.001494 | 0.001497 |
| UNRATE | 3 | 0.1732 | 0.1179 | 0.1120 |
| UNRATE | 12 | 0.1641 | 0.09460 | 0.09524 |
| INDPRO | 3 | 0.006274 | 0.005304 | 0.005355 |
| INDPRO | 12 | 0.007376 | 0.005443 | 0.005290 |

<!-- src: data/processed/Baseline_Results.csv:2-19 (RMSE column) -->
\* ARIMA was fitted on data that include the validation months, so its errors are not out-of-sample; the size and direction of the bias were not measured. <!-- src: src/baseline.py:27,29,79,93 -->

**TFT.** There is no TFT column because no TFT metrics file is committed ([FINDINGS §2](docs/FINDINGS.md#2-tft-validation)). Victor's notes: "TFT vs best baseline (RMSE): 3 wins, 3 losses, all margins ≤10%. Competitive, doesn't systematically beat tuned ARIMA/VAR." <!-- src: notes/notes_models.md:158 --> TODO(Victor): confirm this count — a load-mode re-run of the committed checkpoints (REPRODUCE step 7) gave a different count; commit the TFT metrics file.

What this shows: on validation, ARIMA and VAR have lower RMSE than the random walk in every cell. <!-- src: data/processed/Baseline_Results.csv:2-19 --> What it does not show: anything about 2020–2024, or whether the TFT beats the best baseline.

## Main limitations

1. **Validation only, and reused:** the same 2018–2019 months were used to stop and tune the TFTs and to pick the "best baseline". <!-- src: src/trainer.py:18-20; notes/notes_models.md:146,149; src/dm_test.py:46-49 -->
2. **ARIMA look-ahead** (above). ARIMA has the lowest baseline RMSE in 4 of 6 cells, so it is usually the TFT's comparison model. <!-- src: data/processed/Baseline_Results.csv:2-19; src/dm_test.py:46-49 -->
3. **Tiny samples:** 22 or 13 dates; DM tests have little power. <!-- src: src/dm_test.py:52-59; notes/notes_models.md:87-89 -->
4. **Revised data**, not the data available in real time. <!-- src: notes/notes_data.md:73 -->
5. **COVID** (2020–2021) falls inside the future test period; in Victor's words, "test RMSE will be dominated by a few extreme months". <!-- src: notes/notes_data.md:75-76 -->

Full list: [FINDINGS, Limitations](docs/FINDINGS.md#limitations-read-first).

## Repository map

- `src/`: pipeline scripts; `scripts/reproduce.sh` runs them in order, TFTs in load mode
- `data/raw/fred_md.csv`: frozen FRED-MD input; `data/processed/`: splits, forecasts, results
- `models/`: six TFT checkpoints and a toy model; `figures/`: forecast plots
- `notes/`, `paper/spec.md`: Victor's working notes and research design
- `docs/`: [REPRODUCE](docs/REPRODUCE.md), [FINDINGS](docs/FINDINGS.md), [DATA](docs/DATA.md), [DECISIONS](docs/DECISIONS.md), TFT-paper cheat sheets; `archive/`: unused scratch scripts
- [CHANGELOG.md](CHANGELOG.md): project history

<!-- src: git ls-files at 80e837f; scripts/reproduce.sh:2-3 -->

## References

- Lim, Arık, Loeff, Pfister (2021). *Temporal Fusion Transformers for Interpretable Multi-Horizon Time Series Forecasting.* International Journal of Forecasting.
- McCracken, Ng (2016). *FRED-MD: A Monthly Database for Macroeconomic Research.* Journal of Business & Economic Statistics.

<!-- src: README.md@80e837f:66-67 -->

## Author

Victor Grzeszick, secondary-school (Gymnasium) student, Bonn. <!-- src: README.md@80e837f:9; git log --format=%an --> TODO(Victor): add the school's current name (the old README said EMA-Gymnasium Bonn) and confirm this wording.
Licence: no licence file yet. TODO(Victor): choose one. <!-- src: git ls-files at 80e837f -->
