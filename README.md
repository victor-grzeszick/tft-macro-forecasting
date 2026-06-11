# tft-macro-forecasting

Undergraduate research project comparing Temporal Fusion Transformers (TFT) against
classical econometric baselines (ARIMA, VAR) for short-horizon forecasting of US CPI
inflation, unemployment, and industrial production growth, using the FRED-MD dataset
(McCracken & Ng 2016). Stage 1 targets a 6-page short paper and arXiv submission
by July 5, 2026.

**Author:** Victor Grzeszick, EMA-Gymnasium Bonn.
**Status:** Work in progress. All results reported so far are validation-set metrics
(2018–2019); the test set (2020–2024) is reserved for the final evaluation and has
not been touched.

## Setup

Requires Python 3.11.

```bash
git clone https://github.com/victor-grzeszick/tft-macro-forecasting.git
cd tft-macro-forecasting
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The raw dataset is included: `data/raw/fred_md.csv` is the FRED-MD vintage of
2026-05, downloaded 2026-05-10 from the McCracken/St. Louis Fed FRED-MD page.
It is committed deliberately: FRED-MD is re-issued monthly with revised values,
so replication on a later download would not reproduce these numbers exactly.

## Data pipeline

`src/build_dataset.py` reads `data/raw/fred_md.csv`, sets the date column as index,
applies the McCracken-Ng transformation codes per series, and writes
`data/processed/fred_md_transformed.parquet`.

`src/validate.py` runs ADF tests on the transformed series (INDPRO, CPIAUCSL,
UNRATE, FEDFUNDS, T10YFFM) to confirm the transformations produced stationary data.

`src/data_pipeline.py` drops columns with >5% missing values, splits the data into

- train: 1990-01 – 2017-12 (`X_train.parquet`)
- validation (hyperparameter tuning): 2018-01 – 2019-12 (`X_val.parquet`)
- test (final evaluation only): 2020-01 – 2024-12 (`X_test.parquet`)

and extracts the three targets (CPIAUCSL, UNRATE, INDPRO) into matching
`Y_train/Y_val/Y_test.parquet` files, all under `data/processed/`.

## Replication

Run from the repo root, in this order:

1. `python src/build_dataset.py` — transformed dataset → `data/processed/fred_md_transformed.parquet`
2. `python src/validate.py` — ADF stationarity checks (prints results, no files)
3. `python src/data_pipeline.py` — train/val/test splits → six `X_/Y_` parquets
4. `python src/baseline.py` — RW, ARIMA and VAR forecasts → `data/processed/predictions_unalligned.pkl`
5. `python src/align_results.py` — aligned RMSE/MAE per model/target/horizon → `data/processed/Baseline_Results.csv`
6. `python src/dm_test.py` — Diebold-Mariano tests (h=3, validation) → `data/processed/dm_test_results.parquet`
7. `python src/plot_baselines.py` — predicted-vs-actual plots → `figures/*.png`

All random seeds are fixed; rerunning the pipeline on the committed raw data
reproduces the reported numbers exactly.

## References

- Lim, Arık, Loeff, Pfister (2021). *Temporal Fusion Transformers for Interpretable Multi-Horizon Time Series Forecasting.* International Journal of Forecasting.
- McCracken, Ng (2016). *FRED-MD: A Monthly Database for Macroeconomic Research.* Journal of Business & Economic Statistics.