# tft-macro-forecasting

Undergraduate research project comparing Temporal Fusion Transformers (TFT) against
classical econometric baselines (ARIMA, VAR) for short-horizon forecasting of US
CPI inflation, and unemployment and industrial production growth, using the FRED-MD dataset
(McCracken & Ng 2016). Stage 1 targets a 6-page short paper and arXiv submission
by July 5, 2026.

**Author:** Victor Grzeszick, EMA-Gymnasium Bonn.
**Status:** Work in progress.

## Setup

Requires Python 3.11.

```bash
git clone https://github.com/victor-grzeszick/tft-macro-forecasting.git
cd tft-macro-forecasting
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Create a `.env` file in the repo root with your FRED API key:

FRED_API_KEY=your_key_here

Get a free FRED key at https://fredaccount.stlouisfed.org/apikeys.

## Data pipeline
The raw data gets imported via data/raw,

then first gets processed by src/build_dataset.py where the transformation codes are applied and where the date column is set as the index and is finally saved as data/processed/fred_md_transformed.parquet, 

afterwards in (src/validate.py) the ADF test is performed to check if the transformation worked and the data is now stationary

next in src/data_pipeline.py the data/processed/fred_md_transformed.parquet is used to split the data into 3 segments: 
- 1990-01 till 2017-12 for training (data/processed/X_train.parquet)
- 2018-01 till 2019-12 for validation and tuning of hyperparameters (data/processed/X_val.parquet)
- 2020-01 till 2024-12 for testing (saved in data/processed/X_test.parquet)
 
Additionally the target parameters CPIAUCSL, UNRATE and INDPRO are extracted and separated by the same time interval and saved in 
(data/processed/Y_train.parquet, data/processed/Y_val.parquet, data/processed/Y_test.parquet)

## References

- Lim, Arık, Loeff, Pfister (2021). *Temporal Fusion Transformers for Interpretable Multi-Horizon Time Series Forecasting.* International Journal of Forecasting.
- McCracken, Ng (2016). *FRED-MD: A Monthly Database for Macroeconomic Research.* Journal of Business & Economic Statistics.
