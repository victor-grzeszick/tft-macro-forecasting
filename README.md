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

## References

- Lim, Arık, Loeff, Pfister (2021). *Temporal Fusion Transformers for Interpretable Multi-Horizon Time Series Forecasting.* International Journal of Forecasting.
- McCracken, Ng (2016). *FRED-MD: A Monthly Database for Macroeconomic Research.* Journal of Business & Economic Statistics.
