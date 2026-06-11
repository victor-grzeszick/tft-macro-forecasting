# Data notes — TFT macro forecasting

## Data source

'current.csv' from https://research.stlouisfed.org/econ/mccracken/fred-databases/,
downloaded 2026-05-10, saved as data/raw/fred_md.csv and committed (frozen vintage).

Raw shape 808 rows × 127 columns, monthly, 1959-01 to 2026-03 (the committed
vintage). FRED-MD is a large monthly macroeconomic database of ~126 US series,
maintained by Michael McCracken at the St. Louis Fed, described in McCracken & Ng (2016).

Layout: columns = variable names (e.g. RPI); row 0 = transformation codes;
each following row = one month, first column the date, remaining cells the values.

Note: T10Y3M (spec's term spread) is not a FRED-MD column. T10YFFM (10yr minus
fed funds, the native term spread) is used instead — affects the VAR system and
the validation list below.

## Pipeline

build_dataset.py — reads data/raw/fred_md.csv, sets the date index, applies the
transformation codes per series, writes data/processed/fred_md_transformed.parquet.

validate.py — ADF tests on the transformed series (INDPRO, CPIAUCSL, UNRATE,
FEDFUNDS, T10YFFM) to confirm stationarity.

data_pipeline.py — drops >5%-missing columns, splits train/val/test
(1990–2017 / 2018–2019 / 2020–2024), extracts the three targets
(CPIAUCSL, UNRATE, INDPRO), writes the six X_/Y_ parquets.

## Transformation codes

Codes appearing in FRED-MD: 1 = level, 2 = first difference, 4 = log,
5 = log first difference, 6 = second log difference, 7 = difference of
simple growth rate.

Targets: INDPRO code 5; CPIAUCSL code 5 (deliberate deviation from
McCracken-Ng's code 6, for interpretability as an inflation rate); UNRATE code 2.

## Validation results

INDPRO and UNRATE pass ADF decisively; FEDFUNDS clearly; CPIAUCSL passes at
5% but fails at 1% — the accepted code-5 trade-off; T10YFFM passes 5%
marginally (mean-reverting spread, unrelated to the code-5 issue).

CPIAUCSL:
  ADF statistic: -3.1210
  p-value:       0.025037
  critical 5%:   -2.8652

INDPRO:
  ADF statistic: -7.3170
  p-value:       0.000000
  critical 5%:   -2.8652

UNRATE:
  ADF statistic: -15.5832
  p-value:       0.000000
  critical 5%:   -2.8652

FEDFUNDS:
  ADF statistic: -4.5576
  p-value:       0.000155
  critical 5%:   -2.8706

T10YFFM:
  ADF statistic: -2.8747
  p-value:       0.048367
  critical 5%:   -2.8703

## Known limitations

1) Revised FRED data, not real-time vintages — limits real-time forecasting claims.
2) Series have different start dates (consumer sentiment, VIX, … don't reach back to 1959).
3) COVID outliers (2020–2021) fall inside the test set — test RMSE will be
   dominated by a few extreme months; interpret final results accordingly.
4) ACOGNO dropped (>5% missing) — only column above the threshold.

## Decisions made

- GDPC1 dropped as target: quarterly frequency → too few training observations
  and interpolation would introduce look-ahead bias.
- INDPRO adopted instead: the standard monthly real-activity proxy.
- Single monthly frequency throughout; ~360 monthly training observations.

## Change log

2026-05-25: Completed W2 S1. Built data pipeline, validated INDPRO and CPIAUCSL with ADF.
2026-05-25: Dropped GDP for INDPRO, marginal value but lots of more work
2026-05-25: Dropped ACOGNO due to lack of data
2026-05-26: Added UNRATE to the validation and confirmed via heatmap and plotting that transformation worked
2026-05-28: Added FEDFUNDS and T10YFFM to the validation and added data_pipeline