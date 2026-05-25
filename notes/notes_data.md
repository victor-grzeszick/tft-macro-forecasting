# Data notes — TFT macro forecasting

## Data source

The data is the 'current.csv' from the website https://research.stlouisfed.org/econ/mccracken/fred-databases/ downloaded on 2026-05-10

The raw shape is 808 rows × 127 columns, the data is monthly, the time span runs from January 1959 to (currently) March 2026, and FRED-MD itself is "large, monthly frequency, macroeconomic database" of about 126 US time series, maintained by Michael McCracken at the St. Louis Fed and described in McCracken and Ng (2016).

It's a big table, with in which the index of the colums are the variable name, e.g RPI, in the 0. row are the transformation codes, used to make the data stationary
From the 1. row onwards each row follows the same pattern, the first column represents the month and afterwards each cell gives the data from the given month for the given variable indexed for its row


## Pipeline

build_dataset: it imports the raw data from the data/raw/fred_md.csv - makes the dates the index of each row - then applies the transformation codes for each cell accordingly - exports the table into data/processed/fred_md_transformed.parquet

validate: it validates if in data/processed/fred_md_transformed.parquet the data is indeed stationary by using the ADF test. 
validate.py lives in src/ and runs ADF on INDPRO and CPIAUCSL specifically

## Transformation codes

The following codes appear in FRED-MD: 1, 2, 4, 5, 6, 7. 
code 1 is level, code 2 is first difference,
code 4 is log, code 5 is log first difference, code 6 is second log
difference, code 7 is the difference of the simple growth rate.

INDPRO uses code 5, CPIAUCSL uses code 5 (deliberate deviation from McCracken-Ng for interpretability), UNRATE uses code 2.

## Validation results

INDPRO pass the ADF test decisively well, CPIAUCSL passes 5% however fails 1%, known trade - off for choosing code 5: 
CPIAUCSL:
  ADF statistic: -3.1210
  p-value:       0.025037
  critical 5%:   -2.8652

INDPRO:
  ADF statistic: -7.3170
  p-value:       0.000000
  critical 5%:   -2.8652

## Known limitations

1) using revised FRED data
rather than real-time vintage data — limits real-time forecasting claims.
2) Different series have different start dates — consumer sentiment,
VIX, and others don't go back to 1959. 
3) COVID outliers in 2020-2021 are extreme and will need methodological attention before
modeling.

## Open decisions

GDPC1 isn't in FRED-MD because FRED-MD is monthly and GDP is quarterly. Decision: keep FRED-MD throughout; for the GDP model, aggregate monthly features to quarterly by within-quarter mean to match GDPC1's frequency. Trade-off: roughly 120 quarterly training observations is tight for TFT, disclosed as a limitation in the paper.

## Change log

"2026-05-25: Completed W2 S1. Built data pipeline, validated INDPRO and CPIAUCSL
with ADF."