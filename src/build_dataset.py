"""Step 1: apply the FRED-MD transformation codes to the raw vintage.

Reads data/raw/fred_md.csv and writes data/processed/fred_md_transformed.parquet.
"""
import numpy as np
import pandas as pd

import config


def transform(series, code):
    """Apply one McCracken-Ng (2016) transformation code to a series of levels.

    1 level, 2 first difference, 3 second difference, 4 log, 5 log first difference,
    6 log second difference, 7 first difference of the one-period growth rate.
    """
    if code == 1:
        return series
    if code == 2:
        return series.diff()
    if code == 3:
        return series.diff().diff()
    if code == 4:
        return np.log(series)
    if code == 5:
        return np.log(series).diff()
    if code == 6:
        return np.log(series).diff().diff()
    if code == 7:
        return series.pct_change(fill_method=None).diff()
    raise ValueError(f"unknown transformation code {code}")


def load_raw(path=config.RAW_FILE):
    """Read a FRED-MD csv. Returns (levels indexed by month, transformation code per series)."""
    raw = pd.read_csv(path)
    codes = raw.iloc[0, 1:].astype(int)
    levels = raw.iloc[1:].dropna(how="all").copy()
    levels.index = pd.to_datetime(levels.pop("sasdate"), format="%m/%d/%Y")
    levels.index.name = "date"
    return levels.apply(pd.to_numeric, errors="coerce"), codes


def apply_transformations(levels, codes, overrides=None):
    """Transform every column of `levels` with its code; `overrides` replaces codes by series name."""
    codes = codes.copy()
    for name, code in (overrides or {}).items():
        codes[name] = code
    return pd.DataFrame({name: transform(levels[name], codes[name]) for name in levels.columns})


def main():
    levels, codes = load_raw()
    transformed = apply_transformations(levels, codes, config.CODE_OVERRIDES)
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    transformed.to_parquet(config.TRANSFORMED_FILE)
    print(f"transformed {transformed.shape[1]} series, {transformed.index[0]:%Y-%m} to {transformed.index[-1]:%Y-%m}")


if __name__ == "__main__":
    main()
