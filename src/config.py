"""Project-wide constants: paths, sample periods, targets, horizons and fixed model settings.

Every script imports its settings from here so that the experimental design is stated in one place.
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW_FILE = ROOT / "data" / "raw" / "fred_md.csv"
PROCESSED_DIR = ROOT / "data" / "processed"
TRANSFORMED_FILE = PROCESSED_DIR / "fred_md_transformed.parquet"
PANEL_FILE = PROCESSED_DIR / "panel.parquet"
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = ROOT / "figures"
MODELS_DIR = ROOT / "models"

# Forecast targets (FRED-MD mnemonics) and horizons in months.
TARGETS = ["CPIAUCSL", "UNRATE", "INDPRO"]
HORIZONS = [3, 12]

# CPIAUCSL uses transformation code 5 (monthly inflation) instead of McCracken-Ng's code 6.
CODE_OVERRIDES = {"CPIAUCSL": 5}

# Variables of the VAR baseline: the three targets, the policy rate and the term spread.
VAR_SYSTEM = ["CPIAUCSL", "UNRATE", "INDPRO", "FEDFUNDS", "T10YFFM"]
VAR_MAX_LAGS = 12

# Sample periods (inclusive month bounds).
PERIODS = {
    "train": ("1990-01", "2017-12"),
    "val": ("2018-01", "2019-12"),
    "test": ("2020-01", "2024-12"),
}
SAMPLE_START = PERIODS["train"][0]
SAMPLE_END = PERIODS["test"][1]

# A series is dropped if more than this share of its training observations is missing.
MAX_MISSING_SHARE = 0.05

# TFT settings fixed a priori (not tuned).
SEED = 42
ENCODER_LENGTH = 24
QUANTILES = [0.1, 0.5, 0.9]
ATTENTION_HEADS = 4
MAX_EPOCHS = 200
PATIENCE = 15
BATCH_SIZE = 64

# Hyperparameter grid, searched once on UNRATE h=3 (validation loss) and then frozen for all six models.
TUNING_TARGET, TUNING_HORIZON = "UNRATE", 3
GRID = {"hidden_size": [8, 16, 32], "dropout": [0.1, 0.2], "learning_rate": [0.01, 0.03]}


def period_bounds(period):
    """Return (first month, last month) of a named period as Timestamps."""
    start, end = PERIODS[period]
    return pd.Timestamp(start), pd.Timestamp(end)


def assign_period(origin, target):
    """Name the evaluation period of a forecast made at `origin` for month `target`, or None.

    A forecast counts towards a period only if its target lies inside the period and its origin is no
    earlier than the last month before the period. All data used to fit or select a model for that
    period then lies at or before every forecast origin, so no forecast is made 'from inside' the
    sample the model was fitted on.
    """
    for period in ("val", "test"):
        start, end = period_bounds(period)
        if start <= target <= end and origin >= start - pd.DateOffset(months=1):
            return period
    return None
