"""Panel construction and sample splits."""
import numpy as np
import pandas as pd

import config
from data_pipeline import build_panel, split


def make_transformed():
    index = pd.date_range("1985-01-01", "2026-03-01", freq="MS")
    rng = np.random.default_rng(1)
    df = pd.DataFrame(rng.normal(size=(len(index), 4)), index=index, columns=["full", "sparse_train", "sparse_test", "gap"])
    df.loc["1990-01":"1995-12", "sparse_train"] = np.nan  # 72 of 336 training months missing
    df.loc["2020-01":"2024-12", "sparse_test"] = np.nan   # missing only in the test period
    df.loc["2020-04", "gap"] = np.nan
    return df


def test_panel_covers_sample_without_gaps():
    panel, _ = build_panel(make_transformed())
    assert panel.index[0] == pd.Timestamp(config.SAMPLE_START)
    assert panel.index[-1] == pd.Timestamp(config.SAMPLE_END)
    assert not panel.isna().any().any()


def test_filter_looks_at_training_period_only():
    panel, dropped = build_panel(make_transformed())
    assert dropped == ["sparse_train"]
    assert "sparse_test" in panel.columns


def test_gaps_are_filled_with_the_previous_value():
    transformed = make_transformed()
    panel, _ = build_panel(transformed)
    assert panel.at[pd.Timestamp("2020-04-01"), "gap"] == transformed.at[pd.Timestamp("2020-03-01"), "gap"]


def test_splits_partition_the_sample():
    panel, _ = build_panel(make_transformed())
    parts = [split(panel, period) for period in ("train", "val", "test")]
    assert [len(p) for p in parts] == [336, 24, 60]
    assert pd.concat(parts).index.equals(panel.index)
    assert parts[0].index.max() < parts[1].index.min() and parts[1].index.max() < parts[2].index.min()


def test_period_assignment_requires_origin_after_fitting_sample():
    ts = pd.Timestamp
    assert config.assign_period(ts("2017-12-01"), ts("2018-03-01")) == "val"
    assert config.assign_period(ts("2017-11-01"), ts("2018-02-01")) is None   # origin inside the training sample
    assert config.assign_period(ts("2019-12-01"), ts("2020-12-01")) == "test"
    assert config.assign_period(ts("2019-06-01"), ts("2020-06-01")) is None   # origin inside the validation sample
    assert config.assign_period(ts("2019-01-01"), ts("2019-04-01")) == "val"
    assert config.assign_period(ts("2024-09-01"), ts("2024-12-01")) == "test"
