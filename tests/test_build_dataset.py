"""Transformation codes and parsing of the raw FRED-MD file."""
import numpy as np
import pandas as pd
import pytest

import config
from build_dataset import apply_transformations, load_raw, transform

LEVELS = pd.Series([100.0, 110.0, 121.0, 121.0, 133.1])


def test_code_1_keeps_levels():
    pd.testing.assert_series_equal(transform(LEVELS, 1), LEVELS)


def test_code_2_is_first_difference():
    assert transform(LEVELS, 2).tolist()[1:] == pytest.approx([10.0, 11.0, 0.0, 12.1])


def test_code_5_is_log_growth():
    growth = transform(LEVELS, 5)
    assert np.isnan(growth[0])
    assert growth.tolist()[1:] == pytest.approx([np.log(1.1), np.log(1.1), 0.0, np.log(1.1)])


def test_code_6_is_change_in_log_growth():
    assert transform(LEVELS, 6).tolist()[2:] == pytest.approx([0.0, -np.log(1.1), np.log(1.1)])


def test_code_7_is_change_in_growth_rate():
    assert transform(LEVELS, 7).tolist()[2:] == pytest.approx([0.0, -0.1, 0.1])


def test_transformations_use_only_current_and_past_values():
    """Changing a later observation must not change any earlier transformed value."""
    changed = LEVELS.copy()
    changed.iloc[-1] = 999.0
    for code in range(1, 8):
        pd.testing.assert_series_equal(transform(LEVELS, code).iloc[:-1], transform(changed, code).iloc[:-1])


def test_unknown_code_is_rejected():
    with pytest.raises(ValueError):
        transform(LEVELS, 8)


def test_override_replaces_code():
    levels = pd.DataFrame({"A": LEVELS, "B": LEVELS})
    out = apply_transformations(levels, pd.Series({"A": 6, "B": 6}), {"A": 5})
    pd.testing.assert_series_equal(out["A"], transform(LEVELS, 5), check_names=False)
    pd.testing.assert_series_equal(out["B"], transform(LEVELS, 6), check_names=False)


def test_raw_file_is_parsed_as_monthly_levels():
    levels, codes = load_raw()
    assert levels.index.is_monotonic_increasing and levels.index.is_unique
    assert (levels.index.day == 1).all()
    assert levels.index[0] == pd.Timestamp("1959-01-01")
    assert set(codes.unique()) <= set(range(1, 8))
    assert set(config.VAR_SYSTEM) <= set(levels.columns)
    assert levels.loc["2000-01-01", "UNRATE"] == pytest.approx(4.0)  # published US unemployment rate, Jan 2000


def test_targets_have_the_intended_meaning():
    """INDPRO and CPIAUCSL become monthly log growth, UNRATE a monthly change in percentage points."""
    levels, codes = load_raw()
    out = apply_transformations(levels, codes, config.CODE_OVERRIDES)
    month, previous = pd.Timestamp("2000-02-01"), pd.Timestamp("2000-01-01")
    for name in ["CPIAUCSL", "INDPRO"]:
        assert out.at[month, name] == pytest.approx(np.log(levels.at[month, name] / levels.at[previous, name]))
    assert out.at[month, "UNRATE"] == pytest.approx(levels.at[month, "UNRATE"] - levels.at[previous, "UNRATE"])
