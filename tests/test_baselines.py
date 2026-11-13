"""Walk-forward benchmarks: correct timing and no use of future data."""
import numpy as np
import pandas as pd
import pytest

import config
from baselines import arima_forecasts, first_origin, naive_forecasts, select_var_lag, var_forecasts
from data_pipeline import split

HORIZONS = [3, 12]


def frame(rows):
    return pd.DataFrame(rows).set_index(["model", "horizon", "origin"]).sort_index()


def all_forecasts(panel, start):
    y = panel["UNRATE"]
    rows = naive_forecasts(y, HORIZONS, start)
    for refit in (True, False):
        rows += arima_forecasts(y, (1, 0, 0), True, HORIZONS, start, refit)
        rows += var_forecasts(panel, 1, ["UNRATE"], HORIZONS, start, refit)
    return frame(rows)


@pytest.fixture
def short_panel(synthetic_panel):
    """Training sample plus eight months, to keep the walk-forward loops fast."""
    return synthetic_panel.iloc[: 336 + 8].asfreq("MS")


def test_first_origin_is_last_training_month(synthetic_panel):
    assert synthetic_panel.index[first_origin(synthetic_panel.index)] == pd.Timestamp("2017-12-01")


def test_target_date_is_h_months_after_origin(short_panel):
    out = all_forecasts(short_panel, first_origin(short_panel.index)).reset_index()
    months = (out["date"].dt.year - out["origin"].dt.year) * 12 + out["date"].dt.month - out["origin"].dt.month
    assert (months == out["horizon"]).all()
    assert out["date"].max() <= short_panel.index[-1]


def test_random_walk_and_mean_values(short_panel):
    start = first_origin(short_panel.index)
    out = frame(naive_forecasts(short_panel["UNRATE"], HORIZONS, start))
    origin = short_panel.index[start + 2]
    history = short_panel["UNRATE"].loc[:origin]
    assert out.at[("RW", 3, origin), "forecast"] == pytest.approx(history.iloc[-1])
    assert out.at[("Mean", 3, origin), "forecast"] == pytest.approx(history.mean())


def test_no_forecast_uses_data_after_its_origin(short_panel):
    """Scrambling every observation after month k must leave all forecasts with origin <= k unchanged."""
    start = first_origin(short_panel.index)
    cut = start + 3
    scrambled = short_panel.copy()
    scrambled.iloc[cut + 1:] = np.random.default_rng(2).normal(scale=50, size=scrambled.iloc[cut + 1:].shape)
    before = all_forecasts(short_panel, start)["forecast"]
    after = all_forecasts(scrambled, start)["forecast"]
    early = before.index.get_level_values("origin") <= short_panel.index[cut]
    assert early.sum() > 0 and (~early).sum() > 0
    np.testing.assert_allclose(before[early], after[early], rtol=1e-10)
    assert not np.allclose(before[~early], after[~early])


def test_frozen_models_keep_training_parameters(short_panel):
    """With a constant series appended, frozen and refitted models must eventually differ."""
    start = first_origin(short_panel.index)
    out = all_forecasts(short_panel, start)["forecast"]
    first = short_panel.index[start]
    for model in ("ARIMA", "VAR"):  # identical at the first origin, where both were estimated on the same data
        assert out[(model, 3, first)] == pytest.approx(out[(f"{model}_frozen", 3, first)], rel=1e-6)
    last = short_panel.index[-4]
    assert out[("VAR", 3, last)] != pytest.approx(out[("VAR_frozen", 3, last)], rel=1e-9)


def test_var_one_step_forecast_matches_manual_ols(short_panel):
    """A VAR(1) forecast is the fitted linear map applied to the last observation."""
    start = first_origin(short_panel.index)
    history = short_panel.iloc[: start + 1].to_numpy()
    X = np.column_stack([np.ones(len(history) - 1), history[:-1]])
    coefficients = np.linalg.lstsq(X, history[1:], rcond=None)[0]
    path = [history[-1]]
    for _ in range(3):
        path.append(np.concatenate([[1.0], path[-1]]) @ coefficients)
    out = frame(var_forecasts(short_panel, 1, ["UNRATE"], [3], start, refit=False))
    column = short_panel.columns.get_loc("UNRATE")
    assert out.at[("VAR_frozen", 3, short_panel.index[start]), "forecast"] == pytest.approx(path[-1][column])


def test_var_lag_selection_recovers_true_order(synthetic_panel):
    assert select_var_lag(split(synthetic_panel, "train").asfreq("MS"))["bic"] == 1
