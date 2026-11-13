"""Step 3: walk-forward forecasts of the classical benchmarks.

Models: random walk (RW), historical mean (Mean), ARIMA and VAR. ARIMA orders and the VAR lag length are
chosen once on the training sample; parameters are then re-estimated every month on an expanding window.
The *_frozen variants keep the training-sample parameters fixed, which mirrors how the TFT is used.

Reads data/processed/panel.parquet.
Writes results/forecasts_baselines.csv and results/baseline_specs.json.
"""
import argparse
import json
import warnings

import pandas as pd
import pmdarima
from statsmodels.tsa.api import VAR
from statsmodels.tsa.arima.model import ARIMA

import config
from data_pipeline import load_panel, split


def first_origin(index):
    """Position of the first forecast origin: the last month of the training sample."""
    return index.get_loc(config.period_bounds("val")[0]) - 1


def _records(index, origin, path, target, model, horizons):
    """Turn one forecast path (steps 1..max h) made at position `origin` into output rows."""
    return [{"target": target, "horizon": h, "model": model, "origin": index[origin], "date": index[origin + h],
             "forecast": float(path[h - 1])} for h in horizons if origin + h < len(index)]


def naive_forecasts(y, horizons, start):
    """Random walk (last observed value) and historical mean (mean of all values up to the origin)."""
    rows = []
    for origin in range(start, len(y) - 1):
        history = y.iloc[: origin + 1]
        rows += _records(y.index, origin, [history.iloc[-1]] * max(horizons), y.name, "RW", horizons)
        rows += _records(y.index, origin, [history.mean()] * max(horizons), y.name, "Mean", horizons)
    return rows


def select_arima_order(y_train):
    """Choose ARMA(p, q) for an already-stationary series by stepwise AIC search (pmdarima.auto_arima)."""
    best = pmdarima.auto_arima(y_train, d=0, seasonal=False, information_criterion="aic")
    return best.order, bool(best.with_intercept)


def arima_forecasts(y, order, with_intercept, horizons, start, refit):
    """Walk-forward ARIMA forecasts.

    The model is first estimated on y[:start + 1]. After each origin the next observation is appended;
    with refit=True the parameters are re-estimated on the enlarged sample, with refit=False only the
    state is updated and the initial parameters are kept.
    """
    model = "ARIMA" if refit else "ARIMA_frozen"
    result = ARIMA(y.iloc[: start + 1], order=order, trend="c" if with_intercept else "n").fit()
    rows = []
    for origin in range(start, len(y) - 1):
        path = result.forecast(max(horizons)).to_numpy()
        rows += _records(y.index, origin, path, y.name, model, horizons)
        result = result.append(y.iloc[origin + 1: origin + 2], refit=refit)
    return rows


def select_var_lag(system_train):
    """Information criteria for the VAR lag length on the training sample. Returns {criterion: lag}."""
    selected = VAR(system_train).select_order(config.VAR_MAX_LAGS).selected_orders
    return {name: int(lag) for name, lag in selected.items()}


def var_forecasts(system, lag, targets, horizons, start, refit):
    """Walk-forward VAR(lag) forecasts for the `targets` columns of `system`.

    With refit=True the VAR is re-estimated on system[:origin + 1] at every origin; with refit=False the
    coefficients estimated on system[:start + 1] are applied to the latest `lag` observations.
    """
    model = "VAR" if refit else "VAR_frozen"
    fitted = VAR(system.iloc[: start + 1]).fit(lag)
    rows = []
    for origin in range(start, len(system) - 1):
        if refit and origin > start:
            fitted = VAR(system.iloc[: origin + 1]).fit(lag)
        path = fitted.forecast(system.to_numpy()[origin + 1 - lag: origin + 1], steps=max(horizons))
        for name in targets:
            rows += _records(system.index, origin, path[:, system.columns.get_loc(name)], name, model, horizons)
    return rows


def main(until):
    warnings.filterwarnings("ignore", category=UserWarning)
    panel = load_panel(until).asfreq("MS")
    train = split(panel, "train")
    start = first_origin(panel.index)

    lag_criteria = select_var_lag(train[config.VAR_SYSTEM])
    specs = {"VAR": {"variables": config.VAR_SYSTEM, "lag": lag_criteria["bic"], "criteria": lag_criteria}, "ARIMA": {}}

    rows = []
    for target in config.TARGETS:
        order, with_intercept = select_arima_order(train[target])
        specs["ARIMA"][target] = {"order": list(order), "intercept": with_intercept}
        rows += naive_forecasts(panel[target], config.HORIZONS, start)
        for refit in (True, False):
            rows += arima_forecasts(panel[target], order, with_intercept, config.HORIZONS, start, refit)
    for refit in (True, False):
        rows += var_forecasts(panel[config.VAR_SYSTEM], lag_criteria["bic"], config.TARGETS, config.HORIZONS, start, refit)

    config.RESULTS_DIR.mkdir(exist_ok=True)
    pd.DataFrame(rows).to_csv(config.RESULTS_DIR / "forecasts_baselines.csv", index=False, date_format="%Y-%m-%d")
    (config.RESULTS_DIR / "baseline_specs.json").write_text(json.dumps(specs, indent=2) + "\n")
    print(json.dumps(specs))
    print(f"{len(rows)} forecasts, origins {panel.index[start]:%Y-%m} to {panel.index[-2]:%Y-%m}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--until", choices=["val", "test"], default="test", help="last period to forecast")
    main(parser.parse_args().until)
