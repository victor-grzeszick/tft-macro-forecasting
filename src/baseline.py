"""Random Walk, rolling ARIMA and expanding-window VAR(1) forecasts for the validation period.

Writes data/processed/predictions_unalligned.pkl (one row per target, horizon and model)."""
import pandas as pd
import numpy as np
import pmdarima
import os
from statsmodels.tsa.api import VAR as var

def rw_forecast(y_history, h):
    """Random-walk forecast: shift `y_history` forward by h rows."""
    return y_history.shift(h)

def rw_forecast_rolling(name, h, df, date_start, date_end):
    """Return the h-step random-walk forecast of column `name` from date_start to date_end."""
    df_ = rw_forecast(df, h)
    df_prediction = df_.loc[date_start: date_end, name]

    return df_prediction

def arima_rolling_forecast(name, train_series, val_series, h):
    """Fit auto_arima (d=0, seasonal=False) on train_series[name]. For each value in val_series: forecast h steps, store
    the h-th forecast under the date h rows after the values seen so far in concat(train_series, val_series), then
    update the model with that value. Returns the forecasts whose dates lie in val_series."""
    train_series = train_series[name]
    val_series = val_series[name]
    model = pmdarima.auto_arima(train_series, d=0, seasonal=False)

    full = pd.concat([train_series, val_series])
    cutoff = len(train_series)   # position of first unseen month

    preds = {}
    for actual in val_series:
        pred = model.predict(n_periods=h)
        target_pos = cutoff + h - 1
        if target_pos < len(full):
            preds[full.index[target_pos]] = np.asarray(pred)[-1]
        model.update(actual)
        cutoff += 1

    pred_series = pd.Series(preds)
    common = pred_series.index.intersection(val_series.index)
    return pred_series[common]

def var_rolling_forecast(name, df_train, df_val, h):
    """For each row of df_val, fit VAR(1) on all earlier rows of concat(df_train, df_val) (expanding window), forecast
    h steps, and store the forecast of column `name` under the date h rows after the window in that concatenation.
    Returns the forecasts whose dates lie in df_val."""

    full = pd.concat([df_train, df_val])
    cutoff = len(df_train)
    col = full.columns.get_loc(name)

    preds = {}
    for i in range(len(df_val)):
        history = full.iloc[:cutoff]
        fitted = var(history).fit(1)
        target_pos = cutoff + h - 1
        if target_pos < len(full):
            fc = fitted.forecast(history.values[-1:], steps=h)
            preds[full.index[target_pos]] = fc[-1, col]
        cutoff += 1
    
    pred_series = pd.Series(preds)
    common = pred_series.index.intersection(df_val.index)
    return pred_series[common]

def main():
    """Run RW, ARIMA and VAR (on CPIAUCSL, UNRATE, INDPRO, FEDFUNDS, T10YFFM) for each target
    and horizon 3 and 12, and pickle the forecasts to data/processed/predictions_unalligned.pkl."""
    df_train = pd.read_parquet("data/processed/Y_train.parquet")
    df_val = pd.read_parquet("data/processed/Y_val.parquet")
    system_train = pd.read_parquet("data/processed/X_train.parquet").loc[:, ["CPIAUCSL","UNRATE","INDPRO","FEDFUNDS","T10YFFM"]]
    system_val = pd.read_parquet("data/processed/X_val.parquet").loc[:, ["CPIAUCSL","UNRATE","INDPRO","FEDFUNDS","T10YFFM"]]

    system_train.index.freq = "MS"
    system_val.index.freq = "MS"

    df = pd.concat([df_train, df_val])
    
    targets = ["CPIAUCSL", "UNRATE", "INDPRO"]
    horizons = [3, 12]

    predictions = []

    for name in targets:
        for h in horizons:
            pred = rw_forecast_rolling(name, h, df, "2018-01", "2019-12")
            predictions.append({"target": name, "horizon": h,"model": "RW", "prediction": pred})

    for name in targets:
        for h in horizons:
            pred = arima_rolling_forecast(name, df, df_val, h)
            predictions.append({"target": name, "horizon": h,"model": "ARIMA", "prediction": pred})

    for name in targets:
        for h in horizons:
            pred = var_rolling_forecast(name, system_train, system_val, h)
            predictions.append({"target": name, "horizon": h,"model": "VAR", "prediction": pred})

    predictions_df = pd.DataFrame(predictions)

    os.makedirs("data/processed", exist_ok=True)
    predictions_df.to_pickle("data/processed/predictions_unalligned.pkl")


if __name__ == "__main__":
    main()