import pandas as pd
import numpy as np
import pmdarima
import os

def rw_forecast(y_history, h):
    return y_history.shift(h)

def evaluate(y_history, y_val):
    return (abs(y_history - y_val).mean(), ((y_history - y_val)**2).mean()**0.5)

def rw_forcast_and_evaluation(name, h, df, df_val, date_start, date_end):
    df_ = rw_forecast(df, h)
    df_prediction = df_.loc[date_start: date_end, name]
    df_val_new = df_val[name]

    error = evaluate(df_prediction, df_val_new)

    return df_prediction, error

def arima_rolling_forecast_and_evaluation(name, train_series, val_series, h):
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
    return pred_series[common], evaluate(pred_series[common], val_series[common])

def main():
    df_train = pd.read_parquet("data/processed/Y_train.parquet")
    df_val = pd.read_parquet("data/processed/Y_val.parquet")

    df = pd.concat([df_train, df_val])
    
    targets = ["CPIAUCSL", "UNRATE", "INDPRO"]
    horizons = [3, 12]

    results = []
    for name in targets:
        for h in horizons:
            _, error = rw_forcast_and_evaluation(name, h, df, df_val, "2018-01", "2019-12")
            mae, rmse = error
            results.append({"target": name, "horizon": h, "MAE": mae, "RMSE": rmse, "model": "RW"})

    for name in targets:
        for h in horizons:
            _, error = arima_rolling_forecast_and_evaluation(name, df, df_val, h)
            mae, rmse = error
            results.append({"target": name, "horizon": h, "MAE": mae, "RMSE": rmse, "model": "ARIMA"})

    results_df = pd.DataFrame(results)
    print(results_df)

    os.makedirs("data/processed", exist_ok=True)
    results_df.to_csv("data/processed/Baseline_Results.csv", index=False)


if __name__ == "__main__":
    main()