import pandas as pd
import os
from evaluation_utils import evaluate, align

def main():
    df_val = pd.read_parquet("data/processed/Y_val.parquet")
    predictions_df = pd.read_pickle("data/processed/predictions_unalligned.pkl")

    targets = ["CPIAUCSL", "UNRATE", "INDPRO"]
    horizons = [3, 12]

    clean_results = []

    for name in targets:
        for h in horizons:
            actual, rw, arima, varr = align(name, h, df_val, predictions_df)

            error = evaluate(rw, actual)
            mae, rmse = error
            clean_results.append({"target": name, "horizon": h, "MAE": mae, "RMSE": rmse, "model": "RW"})

            error = evaluate(arima, actual)
            mae, rmse = error
            clean_results.append({"target": name, "horizon": h, "MAE": mae, "RMSE": rmse, "model": "ARIMA"})

            error = evaluate(varr, actual)
            mae, rmse = error
            clean_results.append({"target": name, "horizon": h, "MAE": mae, "RMSE": rmse, "model": "VAR"})

    clean_results = pd.DataFrame(clean_results)

    os.makedirs("data/processed", exist_ok=True)
    clean_results.to_csv("data/processed/Baseline_Results.csv", index=False)

if __name__ == "__main__":
    main()