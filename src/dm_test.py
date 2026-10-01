"""Diebold-Mariano statistic of the best validation baseline against the TFT at h=3.

Writes data/processed/dm_test_results.parquet."""
import pandas as pd
import numpy as np
import os
from evaluation_utils import align

def evalute(actual, compare1, compare2):
    """Return the loss differential d = (actual - compare1)**2 - (actual - compare2)**2."""
    e_1 = actual - compare1
    e_2   = actual - compare2
    d = e_1**2 - e_2**2

    return d

def gamma(d, k):
    """Return the lag-k autocovariance of d: sum of (d_t - mean) * (d_{t-k} - mean) over available pairs, divided by len(d)."""
    d_bar = d.mean()
    return ((d - d_bar) * (d.shift(k) - d_bar)).dropna().sum() / len(d)


def main():
    """For each target at h=3, take the baseline with the lowest RMSE in Baseline_Results.csv, compute the DM
    statistic against the TFT with autocovariances up to lag h-1, and write and print the results."""
    preds = pd.read_pickle("data/processed/predictions_unalligned.pkl")
    tft = pd.read_pickle("data/processed/predictions_unalligned_tft.pkl")
    preds = pd.concat([preds, tft], ignore_index=True)

    scores = pd.read_csv("data/processed/Baseline_Results.csv")

    df_val = pd.read_parquet("data/processed/Y_val.parquet")


    targets = ["CPIAUCSL", "UNRATE", "INDPRO"]
    horizons = [3] ########h=12 dropped here: 13-month validation window too short for lag-11 autocorrelation correction; run on test set later
    
    results = []


    for h in horizons:
        for name in targets:

            actual, rw, arima, tft, varr = align(name, h, df_val, preds)

            row = scores[(scores.target == name) & (scores.horizon == h) & (scores.model != "TFT")]
            by_model = {"RW": rw, "ARIMA": arima, "VAR": varr}
            best_name = row.loc[row.RMSE.idxmin(), "model"]
            best = by_model[best_name]


            d = evalute(actual, best, tft)

            d_bar = d.mean()
            gamma0 = gamma(d, 0)
            corrected_variance = gamma0 + 2 * sum(gamma(d, k) for k in range(1, h)) 
            dm = d_bar / np.sqrt(corrected_variance / len(d))

            results.append({"target": name, "horizon": h, "DM": dm, "Compare1": best_name, "Compare2": "TFT"})
    
    results_df = pd.DataFrame(results)

    os.makedirs("data/processed", exist_ok=True)
    results_df.to_parquet("data/processed/dm_test_results.parquet")
    print(results_df)


if __name__ == "__main__":
    main()