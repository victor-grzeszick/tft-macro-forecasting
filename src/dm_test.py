import pandas as pd
import numpy as np
import os
from evaluation_utils import align

def evalute(actual, compare1, compare2):
    e_1 = actual - compare1
    e_2   = actual - compare2
    d = e_1**2 - e_2**2

    return d

def gamma(d, k):
    d_bar = d.mean()
    return ((d - d_bar) * (d.shift(k) - d_bar)).dropna().sum() / len(d)


def main():
    predictions_df = pd.read_pickle("data/processed/predictions_unalligned.pkl")
    df_val = pd.read_parquet("data/processed/Y_val.parquet")

    targets = ["CPIAUCSL", "UNRATE", "INDPRO"]
    horizons = [3] ####################################################################################h=12 dropped here: 13-month validation window too short for lag-11 autocorrelation correction; run on test set later
    
    results = []


    for name in targets:
        for h in horizons:
            actual, rw, arima, varr = align(name, h, df_val, predictions_df)

            tests1 = [arima, arima, varr]
            tests2 = [varr, rw, rw]
            pair_a = ["ARIMA", "ARIMA", "VAR"]
            pair_b = ["VAR", "RW", "RW"]

            for i in range(len(tests1)):

                d = evalute(actual, tests1[i], tests2[i])

                d_bar = d.mean()
                gamma0 = gamma(d, 0)
                corrected_variance = gamma0 + 2 * sum(gamma(d, k) for k in range(1, h))   
                dm = d_bar / np.sqrt(corrected_variance / len(d))

                results.append({"target": name, "horizon": h, "DM": dm, "Compare1": pair_a[i], "Compare2": pair_b[i]})
    
    results_df = pd.DataFrame(results)

    os.makedirs("data/processed", exist_ok=True)
    results_df.to_parquet("data/processed/dm_test_results.parquet")


if __name__ == "__main__":
    main()