"""Plot actual values against the RW, ARIMA and VAR validation forecasts."""
import pandas as pd
import os
import matplotlib.pyplot as plt
from evaluation_utils import align

def main():
    """Save one figure per target and horizon (3, 12) to figures/{target}_h{h}.png."""
    predictions_df = pd.read_pickle("data/processed/predictions_unalligned.pkl")
    df_val = pd.read_parquet("data/processed/Y_val.parquet")
    os.makedirs("figures", exist_ok=True)

    targets = ["CPIAUCSL", "UNRATE", "INDPRO"]
    horizons = [3, 12] 


    for name in targets:
        for h in horizons:
            actual, rw, arima, varr = align(name, h, df_val, predictions_df)

            actual.plot(label="Actual")
            rw.plot(label="RW")
            arima.plot(label="ARIMA")
            varr.plot(label="VAR")

            plt.title(f"{name} h={h}")
            plt.ylabel("transformed values")
            
            plt.legend()
            plt.savefig(f"figures/{name}_h{h}.png")
            plt.close()


if __name__ == "__main__":
    main()