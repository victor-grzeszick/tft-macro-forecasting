"""Print Augmented Dickey-Fuller test results for CPIAUCSL, INDPRO, UNRATE, FEDFUNDS and T10YFFM."""
from statsmodels.tsa.stattools import adfuller
import pandas as pd


def run_adf(series, name):
    """Run adfuller on `series` with NaNs dropped; print the statistic, p-value and 5% critical value."""
    result = adfuller(series.dropna())
    print(f"{name}:")
    print(f"  ADF statistic: {result[0]:.4f}")
    print(f"  p-value:       {result[1]:.6f}")
    print(f"  critical 5%:   {result[4]['5%']:.4f}")
    print()

def main():
    """Run ADF on CPIAUCSL, INDPRO and UNRATE over the whole transformed file,
    and on FEDFUNDS and T10YFFM over 1990-01 to 2017-12."""
    df = pd.read_parquet("data/processed/fred_md_transformed.parquet")

    run_adf(df["CPIAUCSL"], "CPIAUCSL")
    run_adf(df["INDPRO"], "INDPRO")
    run_adf(df["UNRATE"], "UNRATE")


    system_train = df.loc["1990-01":"2017-12", ["CPIAUCSL","UNRATE","INDPRO","FEDFUNDS","T10YFFM"]]
    run_adf(system_train["FEDFUNDS"], "FEDFUNDS")   
    run_adf(system_train["T10YFFM"], "T10YFFM")

if __name__ == "__main__":
    main()