from statsmodels.tsa.stattools import adfuller
import pandas as pd


def run_adf(series, name):
    result = adfuller(series.dropna())
    print(f"{name}:")
    print(f"  ADF statistic: {result[0]:.4f}")
    print(f"  p-value:       {result[1]:.6f}")
    print(f"  critical 5%:   {result[4]['5%']:.4f}")
    print()

def main():
    df = pd.read_parquet("data/processed/fred_md_transformed.parquet")

    run_adf(df["CPIAUCSL"], "CPIAUCSL")
    run_adf(df["INDPRO"], "INDPRO")

if __name__ == "__main__":
    main()