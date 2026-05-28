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
    run_adf(df["UNRATE"], "UNRATE")


    system_train = df.loc["1990-01":"2017-12", ["CPIAUCSL","UNRATE","INDPRO","FEDFUNDS","T10YFFM"]]
    run_adf(system_train["FEDFUNDS"], "FEDFUNDS")   
    run_adf(system_train["T10YFFM"], "T10YFFM")

if __name__ == "__main__":
    main()