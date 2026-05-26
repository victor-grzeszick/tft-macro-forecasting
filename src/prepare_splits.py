import pandas as pd

def extract_targets(split):
    return split["CPIAUCSL"], split["UNRATE"], split["INDPRO"]


def main():
    df = pd.read_parquet("data/processed/fred_md_transformed.parquet")
    df = df.loc["1990-01":"2024-12", :]
    
    mask = df.isna().mean() > 0.05
    df = df.drop(columns=mask[mask].index)

    print(df.shape)
    
    print(mask[mask].index.tolist())

    train = df.loc["1990-01":"2017-12", :]
    print(train.shape)

    val = df.loc["2018-01":"2019-12", :]
    print(val.shape)

    test = df.loc["2020-01":"2024-12", :]
    print(test.shape)

    y_cpi_train, y_unrate_train, y_indpro_train = extract_targets(train)

    y_cpi_val, y_unrate_val, y_indpro_val = extract_targets(val)

    y_cpi_test, y_unrate_test, y_indpro_test = extract_targets(test)
    

if __name__ == "__main__":
    main()