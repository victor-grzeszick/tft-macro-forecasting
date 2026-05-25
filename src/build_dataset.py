import pandas as pd
import numpy as np
import os

def code_1(pRow):
    return pRow

def code_2(pRow):
    return pRow.diff()

def code_4(pRow):
    return np.log(pRow)

def code_5(pRow):
    return np.log(pRow).diff()

def code_6(pRow):
    return np.log(pRow).diff().diff()

def code_7(pRow):
    return (pRow / pRow.shift(1) - 1).diff()

def main(pTransformations):

    df = pd.read_csv("data/raw/fred_md.csv")

    codes = df.iloc[0, 1:].astype(int)

    codes["CPIAUCSL"] = 5

    clean = df.iloc[1:, :]

    clean["sasdate"] = pd.to_datetime(clean['sasdate'])
    clean = clean.set_index('sasdate')
    clean = clean.apply(pd.to_numeric, errors="coerce")


    out = pd.DataFrame(index=clean.index)

    transformed_columns = {
        name: pTransformations[codes[name]](clean[name])
        for name in clean.columns
    }

    out = pd.DataFrame(transformed_columns)


    os.makedirs("data/processed", exist_ok=True)
    out.to_parquet("data/processed/fred_md_transformed.parquet")

transformations = {1: code_1, 2: code_2, 4: code_4, 5: code_5, 6: code_6, 7: code_7}

if __name__ == "__main__":
    main(transformations)
