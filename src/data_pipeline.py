"""Keep 1990-01 to 2024-12, drop columns with more than 5% missing values, and write
train (1990-2017), validation (2018-2019) and test (2020-2024) X_/Y_ parquet files to data/processed/."""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def extract_targets(split):
    """Return the CPIAUCSL, UNRATE and INDPRO columns of `split`."""
    return split["CPIAUCSL"], split["UNRATE"], split["INDPRO"]

def make_recession_grey():
    """Shade the hard-coded recession date ranges grey on the current plot."""
    recession_dates = [("1990-07", "1991-03"), ("2001-03", "2001-11"), ("2007-12", "2009-06"), ("2020-02", "2020-04")]
    for data in recession_dates:
        plt.axvspan(data[0], data[1], alpha=0.2, color="grey")


def main():
    """Filter and split the data, show the target plots and correlation heatmap (not saved), and write
    X_{train,val,test} (all kept columns) and Y_{train,val,test} (the three targets) as parquet."""
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


    df["CPIAUCSL"].plot(title='CPIAUCSL', ylabel='', xlabel='')
    make_recession_grey()
    plt.show()
    
    df["UNRATE"].plot(title='UNRATE', ylabel='', xlabel='')
    make_recession_grey()
    plt.show()

    df["INDPRO"].plot(title='INDPRO', ylabel='', xlabel='')
    make_recession_grey()
    plt.show()

    heatmap_dataframe = df.corr()
    plt.figure(figsize=(8, 8))
    heatmap = sns.heatmap(heatmap_dataframe, cmap="coolwarm", center=0)
    plt.show()

    target_train = pd.DataFrame({"CPIAUCSL": y_cpi_train, "UNRATE": y_unrate_train, "INDPRO": y_indpro_train})
    target_val = pd.DataFrame({"CPIAUCSL": y_cpi_val, "UNRATE": y_unrate_val, "INDPRO": y_indpro_val})
    target_test = pd.DataFrame({"CPIAUCSL": y_cpi_test, "UNRATE": y_unrate_test, "INDPRO": y_indpro_test})

    os.makedirs("data/processed", exist_ok=True)
    train.to_parquet("data/processed/X_train.parquet")
    val.to_parquet("data/processed/X_val.parquet")
    test.to_parquet("data/processed/X_test.parquet")

    target_train.to_parquet("data/processed/Y_train.parquet")
    target_val.to_parquet("data/processed/Y_val.parquet")
    target_test.to_parquet("data/processed/Y_test.parquet")
    

if __name__ == "__main__":
    main()