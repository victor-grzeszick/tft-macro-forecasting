"""Build pytorch-forecasting TimeSeriesDataSets (training and validation) per target and horizon."""
import pandas as pd
import numpy as np
from pytorch_forecasting import TimeSeriesDataSet

def prepare(df):
    """Drop the date index and add an integer `time_idx` column and a constant group column `series`."""
    df = df.reset_index(drop=True)   
    df["time_idx"] = np.arange(len(df))
    df["series"] = "macro"
    return df

def build_one_dataset(df_train, df_val, h, name):
    """Return (train_ds, val_ds) for target `name` and horizon h: max encoder length 24, time_idx as known real,
    all other columns as unknown reals. val_ds is built on concat(df_train, df_val) and predicts only
    time steps after the training rows."""
    df_full = pd.concat((df_train, df_val))
    
    df_train = prepare(df_train)
    df_full = prepare(df_full)
    train_ds = TimeSeriesDataSet(df_train, time_idx="time_idx", target=name, group_ids=["series"], max_encoder_length=24, max_prediction_length=h, time_varying_known_reals=["time_idx"], time_varying_unknown_reals=[c for c in df_train.columns if c not in ["time_idx", "series"]])
    val_ds = TimeSeriesDataSet.from_dataset(train_ds, df_full, stop_randomization=True, min_prediction_idx=len(df_train))

    return train_ds, val_ds


def build_data_set(df_train, df_val, horizons, targets):
    """Call build_one_dataset for every target and horizon.
    Returns two lists of {Target, Horizon, DataSet} dicts: training sets and validation sets."""

    train_data_sets = []
    val_data_sets = []

    for name in targets:
        for h in horizons:
            train_ds, val_ds = build_one_dataset(df_train,  df_val, h, name)

            train_data_sets.append({"Target": name, "Horizon": h, "DataSet": train_ds})
            val_data_sets.append({"Target": name, "Horizon": h, "DataSet": val_ds})

    return train_data_sets, val_data_sets

def main():
    """Build the datasets for CPIAUCSL, UNRATE and INDPRO at horizons 3 and 12 from X_train and X_val.
    Writes nothing."""
    df_train = pd.read_parquet("data/processed/X_train.parquet")
    df_val = pd.read_parquet("data/processed/X_val.parquet")

    targets = ["CPIAUCSL", "UNRATE", "INDPRO"]
    horizons = [3, 12]

    train_data_sets, val_data_sets = build_data_set(df_train, df_val, horizons, targets)

if __name__ == "__main__":
    main()