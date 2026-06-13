import pandas as pd
import numpy as np
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer, QuantileLoss
from lightning.pytorch import Trainer, seed_everything
import os

def prepare(df):
    df = df.reset_index(drop=True)   
    df["time_idx"] = np.arange(len(df))
    df["series"] = "macro"
    return df

def build_one_dataset(df_train, df_val, h, name):
    df_full = pd.concat((df_train, df_val))
    
    df_train = prepare(df_train)
    df_full = prepare(df_full)
    train_ds = TimeSeriesDataSet(df_train, time_idx="time_idx", target=name, group_ids=["series"], max_encoder_length=24, max_prediction_length=h, time_varying_known_reals=["time_idx"], time_varying_unknown_reals=[c for c in df_train.columns if c not in ["time_idx", "series"]])
    val_ds = TimeSeriesDataSet.from_dataset(train_ds, df_full, stop_randomization=True, min_prediction_idx=len(df_train))

    return train_ds, val_ds


def build_data_set(df_train, df_val, horizons, targets):

    train_data_sets = []
    val_data_sets = []

    for name in targets:
        for h in horizons:
            train_ds, val_ds = build_one_dataset(df_train,  df_val, h, name)

            train_data_sets.append({"Target": name, "Horizon": h, "DataSet": train_ds})
            val_data_sets.append({"Target": name, "Horizon": h, "DataSet": val_ds})

    return train_data_sets, val_data_sets

def main():
    df_train = pd.read_parquet("data/processed/X_train.parquet")
    df_val = pd.read_parquet("data/processed/X_val.parquet")

    targets = ["CPIAUCSL", "UNRATE", "INDPRO"]
    horizons = [3, 12]

    train_data_sets, val_data_sets = build_data_set(df_train, df_val, horizons, targets)

if __name__ == "__main__":
    main()