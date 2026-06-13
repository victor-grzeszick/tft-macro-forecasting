import pandas as pd
import numpy as np
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer, QuantileLoss
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import EarlyStopping
import os
import matplotlib.pyplot as plt
from build_data_sets import build_one_dataset

def main():
    seed_everything(42)
    df_train = pd.read_parquet("data/processed/X_train.parquet")
    df_val = pd.read_parquet("data/processed/X_val.parquet")

    case_switch = 0 ##0 = import training, 1 = new training

    target = "CPIAUCSL"
    horizon = 3

    train_ds, val_ds = build_one_dataset(df_train, df_val, horizon, target)

    train_dl = train_ds.to_dataloader(train=True, batch_size=64)
    val_dl = val_ds.to_dataloader(train=False, batch_size=64)

    if(case_switch == 1):
        tft = TemporalFusionTransformer.from_dataset(train_ds, hidden_size=16, learning_rate=0.03, loss=QuantileLoss())
        early_stop = EarlyStopping(monitor="val_loss", patience=15)
        trainer = Trainer(max_epochs=500, callbacks=[early_stop])
        trainer.fit(model=tft, train_dataloaders=train_dl, val_dataloaders=val_dl)
        trainer.save_checkpoint("models/tft_CPIAUCSL_h3.ckpt")

    elif(case_switch == 0):
        tft = TemporalFusionTransformer.load_from_checkpoint("models/tft_CPIAUCSL_h3.ckpt")
    
    prediction = tft.predict(val_dl, mode="raw", return_x=True)
    fig = tft.plot_prediction(prediction.x, prediction.output, idx=0)
    os.makedirs("figures", exist_ok=True)
    fig.savefig(f"figures/tft_CPIAUCSL_h3.png")

if __name__ == "__main__":
    main()