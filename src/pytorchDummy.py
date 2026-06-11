import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer, QuantileLoss
from lightning.pytorch import Trainer, seed_everything
import os

def main():
    seed_everything(42)
    frame = []
    case_switch = 0 ##0 = import training, 1 = new training

    for i in range(200):
        frame.append({"time_idx": i, "series": "Test", "value": (np.sin(i * 2*np.pi / 25) + np.random.normal(0, 0.1))})

    df_frame = pd.DataFrame(frame)
    
    df_train = df_frame[df_frame.time_idx < 170]

    train_ds = TimeSeriesDataSet(df_train, time_idx="time_idx", target="value", group_ids=["series"], max_encoder_length=24, max_prediction_length=3, time_varying_known_reals=["time_idx"], time_varying_unknown_reals=["value"])
    val_ds = TimeSeriesDataSet.from_dataset(train_ds, df_frame, stop_randomization=True, min_prediction_idx=170)

    train_dl = train_ds.to_dataloader(train=True, batch_size=64)
    val_dl = val_ds.to_dataloader(train=False, batch_size=64)

    if(case_switch == 1):
        tft = TemporalFusionTransformer.from_dataset(train_ds, hidden_size=16, learning_rate=0.01, loss=QuantileLoss())

        trainer = Trainer(max_epochs=300)

        trainer.fit(model=tft, train_dataloaders=train_dl, val_dataloaders=val_dl)

    elif(case_switch == 0):
        tft = TemporalFusionTransformer.load_from_checkpoint("lightning_logs/version_7/checkpoints/epoch=299-step=600.ckpt")

    prediction = tft.predict(val_dl, mode="raw", return_x=True)

    plotting_index = [0, 10, 20]
    os.makedirs("figures", exist_ok=True)

    for index in plotting_index:
        fig = tft.plot_prediction(prediction.x, prediction.output, idx=index)
        fig.savefig(f"figures/toy_idx{index}.png")

if __name__ == "__main__":
    main()