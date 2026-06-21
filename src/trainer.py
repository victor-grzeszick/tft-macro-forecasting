import pandas as pd
import numpy as np
import os
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer, QuantileLoss
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import EarlyStopping
import matplotlib.pyplot as plt
from build_data_sets import build_one_dataset
from evaluation_utils import evaluate


def load_or_train(train_ds, train_dl, val_dl, target, horizon, case_switch):
    ckpt_path = f"models/tft_{target}_h{horizon}.ckpt"
    if case_switch == 1:
        tft = TemporalFusionTransformer.from_dataset(train_ds, hidden_size=16, learning_rate=0.03, loss=QuantileLoss())
        early_stop = EarlyStopping(monitor="val_loss", patience=15)
        trainer = Trainer(max_epochs=500, callbacks=[early_stop])
        trainer.fit(model=tft, train_dataloaders=train_dl, val_dataloaders=val_dl)
        os.makedirs("models", exist_ok=True)
        trainer.save_checkpoint(ckpt_path)
    else:
        tft = TemporalFusionTransformer.load_from_checkpoint(ckpt_path)
    return tft


def eval_vs_baseline(tft, val_dl, horizon):
    predict = tft.predict(val_dl, mode="prediction", return_y=True)
    output = predict.output[:, horizon - 1]
    y = predict.y[0][:, horizon - 1].cpu()
    mae, rmse = evaluate(output, y)
    return mae.item(), rmse.item()


def top_k_importance(tft, val_dl, k):
    prediction = tft.predict(val_dl, mode="raw", return_x=True)
    interpretation = tft.interpret_output(prediction.output, reduction="sum")
    scores = interpretation["encoder_variables"]
    names = tft.encoder_variables
    variables = []
    for n, s in zip(names, scores):
        variables.append({"Name": n, "Value": s.item()})
    variables.sort(key=lambda d: d["Value"], reverse=True)
    return variables[:k]

def save_plot(tft, val_dl, target, horizon):
    prediction = tft.predict(val_dl, mode="raw", return_x=True)
    fig = tft.plot_prediction(prediction.x, prediction.output, idx=0)
    os.makedirs("figures", exist_ok=True)
    fig.savefig(f"figures/tft_{target}_h{horizon}.png")


def main():
    seed_everything(42)
    df_train = pd.read_parquet("data/processed/X_train.parquet")
    df_val = pd.read_parquet("data/processed/X_val.parquet")

    case_switch = 0  ## 0 = load, 1 = train

    target = "CPIAUCSL"
    horizon = 3

    train_ds, val_ds = build_one_dataset(df_train, df_val, horizon, target)
    train_dl = train_ds.to_dataloader(train=True, batch_size=64)
    val_dl = val_ds.to_dataloader(train=False, batch_size=64)

    tft = load_or_train(train_ds, train_dl, val_dl, target, horizon, case_switch)

    save_plot(tft, val_dl, target, horizon)

    mae, rmse = eval_vs_baseline(tft, val_dl, horizon)
    print("MAE:", mae, "RMSE:", rmse)

    top10 = top_k_importance(tft, val_dl, 10)
    for v in top10:
        print(v)


if __name__ == "__main__":
    main()