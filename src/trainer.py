import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer, QuantileLoss
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import EarlyStopping
from build_data_sets import build_one_dataset
from evaluation_utils import evaluate


def load_or_train(train_ds, train_dl, val_dl, target, horizon, learning_rate, case_switch):
    ckpt_path = f"models/tft_{target}_h{horizon}.ckpt"
    if case_switch == 1:
        tft = TemporalFusionTransformer.from_dataset(train_ds, hidden_size=16, learning_rate=learning_rate, loss=QuantileLoss())
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

def tune_architecture(train_ds, train_dl, val_dl):

    learning_rate = [0.01, 0.02, 0.0234, 0.03, 0.01]
    hidden_size = [8, 12, 16, 32]
    dropout = [0.1, 0.2, 0.3]

    best_loss = 99
    result = ()

    for size in hidden_size:
        for drop in dropout:
            for rate in learning_rate:
                tft = TemporalFusionTransformer.from_dataset(train_ds, hidden_size=size, learning_rate=rate, loss=QuantileLoss(), dropout=drop)
                early_stop = EarlyStopping(monitor="val_loss", patience=15)
                trainer = Trainer(max_epochs=50, callbacks=[early_stop])
                trainer.fit(model=tft, train_dataloaders=train_dl, val_dataloaders=val_dl)
                loss = early_stop.best_score.item()
                if(loss < best_loss):
                    best_loss = loss
                    result = (size, drop, rate)

    return result



def main():
    seed_everything(42)
    df_train = pd.read_parquet("data/processed/X_train.parquet")
    df_val = pd.read_parquet("data/processed/X_val.parquet")

    case_switch = 1  ## 0 = load, 1 = train

    target = "UNRATE"
    horizon = 3

    train_ds, val_ds = build_one_dataset(df_train, df_val, horizon, target)
    train_dl = train_ds.to_dataloader(train=True, batch_size=64)
    val_dl = val_ds.to_dataloader(train=False, batch_size=64)

    hidden_size, dropout_rate, learning_rate = tune_architecture(train_ds, train_dl, val_dl)
    print("###########")
    print(hidden_size, dropout_rate, learning_rate)
    print("###########")

    tft = load_or_train(train_ds, train_dl, val_dl, target, horizon, learning_rate, case_switch)

    save_plot(tft, val_dl, target, horizon)

    mae, rmse = eval_vs_baseline(tft, val_dl, horizon)
    print("MAE:", mae, "RMSE:", rmse)

    top10 = top_k_importance(tft, val_dl, 10)
    for v in top10:
        print(v)


if __name__ == "__main__":
    main()