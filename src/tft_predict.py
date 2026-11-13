"""Step 5: forecasts and interpretation output of the trained TFTs. Model weights stay frozen.

Reads models/*.ckpt and data/processed/panel.parquet.
Writes results/forecasts_tft.csv, results/variable_importance.csv and results/attention.csv.
"""
import argparse
import logging
import warnings

import pandas as pd
from pytorch_forecasting import TemporalFusionTransformer

import config
from baselines import first_origin
from data_pipeline import load_panel, split
from tft_data import evaluation_dataset, training_dataset
from tft_train import checkpoint_path

def trainer_options():
    """Fresh options for every predict call (the library adds callbacks to the dictionary it is given)."""
    return {"accelerator": "cpu", "logger": False, "enable_progress_bar": False}


def forecast(model, eval_ds, index, target, horizon, label):
    """Quantile forecasts for the last decoder step (t+h) of every sample in `eval_ds`."""
    loader = eval_ds.to_dataloader(train=False, batch_size=256)
    prediction = model.predict(loader, mode="quantiles", return_index=True, trainer_kwargs=trainer_options())
    quantiles = prediction.output[:, horizon - 1, :].cpu().numpy()
    first_step = prediction.index["time_idx"].to_numpy()  # position of month t+1 of each sample
    out = pd.DataFrame({"target": target, "horizon": horizon, "model": label,
                        "origin": index[first_step - 1], "date": index[first_step + horizon - 1]})
    for column, values in zip(["q10", "forecast", "q90"], quantiles.T):
        out[column] = values
    return out


def interpret(model, eval_ds):
    """Average variable-selection weights (encoder) and attention weights over all samples in `eval_ds`."""
    loader = eval_ds.to_dataloader(train=False, batch_size=256)
    raw = model.predict(loader, mode="raw", trainer_kwargs=trainer_options())
    summary = model.interpret_output(raw, reduction="mean")
    importance = pd.Series(summary["encoder_variables"].cpu().numpy(), index=model.encoder_variables)
    attention = pd.Series(summary["attention"].cpu().numpy()[: config.ENCODER_LENGTH],
                          index=range(-config.ENCODER_LENGTH + 1, 1))
    return importance / importance.sum(), attention


def main(until, seeds):
    panel = load_panel(until)
    train_panel = split(panel, "train")
    forecasts, importances, attentions = [], [], []
    for seed in seeds:
        label = "TFT" if seed == config.SEED else f"TFT_seed{seed}"
        for target in config.TARGETS:
            for horizon in config.HORIZONS:
                model = TemporalFusionTransformer.load_from_checkpoint(checkpoint_path(target, horizon, seed))
                train_ds = training_dataset(train_panel, target, horizon)
                eval_ds = evaluation_dataset(train_ds, panel, first_origin(panel.index))
                forecasts.append(forecast(model, eval_ds, panel.index, target, horizon, label))
                importance, attention = interpret(model, eval_ds)
                top = importance.sort_values(ascending=False).head(10)
                importances.append(pd.DataFrame({"target": target, "horizon": horizon, "seed": seed, "rank": range(1, 11),
                                                 "variable": top.index, "weight": top.to_numpy()}))
                attentions.append(pd.DataFrame({"target": target, "horizon": horizon, "seed": seed,
                                                "lag": attention.index, "attention": attention.to_numpy()}))
    forecasts = pd.concat(forecasts).assign(seed=lambda f: f["model"].str.extract(r"seed(\d+)")[0].fillna(config.SEED).astype(int))
    outputs = {"forecasts_tft.csv": (forecasts, "%.8g"), "variable_importance.csv": (pd.concat(importances), "%.6f"),
               "attention.csv": (pd.concat(attentions), "%.6f")}
    for name, (table, number_format) in outputs.items():
        path = config.RESULTS_DIR / name
        if path.exists():  # keep the stored results of seeds that were not recomputed in this run
            old = pd.read_csv(path, parse_dates=[c for c in ("origin", "date") if c in table.columns])
            table = pd.concat([old[~old["seed"].isin(seeds)], table], ignore_index=True)
        table = table.sort_values(["seed", "target", "horizon"], kind="stable")
        table.to_csv(path, index=False, date_format="%Y-%m-%d", float_format=number_format)
    print(f"{len(forecasts)} TFT forecasts written for seeds {seeds}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--until", choices=["val", "test"], default="test", help="last period to forecast")
    parser.add_argument("--seeds", type=int, nargs="+", default=[config.SEED], help="seeds of the models to use")
    args = parser.parse_args()
    warnings.filterwarnings("ignore")
    logging.getLogger("lightning.pytorch").setLevel(logging.ERROR)
    main(args.until, args.seeds)
