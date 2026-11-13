"""Step 4: tune and train the Temporal Fusion Transformers. Uses training and validation data only.

    python src/tft_train.py --tune    grid search on UNRATE h=3 -> results/tuning_grid.csv, results/tft_config.json
    python src/tft_train.py           one model per target and horizon -> models/*.ckpt, results/training_log.csv

Each run trains on 1990-2017, stops early on the 2018-2019 validation loss, and keeps the weights of the
epoch with the lowest validation loss.
"""
import argparse
import itertools
import json
import logging
import shutil
import tempfile
import warnings

import pandas as pd
from lightning.pytorch import Callback, Trainer, seed_everything
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_forecasting import QuantileLoss, TemporalFusionTransformer

import config
from data_pipeline import load_panel, split
from tft_data import evaluation_dataset, training_dataset

CONFIG_FILE = config.RESULTS_DIR / "tft_config.json"


class ValidationHistory(Callback):
    """Record the validation loss after every epoch."""

    def __init__(self):
        self.losses = []

    def on_validation_epoch_end(self, trainer, pl_module):
        self.losses.append(trainer.callback_metrics["val_loss"].item())


def checkpoint_path(target, horizon, seed=config.SEED):
    """Where the trained model for one target, horizon and seed is stored."""
    suffix = "" if seed == config.SEED else f"_seed{seed}"
    return config.MODELS_DIR / f"tft_{target}_h{horizon}{suffix}.ckpt"


def fit(panel, target, horizon, hidden_size, dropout, learning_rate, seed=config.SEED, save_to=None):
    """Train one TFT and return a summary of the run. `panel` must end with the validation period."""
    seed_everything(seed, workers=True)
    train_panel = split(panel, "train")
    train_ds = training_dataset(train_panel, target, horizon)
    val_ds = evaluation_dataset(train_ds, panel, first_origin=len(train_panel) - 1)

    model = TemporalFusionTransformer.from_dataset(
        train_ds, hidden_size=hidden_size, attention_head_size=config.ATTENTION_HEADS, dropout=dropout,
        learning_rate=learning_rate, loss=QuantileLoss(config.QUANTILES))
    history = ValidationHistory()
    with tempfile.TemporaryDirectory() as tmp:
        checkpoint = ModelCheckpoint(dirpath=tmp, monitor="val_loss", mode="min", save_top_k=1)
        trainer = Trainer(
            max_epochs=config.MAX_EPOCHS, accelerator="cpu", logger=False, num_sanity_val_steps=0,
            enable_progress_bar=False, enable_model_summary=False,
            callbacks=[EarlyStopping(monitor="val_loss", patience=config.PATIENCE), checkpoint, history])
        trainer.fit(model,
                    train_dataloaders=train_ds.to_dataloader(train=True, batch_size=config.BATCH_SIZE),
                    val_dataloaders=val_ds.to_dataloader(train=False, batch_size=config.BATCH_SIZE))
        if save_to is not None:
            save_to.parent.mkdir(exist_ok=True)
            shutil.copy(checkpoint.best_model_path, save_to)
    return {"target": target, "horizon": horizon, "seed": seed, "hidden_size": hidden_size, "dropout": dropout,
            "learning_rate": learning_rate, "parameters": sum(p.numel() for p in model.parameters()),
            "train_samples": len(train_ds), "val_samples": len(val_ds), "epochs_run": len(history.losses),
            "best_epoch": history.losses.index(min(history.losses)) + 1, "val_loss": min(history.losses)}


def tune(panel):
    """Grid search on one target/horizon; the configuration with the lowest validation loss is kept."""
    names = list(config.GRID)
    runs = [fit(panel, config.TUNING_TARGET, config.TUNING_HORIZON, **dict(zip(names, values)))
            for values in itertools.product(*config.GRID.values())]
    grid = pd.DataFrame(runs).sort_values("val_loss").reset_index(drop=True)
    grid.to_csv(config.RESULTS_DIR / "tuning_grid.csv", index=False, float_format="%.6g")
    best = {name: grid.loc[0, name].item() for name in names}
    CONFIG_FILE.write_text(json.dumps(best, indent=2) + "\n")
    print(grid.to_string(index=False))
    print("selected:", best)


def train_all(panel, seeds):
    """Train one model per target, horizon and seed with the tuned configuration."""
    best = json.loads(CONFIG_FILE.read_text())
    log_file = config.RESULTS_DIR / "training_log.csv"
    runs = []
    for seed, target, horizon in itertools.product(seeds, config.TARGETS, config.HORIZONS):
        runs.append(fit(panel, target, horizon, seed=seed, save_to=checkpoint_path(target, horizon, seed), **best))
        print(runs[-1])
    log = pd.DataFrame(runs)
    if log_file.exists():  # keep the runs of other seeds
        old = pd.read_csv(log_file)
        log = pd.concat([old[~old["seed"].isin(seeds)], log]).sort_values(["seed", "target", "horizon"])
    log.to_csv(log_file, index=False, float_format="%.6g")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tune", action="store_true", help="run the hyperparameter grid instead of training")
    parser.add_argument("--seeds", type=int, nargs="+", default=[config.SEED], help="random seeds to train")
    args = parser.parse_args()

    warnings.filterwarnings("ignore")
    logging.getLogger("lightning.pytorch").setLevel(logging.ERROR)
    logging.getLogger("lightning.fabric").setLevel(logging.ERROR)
    config.RESULTS_DIR.mkdir(exist_ok=True)
    data = load_panel("val")
    tune(data) if args.tune else train_all(data, args.seeds)
