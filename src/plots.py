"""Step 7: figures.

Reads the files in results/ and data/processed/panel.parquet.
Writes figures/forecasts_{period}.png, figures/relative_rmse.png, figures/variable_importance.png
and figures/attention.png.
"""
import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

import config
from data_pipeline import load_panel
from evaluate import load_forecasts

COLOURS = {"RW": "tab:grey", "Mean": "tab:olive", "ARIMA": "tab:blue", "VAR": "tab:green", "TFT": "tab:red"}


def grid():
    """One panel per target (rows) and horizon (columns)."""
    fig, axes = plt.subplots(len(config.TARGETS), len(config.HORIZONS), figsize=(12, 9))
    cells = [(axes[i, j], target, horizon) for i, target in enumerate(config.TARGETS)
             for j, horizon in enumerate(config.HORIZONS)]
    return fig, cells


def save(fig, name):
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / name, dpi=150)
    plt.close(fig)


def plot_forecasts(forecasts, period):
    """Realised values against the ARIMA, VAR and TFT forecasts, with the TFT's 10-90% interval."""
    fig, cells = grid()
    for ax, target, horizon in cells:
        g = forecasts[(forecasts["period"] == period) & (forecasts["target"] == target) & (forecasts["horizon"] == horizon)]
        wide = g.pivot(index="date", columns="model", values="forecast")
        tft = g[g["model"] == "TFT"].set_index("date")
        ax.plot(tft.index, tft["actual"], color="black", lw=1.2, label="Actual")
        ax.fill_between(tft.index, tft["q10"], tft["q90"], color=COLOURS["TFT"], alpha=0.15, lw=0, label="TFT 10-90%")
        for model in ["ARIMA", "VAR", "TFT"]:
            ax.plot(wide.index, wide[model], color=COLOURS[model], lw=1, label=model)
        ax.set_title(f"{target}, h = {horizon}")
        ax.tick_params(axis="x", labelrotation=30)
    cells[0][0].legend(fontsize=8, ncol=2)
    fig.suptitle(f"Forecasts and realised values, {period} period (transformed units)")
    save(fig, f"forecasts_{period}.png")


def plot_relative_rmse(metrics, period):
    """RMSE of each model divided by the RMSE of the random walk."""
    models = ["Mean", "ARIMA", "VAR", "TFT"]
    part = metrics[(metrics["period"] == period) & metrics["model"].isin(models)]
    wide = part.assign(cell=part["target"] + "\nh=" + part["horizon"].astype(str)).pivot(index="cell", columns="model", values="rel_rmse")
    order = [f"{t}\nh={h}" for t in config.TARGETS for h in config.HORIZONS]
    ax = wide.loc[order, models].plot.bar(figsize=(9, 4.5), color=[COLOURS[m] for m in models], width=0.8, rot=0)
    ax.axhline(1, color="black", lw=0.8, ls="--")
    ax.set_xlabel("")
    ax.set_ylabel("RMSE relative to random walk")
    ax.set_title(f"Relative RMSE, {period} period (below 1: more accurate than the random walk)")
    save(ax.figure, "relative_rmse.png")


def plot_importance(importance):
    """The ten largest encoder variable-selection weights of each model."""
    fig, cells = grid()
    for ax, target, horizon in cells:
        g = importance[(importance["target"] == target) & (importance["horizon"] == horizon)].iloc[::-1]
        ax.barh(g["variable"], g["weight"], color="tab:red")
        ax.set_xlim(0, 1)
        ax.set_title(f"{target}, h = {horizon}")
        ax.tick_params(axis="y", labelsize=7)
    fig.suptitle("TFT variable-selection weights (share of total, top 10 of 126 encoder inputs)")
    save(fig, "variable_importance.png")


def plot_attention(attention):
    """Average attention weight on each month of the encoder window."""
    fig, cells = grid()
    for ax, target, horizon in cells:
        g = attention[(attention["target"] == target) & (attention["horizon"] == horizon)]
        ax.bar(g["lag"], g["attention"], color="tab:red")
        ax.set_title(f"{target}, h = {horizon}")
        ax.set_xlabel("months relative to forecast origin")
    fig.suptitle("TFT attention over the 24-month encoder window (average over forecasts)")
    save(fig, "attention.png")


def main(until):
    config.FIGURES_DIR.mkdir(exist_ok=True)
    forecasts = load_forecasts(load_panel(until))
    metrics = pd.read_csv(config.RESULTS_DIR / "metrics.csv")
    for period in ["val", "test"][: 1 if until == "val" else 2]:
        plot_forecasts(forecasts, period)
    plot_relative_rmse(metrics, until)
    plot_importance(pd.read_csv(config.RESULTS_DIR / "variable_importance.csv"))
    plot_attention(pd.read_csv(config.RESULTS_DIR / "attention.csv"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--until", choices=["val", "test"], default="test", help="last period to plot")
    main(parser.parse_args().until)
