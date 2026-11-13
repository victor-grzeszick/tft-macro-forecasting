"""Step 2: build the estimation panel (1990-01 to 2024-12) and check the stationarity of the VAR variables.

Reads data/processed/fred_md_transformed.parquet.
Writes data/processed/panel.parquet, results/adf_tests.csv and figures/targets.png.
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from statsmodels.tsa.stattools import adfuller

import config

NBER_RECESSIONS = [("1990-07", "1991-03"), ("2001-03", "2001-11"), ("2007-12", "2009-06"), ("2020-02", "2020-04")]


def build_panel(transformed):
    """Cut the sample to 1990-2024, drop sparse series and fill the remaining gaps.

    The missing-value filter looks at the training period only, so the choice of predictors does not
    depend on validation or test data. Remaining gaps are forward-filled, which uses past values only.
    Returns (panel, list of dropped series).
    """
    panel = transformed.loc[config.SAMPLE_START:config.SAMPLE_END]
    train = split(panel, "train")
    dropped = train.columns[train.isna().mean() > config.MAX_MISSING_SHARE].tolist()
    panel = panel.drop(columns=dropped).ffill()
    return panel, dropped


def split(panel, period):
    """Return the rows of `panel` that belong to 'train', 'val' or 'test'."""
    start, end = config.PERIODS[period]
    return panel.loc[start:end]


def load_panel(until="test"):
    """Read the panel written by this script, truncated after the period `until` ('val' or 'test').

    Scripts run with until='val' never load a single test-period observation.
    """
    return pd.read_parquet(config.PANEL_FILE).loc[: config.PERIODS[until][1]]


def adf_table(train, names):
    """Augmented Dickey-Fuller test (constant, AIC lag choice) for each named series on the training sample."""
    rows = []
    for name in names:
        stat, pvalue, lags, nobs, critical, _ = adfuller(train[name].dropna())
        rows.append({"series": name, "adf_stat": stat, "p_value": pvalue, "lags": lags, "n_obs": nobs,
                     "crit_1pct": critical["1%"], "crit_5pct": critical["5%"]})
    return pd.DataFrame(rows)


def plot_targets(panel):
    """Plot the three transformed targets with NBER recessions shaded and the sample splits marked."""
    fig, axes = plt.subplots(len(config.TARGETS), 1, figsize=(9, 7), sharex=True)
    for ax, name in zip(axes, config.TARGETS):
        ax.plot(panel.index, panel[name], lw=0.8, color="black")
        for start, end in NBER_RECESSIONS:
            ax.axvspan(pd.Timestamp(start), pd.Timestamp(end), color="grey", alpha=0.3, lw=0)
        for period in ("val", "test"):
            ax.axvline(config.period_bounds(period)[0], color="tab:red", ls="--", lw=0.8)
        ax.set_ylabel(name)
    axes[0].set_title("Transformed targets, 1990-2024 (grey: NBER recessions; dashed: start of validation and test)")
    fig.tight_layout()
    config.FIGURES_DIR.mkdir(exist_ok=True)
    fig.savefig(config.FIGURES_DIR / "targets.png", dpi=150)
    plt.close(fig)


def main():
    panel, dropped = build_panel(pd.read_parquet(config.TRANSFORMED_FILE))
    assert not panel.isna().any().any(), "panel still contains missing values"
    panel.to_parquet(config.PANEL_FILE)
    print(f"panel: {panel.shape[0]} months x {panel.shape[1]} series; dropped {dropped}")
    for period in config.PERIODS:
        print(f"  {period}: {len(split(panel, period))} months")

    config.RESULTS_DIR.mkdir(exist_ok=True)
    adf = adf_table(split(panel, "train"), config.VAR_SYSTEM)
    adf.to_csv(config.RESULTS_DIR / "adf_tests.csv", index=False, float_format="%.4f")
    print(adf.to_string(index=False))
    plot_targets(panel)


if __name__ == "__main__":
    main()
