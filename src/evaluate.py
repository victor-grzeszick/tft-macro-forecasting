"""Step 6: accuracy metrics and Diebold-Mariano tests for all models.

Reads results/forecasts_baselines.csv, results/forecasts_tft.csv and data/processed/panel.parquet.
Writes results/metrics.csv, results/dm_tests.csv, results/seed_robustness.csv and results/summary.md.
"""
import argparse

import numpy as np
import pandas as pd
from scipy import stats

import config
from data_pipeline import load_panel

CLASSICAL = ["RW", "ARIMA", "VAR"]  # candidates for "strongest classical baseline" in the specification
BENCHMARKS = ["RW", "Mean", "ARIMA", "VAR", "ARIMA_frozen", "VAR_frozen"]
BACKSLASH_STAR = "\\*"  # an asterisk that markdown prints literally
DM_COLUMNS = ["period", "target", "horizon", "model_a", "model_b", "n", "mse_ratio", "dm_stat", "p_value", "kernel"]
SUBPERIODS = {"test_ex2020": ("test", "2021-01")}  # test targets from 2021-01 on (excludes the COVID year)


def rmse(errors):
    return float(np.sqrt(np.mean(np.square(errors))))


def mae(errors):
    return float(np.mean(np.abs(errors)))


def autocovariance(d, k):
    """Sample autocovariance of d at lag k (divisor n)."""
    centred = np.asarray(d, dtype=float) - np.mean(d)
    return float(np.sum(centred[k:] * centred[: len(centred) - k]) / len(centred))


def diebold_mariano(errors_a, errors_b, horizon):
    """Diebold-Mariano (1995) test of equal squared-error accuracy with the Harvey-Leybourne-Newbold (1997) correction.

    d_t = e_a,t^2 - e_b,t^2. Under the null E[d_t] = 0. Errors of h-step forecasts are serially correlated up
    to lag h-1, so the long-run variance of d sums the autocovariances up to that lag (rectangular kernel).
    If that estimate is not positive, Bartlett (Newey-West) weights are used instead. The statistic is
    multiplied by the HLN small-sample factor and compared with a Student-t distribution with n-1 degrees
    of freedom. A negative statistic means model A had the lower mean squared error.
    Returns (statistic, two-sided p-value, kernel used).
    """
    d = np.square(np.asarray(errors_a, dtype=float)) - np.square(np.asarray(errors_b, dtype=float))
    n = len(d)
    gammas = [autocovariance(d, k) for k in range(min(horizon, n))]
    long_run_variance, kernel = gammas[0] + 2 * sum(gammas[1:]), "rectangular"
    if long_run_variance <= 0:
        weights = [1 - k / horizon for k in range(1, len(gammas))]
        long_run_variance, kernel = gammas[0] + 2 * float(np.dot(weights, gammas[1:])), "bartlett"
    if long_run_variance <= 0:
        return np.nan, np.nan, kernel
    hln = np.sqrt((n + 1 - 2 * horizon + horizon * (horizon - 1) / n) / n)
    statistic = hln * np.mean(d) / np.sqrt(long_run_variance / n)
    return float(statistic), float(2 * stats.t.sf(abs(statistic), df=n - 1)), kernel


def load_forecasts(panel):
    """Combine all forecast files, attach the realised values and label each forecast with its period."""
    files = [config.RESULTS_DIR / "forecasts_baselines.csv", config.RESULTS_DIR / "forecasts_tft.csv"]
    forecasts = pd.concat([pd.read_csv(f, parse_dates=["origin", "date"]) for f in files], ignore_index=True)
    forecasts = forecasts[forecasts["date"] <= panel.index[-1]].copy()
    forecasts["actual"] = [panel.at[date, target] for date, target in zip(forecasts["date"], forecasts["target"])]
    forecasts["error"] = forecasts["actual"] - forecasts["forecast"]
    forecasts["period"] = [config.assign_period(o, d) for o, d in zip(forecasts["origin"], forecasts["date"])]
    forecasts = forecasts.dropna(subset=["period"])

    seeds = forecasts[forecasts["model"].str.startswith("TFT")]
    if seeds["model"].nunique() > 1:  # average the forecasts of all seeds into an ensemble
        keys = ["target", "horizon", "origin", "date", "period", "actual"]
        ensemble = seeds.groupby(keys, as_index=False)[["forecast", "q10", "q90"]].mean()
        ensemble["model"], ensemble["error"] = "TFT_ensemble", ensemble["actual"] - ensemble["forecast"]
        forecasts = pd.concat([forecasts, ensemble], ignore_index=True)

    for name, (parent, start) in SUBPERIODS.items():
        sub = forecasts[(forecasts["period"] == parent) & (forecasts["date"] >= start)].copy()
        sub["period"] = name
        forecasts = pd.concat([forecasts, sub], ignore_index=True)
    return forecasts


def metrics_table(forecasts):
    """n, RMSE, MAE, RMSE relative to the random walk and (TFT only) coverage of the 10-90% interval."""
    rows = []
    for (period, target, horizon, model), g in forecasts.groupby(["period", "target", "horizon", "model"], sort=False):
        inside = (g["actual"] >= g["q10"]) & (g["actual"] <= g["q90"])
        rows.append({"period": period, "target": target, "horizon": horizon, "model": model, "n": len(g),
                     "rmse": rmse(g["error"]), "mae": mae(g["error"]),
                     "coverage_80": inside.mean() if g["q10"].notna().all() else np.nan})
    table = pd.DataFrame(rows)
    rw = table[table["model"] == "RW"].set_index(["period", "target", "horizon"])["rmse"]
    table["rel_rmse"] = table["rmse"] / table.set_index(["period", "target", "horizon"]).index.map(rw).to_numpy()
    return table


def dm_table(forecasts, periods):
    """DM tests of the TFT against every benchmark, and of ARIMA and VAR against the random walk."""
    pairs = [("TFT", b) for b in BENCHMARKS] + [("ARIMA", "RW"), ("VAR", "RW"), ("Mean", "RW")]
    if "TFT_ensemble" in set(forecasts["model"]):
        pairs += [("TFT_ensemble", b) for b in ["RW", "Mean", "ARIMA", "VAR"]]
    rows = []
    for (period, target, horizon), g in forecasts.groupby(["period", "target", "horizon"], sort=False):
        if period not in periods:
            continue
        errors = g.pivot(index="date", columns="model", values="error").sort_index()
        for a, b in pairs:
            statistic, pvalue, kernel = diebold_mariano(errors[a], errors[b], horizon)
            rows.append({"period": period, "target": target, "horizon": horizon, "model_a": a, "model_b": b,
                         "n": len(errors), "mse_ratio": np.mean(errors[a] ** 2) / np.mean(errors[b] ** 2),
                         "dm_stat": statistic, "p_value": pvalue, "kernel": kernel})
    return pd.DataFrame(rows, columns=DM_COLUMNS)


def seed_table(metrics):
    """Spread of the TFT's RMSE across random seeds."""
    seeds = metrics[metrics["model"].str.startswith("TFT") & (metrics["model"] != "TFT_ensemble")]
    table = seeds.groupby(["period", "target", "horizon"], sort=False)["rel_rmse"].agg(["count", "min", "median", "max"])
    return table.rename(columns={"count": "seeds"}).reset_index()


def stars(p):
    return "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.10 else ""


def summary_markdown(metrics, dm, best):
    """Human-readable tables: RMSE relative to the random walk, and DM tests of the TFT."""
    lines = ["# Result tables", "", "Generated by `src/evaluate.py`; do not edit by hand.", ""]
    models = [m for m in ["RW", "Mean", "ARIMA", "VAR", "TFT", "TFT_ensemble"] if m in set(metrics["model"])]
    for period in metrics["period"].unique():
        part = metrics[metrics["period"] == period]
        lines += [f"## RMSE relative to the random walk: {period}", "",
                  "| target | h | n | RW RMSE | " + " | ".join(models[1:]) + " |", "|---|---|---|---|" + "---|" * (len(models) - 1)]
        for (target, horizon), g in part.groupby(["target", "horizon"], sort=False):
            g = g.set_index("model")
            cells = [f"{g.at[m, 'rel_rmse']:.3f}" for m in models[1:]]
            lines.append(f"| {target} | {horizon} | {g.at['RW', 'n']} | {g.at['RW', 'rmse']:.5f} | " + " | ".join(cells) + " |")
        lines.append("")
    for period in dm["period"].unique():
        part = dm[(dm["period"] == period) & (dm["model_a"] == "TFT")]
        columns = [b for b in BENCHMARKS if b in set(part["model_b"])]
        lines += [f"## Diebold-Mariano tests, TFT against each benchmark: {period}", "",
                  "HLN-corrected statistic (two-sided p-value). Negative: the TFT has the lower squared error. "
                  "`†` marks the classical baseline with the lowest validation RMSE. \\* p<0.10, \\*\\* p<0.05, \\*\\*\\* p<0.01.", "",
                  "| target | h | n | " + " | ".join(columns) + " |", "|---|---|---|" + "---|" * len(columns)]
        for (target, horizon), g in part.groupby(["target", "horizon"], sort=False):
            g = g.set_index("model_b")
            cells = [f"{g.at[b, 'dm_stat']:.2f}{stars(g.at[b, 'p_value']).replace('*', BACKSLASH_STAR)} ({g.at[b, 'p_value']:.3f})"
                     + ("†" if best.get((target, horizon)) == b else "") for b in columns]
            lines.append(f"| {target} | {horizon} | {g['n'].iloc[0]} | " + " | ".join(cells) + " |")
        lines.append("")
    return "\n".join(lines)


def main(until):
    forecasts = load_forecasts(load_panel(until))
    metrics = metrics_table(forecasts)
    validation = metrics[(metrics["period"] == "val") & metrics["model"].isin(CLASSICAL)]
    best = validation.loc[validation.groupby(["target", "horizon"])["rmse"].idxmin()]
    best = {(r.target, r.horizon): r.model for r in best.itertuples()}
    dm = dm_table(forecasts, [p for p in metrics["period"].unique() if p != "val"])

    metrics.to_csv(config.RESULTS_DIR / "metrics.csv", index=False, float_format="%.6g")
    dm.to_csv(config.RESULTS_DIR / "dm_tests.csv", index=False, float_format="%.6g")
    seed_table(metrics).to_csv(config.RESULTS_DIR / "seed_robustness.csv", index=False, float_format="%.4f")
    summary = summary_markdown(metrics, dm, best)
    (config.RESULTS_DIR / "summary.md").write_text(summary + "\n")
    print(summary)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--until", choices=["val", "test"], default="test", help="last period to evaluate")
    main(parser.parse_args().until)
