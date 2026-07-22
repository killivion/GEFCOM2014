"""
Runs the baselines (climatology, seasonal-naive) and the LightGBM model
across every rolling-origin backtest fold ONCE and produces the full
comparison from that single pass: a per-fold pinball-loss table (one
column per method), a summary table (mean/std/n_folds, best first), a
mean-coverage table, a paired t-test across folds of the model against
each baseline, and a calibration reliability diagram (nominal vs.
empirical, pooled across all folds).

Calibration used to live in a separate script (plot_calibration.py), but
it needs the exact same per-fold baseline/model predictions this script
already computes -- running it separately meant retraining the LightGBM
model across all 14 folds a second time for no reason. Now it's all one
run.

Usage:
    python -m src.evaluation.run_comparison [--config configs/default.yaml] [--feature-set full|minimal|both]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from scipy import stats

from src.data.loader import get_data
from src.evaluation.backtest import make_rolling_folds
from src.evaluation.metrics import calibration_curve, pinball_loss
from src.evaluation.report_utils import save_report
from src.models.baselines import climatology_quantiles, seasonal_naive_quantiles
from src.models.lightgbm_model import lightgbm_quantiles

BASELINES = {
    "seasonal_naive": seasonal_naive_quantiles,
    "climatology": climatology_quantiles,
}

DEFAULT_CALIBRATION_PLOT_PATH = Path("reports/run_comparison/calibration.png")


def run(config_path: str, feature_set: str = "full") -> tuple[pd.DataFrame, dict, list[float]]:
    """Returns (results, calibration_curves, quantile_levels).

    results: one row per (fold, method) with pinball_loss and coverage_90.
    calibration_curves: {method: calibration_curve(...)}, pooling every
        fold's predictions per method before computing the curve, so rare
        quantiles get enough observations to be meaningful.
    """
    with open(config_path) as f:
        config = yaml.safe_load(f)

    quantile_levels = config["quantiles"]["levels"]
    use_actual_future_temperature = config["leakage"]["use_actual_future_temperature"]
    model_params = config["model"]["params"]

    data = get_data(
        config["data"]["raw_load_dir"],
        n_tasks=config["data"]["n_tasks"],
        processed_path=config["data"]["processed_path"],
    )
    folds = make_rolling_folds(
        data.df,
        first_test_task=config["backtest"]["first_test_task"],
        last_test_task=config["backtest"]["last_test_task"],
    )
    model_feature_sets = ["full", "minimal"] if feature_set == "both" else [feature_set]
    method_names = list(BASELINES) + [f"lightgbm_{fs}" for fs in model_feature_sets]

    rows = []
    all_y = {name: [] for name in method_names}
    all_preds = {name: [] for name in method_names}

    for fold in folds:
        y_true = fold.test_df["load"].to_numpy()
        valid = ~np.isnan(y_true)

        for name, baseline_fn in BASELINES.items():
            preds = baseline_fn(fold.train_df, fold.test_df, quantile_levels)
            _record(rows, all_y, all_preds, fold.test_task, name, y_true, valid, preds, quantile_levels)

        for fs in model_feature_sets:
            preds = lightgbm_quantiles(
                fold.train_df,
                fold.test_df,
                quantile_levels,
                use_actual_future_temperature=use_actual_future_temperature,
                model_params=model_params,
                feature_set=fs,
            )
            _record(rows, all_y, all_preds, fold.test_task, f"lightgbm_{fs}", y_true, valid, preds, quantile_levels)

    results = pd.DataFrame(rows)
    curves = {
        name: calibration_curve(np.concatenate(all_y[name]), np.concatenate(all_preds[name], axis=0), quantile_levels)
        for name in method_names
    }
    return results, curves, quantile_levels


def _record(rows, all_y, all_preds, test_task, method, y_true, valid, preds, quantile_levels):
    loss = pinball_loss(y_true[valid], preds[valid], quantile_levels)
    coverage_90 = _interval_coverage_90(y_true[valid], preds[valid], quantile_levels)
    rows.append({"test_task": test_task, "method": method, "pinball_loss": loss, "coverage_90": coverage_90})
    all_y[method].append(y_true[valid])
    all_preds[method].append(preds[valid])


def _interval_coverage_90(y_true, preds, quantile_levels, lo=0.05, hi=0.95):
    quantile_levels = list(quantile_levels)
    lo_idx = quantile_levels.index(lo)
    hi_idx = quantile_levels.index(hi)
    lower, upper = preds[:, lo_idx], preds[:, hi_idx]
    return float(((y_true >= lower) & (y_true <= upper)).mean())


def wide_pinball_table(results: pd.DataFrame) -> pd.DataFrame:
    """One row per fold, one column per method -- easy to eyeball."""
    return results.pivot(index="test_task", columns="method", values="pinball_loss")


def summarize(results: pd.DataFrame) -> pd.DataFrame:
    return (
        results.groupby("method")["pinball_loss"]
        .agg(["mean", "std", "count"])
        .rename(columns={"mean": "mean_pinball_loss", "std": "std_pinball_loss", "count": "n_folds"})
        .sort_values("mean_pinball_loss")
    )


def coverage_summary(results: pd.DataFrame) -> pd.Series:
    return results.groupby("method")["coverage_90"].mean().rename("mean_coverage_90").sort_index()


def paired_ttest_vs_model(results: pd.DataFrame, model_method: str) -> pd.DataFrame:
    """Paired t-test across folds of each OTHER method's per-fold pinball
    loss against `model_method`'s. Pairing by fold controls for the fact
    that some months are just harder to forecast than others (see Task 4),
    so this asks "does the model win consistently, fold for fold?" rather
    than relying on a single aggregate mean, per the assignment's request
    for a "sensible statistical comparison ... over folds"."""
    wide = wide_pinball_table(results)
    if model_method not in wide.columns:
        raise ValueError(f"{model_method!r} not among methods: {list(wide.columns)}")

    rows = []
    for other in wide.columns:
        if other == model_method:
            continue
        model_losses = wide[model_method]
        other_losses = wide[other]
        diff = other_losses - model_losses  # positive => model has lower (better) loss
        t_stat, p_value = stats.ttest_rel(other_losses, model_losses)
        rows.append({
            "baseline": other,
            "mean_diff_baseline_minus_model": diff.mean(),
            "t_stat": t_stat,
            "p_value": p_value,
            "model_better_on_average": bool(diff.mean() > 0),
        })
    return pd.DataFrame(rows)


def calibration_tables(curves: dict) -> pd.DataFrame:
    """Long-format table (method, nominal, empirical, gap) -- one row per
    method per quantile level, ready to save or print."""
    tables = []
    for name, curve in curves.items():
        table = pd.DataFrame({"nominal": curve["nominal"], "empirical": curve["empirical"]})
        table["gap"] = table["empirical"] - table["nominal"]
        table.insert(0, "method", name)
        tables.append(table)
    return pd.concat(tables, ignore_index=True)


def plot_calibration(curves: dict, output_path: Path = DEFAULT_CALIBRATION_PLOT_PATH) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="perfect calibration")
    for name, curve in curves.items():
        ax.plot(curve["nominal"], curve["empirical"], marker="o", markersize=3, label=name)

    ax.set_xlabel("nominal quantile")
    ax.set_ylabel("empirical coverage")
    ax.set_title("Calibration: nominal vs. empirical quantile coverage\n(pooled across all rolling-origin folds)")
    ax.legend()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--feature-set", choices=["full", "minimal", "both"], default="full",
                         help="Which LightGBM feature set(s) to include in the comparison.")
    parser.add_argument("--calibration-plot", default=str(DEFAULT_CALIBRATION_PLOT_PATH),
                         help="Where to save the calibration reliability diagram.")
    args = parser.parse_args()

    results, curves, quantile_levels = run(args.config, feature_set=args.feature_set)
    pd.set_option("display.width", 120)

    pinball_table = wide_pinball_table(results)
    print("Per-fold pinball loss (rows = fold, columns = method):")
    print(pinball_table.round(2).to_string())
    print()

    summary = summarize(results)
    print("Summary across folds (best first):")
    print(summary.round(3))
    print()

    coverage = coverage_summary(results)
    print("Mean 90%-interval coverage per method (target ~0.90):")
    print(coverage.round(3))
    print()

    methods = results["method"].unique()
    primary_model = "lightgbm_full" if "lightgbm_full" in methods else "lightgbm_minimal"
    paired_test = paired_ttest_vs_model(results, primary_model)
    print(f"Paired t-test across folds: {primary_model} vs each baseline "
          f"(positive mean_diff = model wins on average):")
    print(paired_test.round(4).to_string(index=False))
    print()

    calibration = calibration_tables(curves)
    print("Calibration (nominal -> empirical), per method:")
    for name in curves:
        print(f"\n{name}:")
        print(calibration[calibration["method"] == name].drop(columns="method").round(3).to_string(index=False))

    plot_path = plot_calibration(curves, Path(args.calibration_plot))
    print(f"\nSaved calibration plot to {plot_path}")

    save_report(results, "run_comparison/results.csv")
    save_report(pinball_table, "run_comparison/pinball_by_fold.csv")
    save_report(summary, "run_comparison/summary.csv")
    save_report(coverage, "run_comparison/coverage.csv")
    save_report(paired_test, "run_comparison/paired_ttest.csv")
    save_report(calibration, "run_comparison/calibration_curves.csv")
    print("Saved results to reports/run_comparison/")
