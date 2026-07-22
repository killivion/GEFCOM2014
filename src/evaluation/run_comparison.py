"""
Runs the baselines (climatology, seasonal-naive) and the LightGBM model
across every rolling-origin backtest fold and produces one neat
comparison: a per-fold pinball-loss table (one column per method), a
summary table (mean/std/n_folds, best first), a mean-coverage table, and
a paired t-test across folds of the model against each baseline -- so
"the model wins" is backed by more than a single aggregate number, per
the assignment's "evidence that improvements are real" requirement.

Usage:
    python -m src.evaluation.run_comparison [--config configs/default.yaml] [--feature-set full|minimal|both]
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import yaml
from scipy import stats

from src.data.loader import get_data
from src.evaluation.backtest import make_rolling_folds
from src.evaluation.metrics import pinball_loss
from src.models.baselines import climatology_quantiles, seasonal_naive_quantiles
from src.models.lightgbm_model import lightgbm_quantiles

BASELINES = {
    "seasonal_naive": seasonal_naive_quantiles,
    "climatology": climatology_quantiles,
}


def run(config_path: str, feature_set: str = "full") -> pd.DataFrame:
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

    rows = []
    for fold in folds:
        y_true = fold.test_df["load"].to_numpy()
        valid = ~np.isnan(y_true)

        for name, baseline_fn in BASELINES.items():
            preds = baseline_fn(fold.train_df, fold.test_df, quantile_levels)
            _record(rows, fold.test_task, name, y_true, valid, preds, quantile_levels)

        for fs in model_feature_sets:
            preds = lightgbm_quantiles(
                fold.train_df,
                fold.test_df,
                quantile_levels,
                use_actual_future_temperature=use_actual_future_temperature,
                model_params=model_params,
                feature_set=fs,
            )
            _record(rows, fold.test_task, f"lightgbm_{fs}", y_true, valid, preds, quantile_levels)

    return pd.DataFrame(rows)


def _record(rows, test_task, method, y_true, valid, preds, quantile_levels):
    loss = pinball_loss(y_true[valid], preds[valid], quantile_levels)
    coverage_90 = _interval_coverage_90(y_true[valid], preds[valid], quantile_levels)
    rows.append({"test_task": test_task, "method": method, "pinball_loss": loss, "coverage_90": coverage_90})


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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--feature-set", choices=["full", "minimal", "both"], default="full",
                         help="Which LightGBM feature set(s) to include in the comparison.")
    args = parser.parse_args()

    results = run(args.config, feature_set=args.feature_set)
    pd.set_option("display.width", 120)

    print("Per-fold pinball loss (rows = fold, columns = method):")
    print(wide_pinball_table(results).round(2).to_string())
    print()
    print("Summary across folds (best first):")
    print(summarize(results).round(3))
    print()
    print("Mean 90%-interval coverage per method (target ~0.90):")
    print(coverage_summary(results).round(3))
    print()

    methods = results["method"].unique()
    primary_model = "lightgbm_full" if "lightgbm_full" in methods else "lightgbm_minimal"
    print(f"Paired t-test across folds: {primary_model} vs each baseline "
          f"(positive mean_diff = model wins on average):")
    print(paired_ttest_vs_model(results, primary_model).round(4).to_string(index=False))
