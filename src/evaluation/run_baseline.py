"""
Runs the seasonal-naive and climatology baselines across every rolling-origin
backtest fold and reports per-fold + summary pinball loss and calibration.

Usage:
    python -m src.evaluation.run_baseline [--config configs/default.yaml]
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import yaml

from src.data.loader import get_data
from src.evaluation.backtest import make_rolling_folds
from src.evaluation.metrics import calibration_curve, pinball_loss
from src.evaluation.report_utils import save_report
from src.models.baselines import climatology_quantiles, seasonal_naive_quantiles

BASELINES = {
    "seasonal_naive": seasonal_naive_quantiles,
    "climatology": climatology_quantiles,
}

# A handful of representative quantiles to inspect in detail, rather than
# dumping all `len(quantile_levels)` (e.g. 27) columns of predicted values.
SELECTED_QUANTILES = [0.05, 0.25, 0.5, 0.75, 0.95]


def run(config_path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    with open(config_path) as f:
        config = yaml.safe_load(f)

    quantile_levels = config["quantiles"]["levels"]
    selected_quantiles = [q for q in SELECTED_QUANTILES if q in quantile_levels]
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

    rows = []
    quantile_rows = []
    for fold in folds:
        y_true = fold.test_df["load"].to_numpy()
        valid = ~np.isnan(y_true)

        for name, baseline_fn in BASELINES.items():
            preds = baseline_fn(fold.train_df, fold.test_df, quantile_levels)
            loss = pinball_loss(y_true[valid], preds[valid], quantile_levels)
            calib = calibration_curve(y_true[valid], preds[valid], quantile_levels)
            coverage_90 = _interval_coverage_from_calibration(y_true[valid], preds[valid], quantile_levels, 0.05, 0.95)
            rows.append({
                "test_task": fold.test_task,
                "baseline": name,
                "n_obs": int(valid.sum()),
                "pinball_loss": loss,
                "coverage_90": coverage_90,
            })

            for q in selected_quantiles:
                idx = quantile_levels.index(q)
                quantile_rows.append({
                    "test_task": fold.test_task,
                    "baseline": name,
                    "quantile": q,
                    "mean_predicted_load": float(preds[valid, idx].mean()),
                    "empirical_coverage": calib["empirical"][idx],
                })

    results = pd.DataFrame(rows)
    quantile_detail = pd.DataFrame(quantile_rows)
    return results, quantile_detail


def _interval_coverage_from_calibration(y_true, preds, quantile_levels, lo, hi):
    quantile_levels = list(quantile_levels)
    lo_idx = quantile_levels.index(lo)
    hi_idx = quantile_levels.index(hi)
    lower = preds[:, lo_idx]
    upper = preds[:, hi_idx]
    return float(((y_true >= lower) & (y_true <= upper)).mean())


def summarize(results: pd.DataFrame) -> pd.DataFrame:
    return (
        results.groupby("baseline")["pinball_loss"]
        .agg(["mean", "std", "count"])
        .rename(columns={"mean": "mean_pinball_loss", "std": "std_pinball_loss", "count": "n_folds"})
    )


def summarize_quantiles(quantile_detail: pd.DataFrame) -> pd.DataFrame:
    """Averages the selected-quantile predictions and calibration across
    folds. `empirical_coverage` should be close to `quantile` for a
    well-calibrated model (e.g. ~0.5 of outcomes at or below the predicted
    median)."""
    summary = (
        quantile_detail.groupby(["baseline", "quantile"])
        .agg(mean_predicted_load=("mean_predicted_load", "mean"), empirical_coverage=("empirical_coverage", "mean"))
        .reset_index()
    )
    summary["calibration_gap"] = summary["empirical_coverage"] - summary["quantile"]
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    results, quantile_detail = run(args.config)
    pd.set_option("display.width", 120)
    print(results.to_string(index=False))
    print()
    summary = summarize(results)
    print("Summary across folds:")
    print(summary)
    print()
    quantile_summary = summarize_quantiles(quantile_detail)
    print(f"Selected-quantile detail (mean across folds): {SELECTED_QUANTILES}")
    print(quantile_summary.to_string(index=False))

    save_report(results, "run_baseline/results.csv")
    save_report(summary, "run_baseline/summary.csv")
    save_report(quantile_summary, "run_baseline/quantile_summary.csv")
    print("\nSaved results to reports/run_baseline/")
