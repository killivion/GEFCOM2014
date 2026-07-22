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
from src.models.baselines import climatology_quantiles, seasonal_naive_quantiles

BASELINES = {
    "seasonal_naive": seasonal_naive_quantiles,
    "climatology": climatology_quantiles,
}


def run(config_path: str) -> pd.DataFrame:
    with open(config_path) as f:
        config = yaml.safe_load(f)

    quantile_levels = config["quantiles"]["levels"]
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

    results = pd.DataFrame(rows)
    return results


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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    results = run(args.config)
    pd.set_option("display.width", 120)
    print(results.to_string(index=False))
    print()
    print("Summary across folds:")
    print(summarize(results))
