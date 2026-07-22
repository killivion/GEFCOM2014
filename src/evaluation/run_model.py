"""
Runs the LightGBM quantile-regression model across every rolling-origin
backtest fold and reports per-fold + summary pinball loss and calibration,
mirroring run_baseline.py's output. No comparison against the baselines
yet -- that's a separate next step.

Usage:
    python -m src.evaluation.run_model [--config configs/default.yaml] [--feature-set full|minimal|both]

--feature-set full (default): only the intended full-feature model --
    fastest useful run.
--feature-set minimal: only the calendar+temperature-only baseline (no
    load-lag features) -- fast, no recursion needed.
--feature-set both: runs both and reports them side by side, so you can
    see whether the load-lag/rolling features in "full" actually help.
    Roughly doubles compute time versus a single feature set.
"""
from __future__ import annotations

import argparse
import sys
import time

import numpy as np
import pandas as pd
import yaml

from src.data.loader import get_data
from src.evaluation.backtest import make_rolling_folds
from src.evaluation.metrics import calibration_curve, pinball_loss
from src.evaluation.report_utils import save_report
from src.models.lightgbm_model import lightgbm_quantiles

# A handful of representative quantiles to inspect in detail, rather than
# dumping all `len(quantile_levels)` (e.g. 27) columns of predicted values.
SELECTED_QUANTILES = [0.05, 0.25, 0.5, 0.75, 0.95]

FEATURE_SET_CHOICES = ["full", "minimal", "both"]


def _format_seconds(seconds: float) -> str:
    minutes, secs = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h{minutes:02d}m{secs:02d}s"
    if minutes:
        return f"{minutes}m{secs:02d}s"
    return f"{secs}s"


class _ProgressBar:
    """Simple in-place text progress bar with an elapsed/ETA estimate.

    One "unit" is one quantile-level model finishing training; ETA is
    extrapolated linearly from the average time per unit so far, which is
    a reasonable estimate since every model here trains on roughly the
    same amount of data.
    """

    def __init__(self, total: int, width: int = 30):
        self.total = total
        self.width = width
        self.done = 0
        self.start = time.time()

    def update(self, done: int) -> None:
        self.done = done
        frac = self.done / self.total if self.total else 1.0
        filled = int(self.width * frac)
        bar = "#" * filled + "-" * (self.width - filled)
        elapsed = time.time() - self.start
        eta = (elapsed / self.done * (self.total - self.done)) if self.done else 0.0
        sys.stdout.write(
            f"\r[{bar}] {self.done}/{self.total} ({frac * 100:5.1f}%) "
            f"elapsed {_format_seconds(elapsed)}, ETA {_format_seconds(eta)}   "
        )
        sys.stdout.flush()

    def close(self) -> None:
        sys.stdout.write("\n")
        sys.stdout.flush()


def run(config_path: str, show_progress: bool = True, feature_set: str = "full") -> tuple[pd.DataFrame, pd.DataFrame]:
    if feature_set not in FEATURE_SET_CHOICES:
        raise ValueError(f"feature_set must be one of {FEATURE_SET_CHOICES}, got {feature_set!r}")
    feature_sets_to_run = ["full", "minimal"] if feature_set == "both" else [feature_set]

    with open(config_path) as f:
        config = yaml.safe_load(f)

    quantile_levels = config["quantiles"]["levels"]
    selected_quantiles = [q for q in SELECTED_QUANTILES if q in quantile_levels]
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

    total_units = len(folds) * len(quantile_levels) * len(feature_sets_to_run)
    progress = _ProgressBar(total_units) if show_progress else None
    units_done = 0

    rows = []
    quantile_rows = []
    for fold in folds:
        y_true = fold.test_df["load"].to_numpy()
        valid = ~np.isnan(y_true)

        for fs in feature_sets_to_run:

            def _on_quantile_done(_i, _total):
                nonlocal units_done
                units_done += 1
                if progress is not None:
                    progress.update(units_done)

            preds = lightgbm_quantiles(
                fold.train_df,
                fold.test_df,
                quantile_levels,
                use_actual_future_temperature=use_actual_future_temperature,
                model_params=model_params,
                on_quantile_done=_on_quantile_done,
                feature_set=fs,
            )
            loss = pinball_loss(y_true[valid], preds[valid], quantile_levels)
            calib = calibration_curve(y_true[valid], preds[valid], quantile_levels)
            coverage_90 = _interval_coverage_from_calibration(y_true[valid], preds[valid], quantile_levels, 0.05, 0.95)
            rows.append({
                "test_task": fold.test_task,
                "feature_set": fs,
                "n_obs": int(valid.sum()),
                "pinball_loss": loss,
                "coverage_90": coverage_90,
            })

            for q in selected_quantiles:
                idx = quantile_levels.index(q)
                quantile_rows.append({
                    "test_task": fold.test_task,
                    "feature_set": fs,
                    "quantile": q,
                    "mean_predicted_load": float(preds[valid, idx].mean()),
                    "empirical_coverage": calib["empirical"][idx],
                })

    if progress is not None:
        progress.close()

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
        results.groupby("feature_set")["pinball_loss"]
        .agg(["mean", "std", "count"])
        .rename(columns={"mean": "mean_pinball_loss", "std": "std_pinball_loss", "count": "n_folds"})
    )


def summarize_quantiles(quantile_detail: pd.DataFrame) -> pd.DataFrame:
    """Averages the selected-quantile predictions and calibration across
    folds, per feature set. `empirical_coverage` should be close to
    `quantile` for a well-calibrated model."""
    summary = (
        quantile_detail.groupby(["feature_set", "quantile"])
        .agg(mean_predicted_load=("mean_predicted_load", "mean"), empirical_coverage=("empirical_coverage", "mean"))
        .reset_index()
    )
    summary["calibration_gap"] = summary["empirical_coverage"] - summary["quantile"]
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--no-progress", action="store_true", help="Disable the progress bar (e.g. when piping output to a file)")
    parser.add_argument("--feature-set", choices=FEATURE_SET_CHOICES, default="full",
                         help="'full' (default, fastest): the intended model. 'minimal': calendar+temperature-only "
                              "baseline. 'both': run both and compare (roughly doubles compute time).")
    args = parser.parse_args()

    results, quantile_detail = run(args.config, show_progress=not args.no_progress, feature_set=args.feature_set)
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

    save_report(results, "run_model/results.csv")
    save_report(summary, "run_model/summary.csv")
    save_report(quantile_summary, "run_model/quantile_summary.csv")
    print("\nSaved results to reports/run_model/")
