"""
Sequential conformal recalibration of the final model (see
src/models/conformal.py), evaluated on the saved predictions of a
run_comparison run -- no model is retrained.

Protocol, the same as for every other decision:
1. Every variant (method x calibration window) recalibrates each test
   month from the earlier test months only.
2. ONE variant is selected by its mean pinball loss on the DEVELOPMENT
   folds; "none" (the unadjusted model) competes too, so the selection
   can also keep the model as it is.
3. The selected variant, lightgbm_full_conformal, is then compared with
   the unadjusted model and every baseline per fold group, with the same
   tests as run_comparison. All variants' holdout scores are written as
   well, for transparency -- not to pick from.

Usage (after run_comparison --save-predictions):
    python -m src.evaluation.run_conformal [--run-dir run_comparison] [--output-dir run_conformal]
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from src.evaluation.metrics import (calibration_curve, interval_coverage_at_quantiles, pinball_loss,
                                    pinball_loss_per_observation)
from src.evaluation.report_utils import REPORTS_DIR, save_report
from src.evaluation.run_comparison import (PRIMARY_MODEL, by_fold_group, calibration_tables, fold_group,
                                           plot_calibration, wide_pinball_table, write_run_info)
from src.models.conformal import CONFORMAL_METHODS, sequential_recalibration

CONFORMAL_MODEL = f"{PRIMARY_MODEL}_conformal"
WINDOWS = (None, 3)  # all earlier test months / the last three


def variant_name(method: str, window: int | None) -> str:
    return f"{method}_{'all' if window is None else f'last{window}'}"


def per_fold(y, preds, test_task, quantile_levels, method_name, dev_last):
    rows = []
    for task in np.unique(test_task):
        m = test_task == task
        rows.append({"test_task": int(task), "fold_group": fold_group(int(task), dev_last), "method": method_name,
                     "pinball_loss": pinball_loss(y[m], preds[m], quantile_levels),
                     "coverage_90": interval_coverage_at_quantiles(y[m], preds[m], quantile_levels)})
    return rows


def main(run_dir: str, output_dir: str) -> None:
    run_info = json.loads((REPORTS_DIR / run_dir / "run_info.json").read_text())
    config = run_info["config"]
    dev_last = config["backtest"]["development_last_test_task"]
    z = np.load(REPORTS_DIR / run_dir / "predictions.npz", allow_pickle=False)
    levels = [float(q) for q in z["quantile_levels"]]
    y, test_task = z["y"], z["test_task"]
    timestamps = pd.read_csv(REPORTS_DIR / run_dir / "hourly_losses.csv", usecols=["timestamp", "test_task"])
    if len(timestamps) != len(y) or not (timestamps["test_task"].to_numpy() == test_task).all():
        raise ValueError(f"predictions.npz and hourly_losses.csv in reports/{run_dir} do not match row for row")
    methods = [k for k in z.files if k not in ("y", "test_task", "timestamp", "quantile_levels")]
    model_preds = z[PRIMARY_MODEL]

    variants = {"none": model_preds}
    for method in CONFORMAL_METHODS:
        for window in WINDOWS:
            variants[variant_name(method, window)] = sequential_recalibration(
                y, model_preds, test_task, levels, method, window)

    variant_rows = []
    for name, preds in variants.items():
        variant_rows += per_fold(y, preds, test_task, levels, name, dev_last)
    variant_folds = pd.DataFrame(variant_rows)
    variant_summary = (variant_folds.groupby(["fold_group", "method"])
                       .agg(mean_pinball_loss=("pinball_loss", "mean"), mean_coverage_90=("coverage_90", "mean"))
                       .reset_index())
    dev = variant_summary[variant_summary["fold_group"] == "development"].set_index("method")
    selected = str(dev["mean_pinball_loss"].idxmin())
    print("Variants, mean pinball loss per fold group (selection uses development only):")
    print(variant_summary.pivot(index="method", columns="fold_group", values="mean_pinball_loss")
          .round(3).to_string())
    print(f"\nSelected on the development folds: {selected}\n")

    rows = []
    for name in methods:
        rows += per_fold(y, z[name], test_task, levels, name, dev_last)
    rows += per_fold(y, variants[selected], test_task, levels, CONFORMAL_MODEL, dev_last)
    results = pd.DataFrame(rows)
    hourly = timestamps.copy()
    for name in methods:
        hourly[name] = pinball_loss_per_observation(y, z[name], levels)
    hourly[CONFORMAL_MODEL] = pinball_loss_per_observation(y, variants[selected], levels)

    if selected == "none":
        # Identical to the unadjusted model; a test against itself is undefined.
        results = results[results["method"] != PRIMARY_MODEL]
        hourly = hourly.drop(columns=PRIMARY_MODEL)
    summary, significance = by_fold_group(results, hourly, CONFORMAL_MODEL, config.get("statistics", {}))
    pd.set_option("display.width", 200)
    print(summary.round(3).to_string())
    print()
    print(significance.round(4).to_string(index=False))

    curves = {name: calibration_curve(y, preds, levels)
              for name, preds in ((PRIMARY_MODEL, model_preds), (CONFORMAL_MODEL, variants[selected]))}
    plot_calibration(curves, REPORTS_DIR / output_dir / "calibration.png")
    save_report(variant_summary, f"{output_dir}/variants_by_fold_group.csv")
    save_report(variant_folds, f"{output_dir}/variants_by_fold.csv")
    save_report(results, f"{output_dir}/results.csv")
    save_report(wide_pinball_table(results), f"{output_dir}/pinball_by_fold.csv")
    save_report(summary, f"{output_dir}/summary.csv")
    save_report(significance, f"{output_dir}/significance.csv")
    save_report(calibration_tables(curves), f"{output_dir}/calibration_curves.csv")
    write_run_info(output_dir, run_info["config_path"], config,
                   {"source_run": run_dir, "source_commit": run_info["git"]["commit"],
                    "selected_variant": selected, "n_quantiles": len(levels)})
    print(f"Saved results to reports/{output_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", default="run_comparison", help="reports/ subfolder with predictions.npz.")
    parser.add_argument("--output-dir", default="run_conformal")
    args = parser.parse_args()
    main(args.run_dir, args.output_dir)
