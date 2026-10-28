"""
Runs the baselines and the LightGBM model across every rolling-origin
backtest fold ONCE and produces the full comparison from that single
pass: a per-fold pinball-loss table (one column per method), a summary
per fold group (development / holdout / all), the significance tests of
the model against every other method, and a calibration reliability
diagram (nominal vs. empirical, pooled across all folds).

Significance testing (see significance_vs_model). The unit is the
per-HOUR pinball loss over the test months (concatenated in time order),
plus the per-month (fold) losses. Month-ahead errors are correlated
within a month (a weather anomaly lasts), so several tests with
different dependence assumptions are reported side by side:
- Diebold-Mariano with a Newey-West variance at config.statistics.hac_max_lag
  (one week) and at hac_max_lag_long (one month) lags;
- a circular block bootstrap (blocks of bootstrap_block_hours) and a
  cluster bootstrap that resamples whole test months -- the most
  conservative of the hourly tests;
- per fold: months won by the model and a Wilcoxon signed-rank test on
  the month losses (with 7 months its exact p cannot go below 0.0156).
p_conservative is the LARGEST of those p-values; it is Holm-adjusted
within each family of comparisons (baselines; model variants), never
across the oracle, which is reference only.

Usage:
    python -m src.evaluation.run_comparison [--config configs/default.yaml] [--feature-set full|minimal|both] [--baseline <name>|all] [--output-dir run_comparison] [--folds all|development|holdout] [--official-benchmark] [--save-predictions]

--official-benchmark also scores the organisers' benchmark files
(src/data/official_benchmark.py) as the method gefcom_benchmark.
--save-predictions writes every method's quantile predictions to
reports/<output-dir>/predictions.npz (not tracked by git), which
run_conformal.py post-processes.

--output-dir is the subfolder of reports/ to write to; an oracle config
(actual forecast-month temperature) refuses to write to the default
folder, so it can never overwrite the honest results. Every output folder
also gets run_info.json (git commit, config) so each report says exactly
what produced it.
"""
from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from scipy import stats

from src.data.loader import get_data
from src.data.official_benchmark import OFFICIAL_BENCHMARK, official_benchmark_quantiles
from src.evaluation.backtest import make_rolling_folds
from src.evaluation.metrics import (
    calibration_curve,
    circular_block_bootstrap_ci,
    cluster_bootstrap_ci,
    diebold_mariano_test,
    holm_adjust,
    interval_coverage_at_quantiles,
    pinball_loss,
    pinball_loss_per_observation,
)
from src.evaluation.report_utils import REPORTS_DIR, save_report
from src.models.baselines import BASELINES
from src.models.lightgbm_model import lightgbm_quantiles

BASELINE_CHOICES = list(BASELINES) + ["all"]

DEFAULT_OUTPUT_DIR = "run_comparison"

DEFAULT_STATISTICS = {"hac_max_lag": 168, "hac_max_lag_long": 720, "bootstrap_block_hours": 168,
                      "n_bootstrap": 2000, "seed": 0}

FOLD_GROUP_CHOICES = ["all", "development", "holdout"]

PRIMARY_MODEL = "lightgbm_full"


def fold_group(test_task: int, development_last_test_task: int) -> str:
    return "development" if test_task <= development_last_test_task else "holdout"


def is_oracle_config(config: dict) -> bool:
    """True if the model is given the forecast month's ACTUAL temperature."""
    scenarios = (config["model"].get("temperature_scenarios") or {}).get("enabled", False)
    return config["leakage"]["temperature_source"] == "actual" and not scenarios


def run(config_path: str, feature_set: str = "full", baseline: str = "all", folds_subset: str = "all",
        official_benchmark: bool = False, predictions_out: dict | None = None
        ) -> tuple[pd.DataFrame, dict, list[float], pd.DataFrame]:
    """Returns (results, calibration_curves, quantile_levels, hourly).

    folds_subset: "all", or only the "development" / "holdout" folds (see
        config.backtest.development_last_test_task).
    official_benchmark: also score the GEFCom2014 benchmark files.
    predictions_out: if given, filled with {"y": ..., method: (n_hours,
        n_levels) predictions}, rows in the same order as `hourly`.

    results: one row per (fold, method) with pinball_loss and coverage_90.
    calibration_curves: {method: calibration_curve(...)}, pooling every
        fold's predictions per method before computing the curve, so rare
        quantiles get enough observations to be meaningful.
    hourly: one row per scored test hour, in time order -- timestamp,
        test_task, and one per-hour pinball loss column per method. What
        the significance tests use.
    """
    if folds_subset not in FOLD_GROUP_CHOICES:
        raise ValueError(f"folds_subset must be one of {FOLD_GROUP_CHOICES}, got {folds_subset!r}")
    if baseline not in BASELINE_CHOICES:
        raise ValueError(f"baseline must be one of {BASELINE_CHOICES}, got {baseline!r}")
    baselines_to_run = BASELINES if baseline == "all" else {baseline: BASELINES[baseline]}

    with open(config_path) as f:
        config = yaml.safe_load(f)

    quantile_levels = config["quantiles"]["levels"]
    model_config = config["model"]

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
    dev_last = config["backtest"]["development_last_test_task"]
    if folds_subset != "all":
        folds = [f for f in folds if fold_group(f.test_task, dev_last) == folds_subset]
    model_feature_sets = ["full", "minimal"] if feature_set == "both" else [feature_set]
    method_names = (list(baselines_to_run) + ([OFFICIAL_BENCHMARK] if official_benchmark else [])
                    + [f"lightgbm_{fs}" for fs in model_feature_sets])

    rows = []
    all_y = {name: [] for name in method_names}
    all_preds = {name: [] for name in method_names}
    hour_index = []

    # Folds are consecutive months, so concatenating them in this order
    # gives one continuous hourly series per method.
    for fold in sorted(folds, key=lambda f: f.test_task):
        y_true = fold.test_df["load"].to_numpy()
        valid = ~np.isnan(y_true)
        hour_index.append(pd.DataFrame({"timestamp": fold.test_df["timestamp"].to_numpy()[valid],
                                        "test_task": fold.test_task}))

        for name, baseline_fn in baselines_to_run.items():
            preds = baseline_fn(fold.train_df, fold.test_df, quantile_levels)
            _record(rows, all_y, all_preds, fold.test_task, name, y_true, valid, preds, quantile_levels)

        if official_benchmark:
            preds = official_benchmark_quantiles(config["data"]["raw_load_dir"], fold.test_task,
                                                 fold.test_df["timestamp"], quantile_levels)
            _record(rows, all_y, all_preds, fold.test_task, OFFICIAL_BENCHMARK, y_true, valid, preds,
                    quantile_levels)

        for fs in model_feature_sets:
            preds = lightgbm_quantiles(
                fold.train_df,
                fold.test_df,
                quantile_levels,
                temperature_source=config["leakage"]["temperature_source"],
                model_params=model_config["params"],
                feature_set=fs,
                temperature_scenarios=model_config.get("temperature_scenarios"),
                recency_half_life_years=model_config.get("recency_half_life_years"),
            )
            _record(rows, all_y, all_preds, fold.test_task, f"lightgbm_{fs}", y_true, valid, preds, quantile_levels)

    results = pd.DataFrame(rows)
    results.insert(1, "fold_group", results["test_task"].map(lambda t: fold_group(t, dev_last)))
    curves = {
        name: calibration_curve(np.concatenate(all_y[name]), np.concatenate(all_preds[name], axis=0), quantile_levels)
        for name in method_names
    }
    hourly = pd.concat(hour_index, ignore_index=True)
    for name in method_names:
        hourly[name] = pinball_loss_per_observation(np.concatenate(all_y[name]),
                                                    np.concatenate(all_preds[name], axis=0), quantile_levels)
    if predictions_out is not None:
        predictions_out["y"] = np.concatenate(all_y[method_names[0]])
        predictions_out.update({name: np.concatenate(all_preds[name], axis=0) for name in method_names})
    return results, curves, quantile_levels, hourly


def _record(rows, all_y, all_preds, test_task, method, y_true, valid, preds, quantile_levels):
    loss = pinball_loss(y_true[valid], preds[valid], quantile_levels)
    coverage_90 = interval_coverage_at_quantiles(y_true[valid], preds[valid], quantile_levels)
    rows.append({"test_task": test_task, "method": method, "pinball_loss": loss, "coverage_90": coverage_90})
    all_y[method].append(y_true[valid])
    all_preds[method].append(preds[valid])


def wide_pinball_table(results: pd.DataFrame) -> pd.DataFrame:
    """One row per fold, one column per method -- easy to eyeball."""
    return results.pivot(index="test_task", columns="method", values="pinball_loss")


def summarize(results: pd.DataFrame) -> pd.DataFrame:
    return (
        results.groupby("method")
        .agg(mean_pinball_loss=("pinball_loss", "mean"), std_pinball_loss=("pinball_loss", "std"),
             n_folds=("pinball_loss", "count"), mean_coverage_90=("coverage_90", "mean"))
        .sort_values("mean_pinball_loss")
    )


def default_family(method: str, reference_methods=()) -> str:
    if method in reference_methods:
        return "reference"
    return "baseline" if method in BASELINES or method == OFFICIAL_BENCHMARK else "model_variant"


def significance_vs_model(
    results: pd.DataFrame,
    hourly_losses: dict,
    model_method: str,
    statistics: dict | None = None,
    clusters: np.ndarray | None = None,
    reference_methods=(),
) -> pd.DataFrame:
    """One row per OTHER method: how `model_method` compares to it
    (positive differences / statistics => model better). See the module
    docstring for the tests. `clusters` (the test month of each hour)
    enables the whole-month bootstrap. Methods in `reference_methods`
    (e.g. the oracle) are reported but excluded from every Holm family.
    """
    statistics = {**DEFAULT_STATISTICS, **(statistics or {})}
    wide = wide_pinball_table(results)
    if model_method not in wide.columns:
        raise ValueError(f"{model_method!r} not among methods: {list(wide.columns)}")

    rows = []
    for other in wide.columns:
        if other == model_method:
            continue
        d = hourly_losses[other] - hourly_losses[model_method]
        dm_stat, dm_p = diebold_mariano_test(hourly_losses[other], hourly_losses[model_method],
                                             max_lag=statistics["hac_max_lag"])
        _, dm_p_long = diebold_mariano_test(hourly_losses[other], hourly_losses[model_method],
                                            max_lag=statistics["hac_max_lag_long"])
        block_low, block_high, block_p = circular_block_bootstrap_ci(
            d, block_length=statistics["bootstrap_block_hours"], n_bootstrap=statistics["n_bootstrap"],
            seed=statistics["seed"])
        fold_diff = wide[other] - wide[model_method]
        row = {
            "model": model_method,
            "other": other,
            "family": default_family(other, reference_methods),
            "mean_diff_other_minus_model": float(d.mean()),
            "relative_improvement": float(d.mean() / hourly_losses[other].mean()),
            "block_ci95_low": block_low,
            "block_ci95_high": block_high,
            "block_p_value": block_p,
            "dm_stat": dm_stat,
            "dm_p_value": dm_p,
            "dm_p_value_long_lag": dm_p_long,
            "folds_won_by_model": f"{int((fold_diff > 0).sum())}/{len(fold_diff)}",
            "wilcoxon_p_value": float(stats.wilcoxon(fold_diff).pvalue),
        }
        p_values = [dm_p, dm_p_long, block_p, row["wilcoxon_p_value"]]
        if clusters is not None:
            month_low, month_high, month_p = cluster_bootstrap_ci(
                d, clusters, n_bootstrap=statistics["n_bootstrap"], seed=statistics["seed"])
            row.update({"month_ci95_low": month_low, "month_ci95_high": month_high, "month_p_value": month_p})
            p_values.append(month_p)
        row["p_conservative"] = max(p_values)
        rows.append(row)

    table = pd.DataFrame(rows)
    table["p_conservative_holm"] = np.nan
    for family, idx in table.groupby("family").groups.items():
        if family != "reference":
            table.loc[idx, "p_conservative_holm"] = holm_adjust(table.loc[idx, "p_conservative"])
    return table


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


def plot_calibration(curves: dict, output_path: Path) -> Path:
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


def by_fold_group(results: pd.DataFrame, hourly: pd.DataFrame, model_method: str,
                  statistics: dict | None = None, reference_methods=()) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Summary and significance tables for each fold group present
    (holdout, development, and "all" when both are present), stacked with
    a fold_group column -- so the holdout result can be read on its own,
    untouched by any decision made on the development folds."""
    groups = [g for g in ("holdout", "development") if g in set(results["fold_group"])]
    if len(groups) > 1:
        groups.append("all")
    method_columns = [c for c in hourly.columns if c not in ("timestamp", "test_task")]
    summaries, significances = [], []
    for group in groups:
        sub = results if group == "all" else results[results["fold_group"] == group]
        mask = hourly["test_task"].isin(set(sub["test_task"])).to_numpy()
        sub_hourly = {name: hourly[name].to_numpy()[mask] for name in method_columns}

        summary = summarize(sub)
        summary.insert(0, "fold_group", group)
        summaries.append(summary)
        if sub["test_task"].nunique() >= 2:
            significance = significance_vs_model(sub, sub_hourly, model_method, statistics,
                                                 clusters=hourly["test_task"].to_numpy()[mask],
                                                 reference_methods=reference_methods)
            significance.insert(0, "fold_group", group)
            significances.append(significance)
    significance_table = pd.concat(significances, ignore_index=True) if significances else pd.DataFrame()
    return pd.concat(summaries), significance_table


def git_state() -> dict:
    """Commit and whether the CODE (src/, configs/) differs from it, for
    run_info.json. reports/ is excluded: a run overwrites tracked report
    files itself, which must not count as a modified code base."""
    def git(*args):
        try:
            return subprocess.run(["git", *args], capture_output=True, text=True, check=True).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            return None
    status = git("status", "--porcelain", "--untracked-files=no", "--", "src", "configs")
    return {"commit": git("rev-parse", "HEAD"), "dirty": None if status is None else bool(status)}


def write_run_info(output_dir: str, config_path: str, config: dict, extra: dict | None = None) -> Path:
    path = REPORTS_DIR / output_dir / "run_info.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    info = {"created": datetime.now().isoformat(timespec="seconds"), "config_path": config_path,
            "git": git_state(), **(extra or {}), "config": config}
    path.write_text(json.dumps(info, indent=2, default=str))
    return path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--feature-set", choices=["full", "minimal", "both"], default="full",
                         help="Which LightGBM feature set(s) to include; 'full' is the reported model.")
    parser.add_argument("--baseline", choices=BASELINE_CHOICES, default="all",
                         help="Which baseline(s) to include in the comparison.")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR,
                         help="Subfolder of reports/ to write results and the calibration plot to.")
    parser.add_argument("--folds", choices=FOLD_GROUP_CHOICES, default="all",
                         help="Run all folds, or only the development / holdout folds.")
    parser.add_argument("--official-benchmark", action="store_true",
                         help="Also score the official GEFCom2014 benchmark files (gefcom_benchmark).")
    parser.add_argument("--save-predictions", action="store_true",
                         help="Write all quantile predictions to reports/<output-dir>/predictions.npz.")
    args = parser.parse_args()

    with open(args.config) as f:
        cli_config = yaml.safe_load(f)
    if is_oracle_config(cli_config) and args.output_dir == DEFAULT_OUTPUT_DIR:
        raise SystemExit("This config uses the ACTUAL forecast-month temperature (oracle). "
                         "Pass a separate --output-dir so it cannot overwrite the honest results.")
    statistics = cli_config.get("statistics", {})

    predictions = {} if args.save_predictions else None
    results, curves, quantile_levels, hourly = run(
        args.config, feature_set=args.feature_set, baseline=args.baseline, folds_subset=args.folds,
        official_benchmark=args.official_benchmark, predictions_out=predictions)
    pd.set_option("display.width", 200)

    pinball_table = wide_pinball_table(results)
    print("Per-fold pinball loss (rows = fold, columns = method):")
    print(pinball_table.round(2).to_string())
    print()

    model_methods = [m for m in results["method"].unique() if m not in BASELINES and m != OFFICIAL_BENCHMARK]
    primary_model = PRIMARY_MODEL if PRIMARY_MODEL in model_methods else model_methods[0]
    summary, significance = by_fold_group(results, hourly, primary_model, statistics)
    print("Summary per fold group (best first within each group):")
    print(summary.round(3).to_string())
    print()
    print(f"Significance per fold group: {primary_model} vs each other method "
          f"(positive mean_diff / dm_stat = {primary_model} better):")
    print(significance.round(4).to_string(index=False))
    print()

    calibration = calibration_tables(curves)
    out = args.output_dir
    plot_path = plot_calibration(curves, REPORTS_DIR / out / "calibration.png")
    print(f"Saved calibration plot to {plot_path}")

    save_report(results, f"{out}/results.csv")
    save_report(pinball_table, f"{out}/pinball_by_fold.csv")
    save_report(summary, f"{out}/summary.csv")
    save_report(significance, f"{out}/significance.csv")
    save_report(calibration, f"{out}/calibration_curves.csv")
    # Per-hour losses, so runs with different configs can be tested against
    # each other afterwards (see compare_runs.py).
    save_report(hourly, f"{out}/hourly_losses.csv")
    if predictions is not None:
        # Rows are in the order of hourly_losses.csv, which holds the timestamps.
        np.savez_compressed(REPORTS_DIR / out / "predictions.npz", test_task=hourly["test_task"].to_numpy(),
                            quantile_levels=np.asarray(quantile_levels), **predictions)
    write_run_info(out, args.config, cli_config, {"n_quantiles": len(quantile_levels),
                                                  "feature_set": args.feature_set, "folds": args.folds,
                                                  "official_benchmark": args.official_benchmark})
    print(f"Saved results to reports/{out}/")
