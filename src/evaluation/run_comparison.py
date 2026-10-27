"""
Runs the baselines and the LightGBM model across every rolling-origin
backtest fold ONCE and produces the full comparison from that single
pass: a per-fold pinball-loss table (one column per method), a summary
table (mean/std/n_folds, best first), a mean-coverage table, the
significance tests of the primary model against every other method, and
a calibration reliability diagram (nominal vs. empirical, pooled across
all folds).

Significance testing (see significance_vs_model): the unit of
comparison is the per-HOUR pinball loss over the whole backtest (all
test months concatenated in time order), not 14 fold averages:
- Diebold-Mariano test with a Newey-West variance over
  config.statistics.hac_max_lag lags, since hourly loss differentials are
  strongly autocorrelated;
- moving-block bootstrap 95% CI of the mean loss difference (blocks of
  config.statistics.bootstrap_block_hours), as an assumption-light check
  of the same question;
- Holm-adjusted DM p-values, since the model is tested against several
  methods at once;
- per fold: how many folds the model wins, and a Wilcoxon signed-rank
  test on the fold losses (rank-based, so the two outlier months can't
  dominate it the way they would a paired t-test).

Usage:
    python -m src.evaluation.run_comparison [--config configs/default.yaml] [--feature-set full|minimal|both] [--baseline <name>|all] [--output-dir run_comparison]

--feature-set controls which LightGBM variant(s) are included; --baseline
controls which baseline(s) are included (default "all"). --output-dir is
the subfolder of reports/ to write to, so e.g. the oracle-temperature run
(configs/oracle_temperature.yaml) doesn't overwrite the honest one.

Optional post-processing (config section `postprocess`, default OFF; see
src/models/postprocess.py): when `postprocess.enabled` is true, each
LightGBM feature set additionally yields the methods
lightgbm_<fs>_calibrated (additive per-quantile correction),
lightgbm_<fs>_recalmap (nominal->empirical recalibration map),
blend_<fs> (quantile-averaging blend with seasonal_climatology) and
blend_<fs>_calibrated, all fitted on inner-validation months carved out of
the fold's own training data. With it off, the methods and their numbers
are exactly as before.
"""
from __future__ import annotations

import argparse
import time
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
from src.evaluation.metrics import (
    calibration_curve,
    diebold_mariano_test,
    holm_adjust,
    interval_coverage_at_quantiles,
    moving_block_bootstrap_ci,
    pinball_loss,
    pinball_loss_per_observation,
)
from src.evaluation.report_utils import REPORTS_DIR, save_report
from src.models.baselines import BASELINES, PRIMARY_BASELINE
from src.models.lightgbm_model import lightgbm_quantiles
from src.models.postprocess import POSTPROCESS_METHODS, postprocessed_forecasts

BASELINE_CHOICES = list(BASELINES) + ["all"]

DEFAULT_OUTPUT_DIR = "run_comparison"

DEFAULT_STATISTICS = {"hac_max_lag": 168, "bootstrap_block_hours": 168, "n_bootstrap": 2000, "seed": 0}

FOLD_GROUP_CHOICES = ["all", "development", "holdout"]

DEFAULT_POSTPROCESS = {
    "enabled": False,
    "inner_validation": "recent",
    "inner_validation_months": 3,
    "methods": list(POSTPROCESS_METHODS),
    "baseline": "seasonal_climatology",
}


def postprocess_method_name(feature_set: str, method: str) -> str:
    """Reported name of a post-processed method, e.g. lightgbm_full_calibrated
    or blend_full."""
    if method.startswith("blend"):
        return method.replace("blend", f"blend_{feature_set}", 1)
    return f"lightgbm_{feature_set}_{method}"


def postprocess_settings(config: dict) -> dict:
    settings = {**DEFAULT_POSTPROCESS, **(config.get("postprocess") or {})}
    unknown = set(settings["methods"]) - set(POSTPROCESS_METHODS)
    if unknown:
        raise ValueError(f"unknown postprocess methods {sorted(unknown)}; choose from {POSTPROCESS_METHODS}")
    return settings


def fold_group(test_task: int, development_last_test_task: int) -> str:
    return "development" if test_task <= development_last_test_task else "holdout"


def run(config_path: str, feature_set: str = "full", baseline: str = "all", folds_subset: str = "all"
        ) -> tuple[pd.DataFrame, dict, list[float], dict, np.ndarray]:
    """Returns (results, calibration_curves, quantile_levels, hourly_losses, hourly_tasks).

    folds_subset: "all", or only the "development" / "holdout" folds (see
        config.backtest.development_last_test_task).

    results: one row per (fold, method) with pinball_loss and coverage_90.
    calibration_curves: {method: calibration_curve(...)}, pooling every
        fold's predictions per method before computing the curve, so rare
        quantiles get enough observations to be meaningful.
    hourly_losses: {method: per-hour pinball loss over every test hour of
        every fold, in time order} -- what the significance tests use.
    hourly_tasks: the test_task each of those hours belongs to.
    """
    if folds_subset not in FOLD_GROUP_CHOICES:
        raise ValueError(f"folds_subset must be one of {FOLD_GROUP_CHOICES}, got {folds_subset!r}")
    if baseline not in BASELINE_CHOICES:
        raise ValueError(f"baseline must be one of {BASELINE_CHOICES}, got {baseline!r}")
    baselines_to_run = BASELINES if baseline == "all" else {baseline: BASELINES[baseline]}

    with open(config_path) as f:
        config = yaml.safe_load(f)

    quantile_levels = config["quantiles"]["levels"]
    temperature_source = config["leakage"]["temperature_source"]
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
    dev_last = config["backtest"]["development_last_test_task"]
    if folds_subset != "all":
        folds = [f for f in folds if fold_group(f.test_task, dev_last) == folds_subset]
    model_feature_sets = ["full", "minimal"] if feature_set == "both" else [feature_set]
    method_names = list(baselines_to_run) + [f"lightgbm_{fs}" for fs in model_feature_sets]
    postprocess = postprocess_settings(config)
    if postprocess["enabled"]:
        method_names += [postprocess_method_name(fs, m) for fs in model_feature_sets for m in postprocess["methods"]]

    rows = []
    all_y = {name: [] for name in method_names}
    all_preds = {name: [] for name in method_names}

    # Folds are consecutive months, so concatenating them in this order
    # gives one continuous hourly series per method.
    for fold in sorted(folds, key=lambda f: f.test_task):
        y_true = fold.test_df["load"].to_numpy()
        valid = ~np.isnan(y_true)

        for name, baseline_fn in baselines_to_run.items():
            preds = baseline_fn(fold.train_df, fold.test_df, quantile_levels)
            _record(rows, all_y, all_preds, fold.test_task, name, y_true, valid, preds, quantile_levels)

        for fs in model_feature_sets:
            def model_fn(train_df, test_df, fs=fs):
                return lightgbm_quantiles(train_df, test_df, quantile_levels, temperature_source=temperature_source,
                                          model_params=model_params, feature_set=fs)

            if not postprocess["enabled"]:
                preds = model_fn(fold.train_df, fold.test_df)
                _record(rows, all_y, all_preds, fold.test_task, f"lightgbm_{fs}", y_true, valid, preds,
                        quantile_levels)
                continue

            started = time.perf_counter()
            pp = postprocessed_forecasts(
                fold.train_df, fold.test_df, quantile_levels, model_fn,
                lambda tr, te: BASELINES[postprocess["baseline"]](tr, te, quantile_levels),
                n_months=postprocess["inner_validation_months"], methods=postprocess["methods"],
                scheme=postprocess["inner_validation"],
            )
            _record(rows, all_y, all_preds, fold.test_task, f"lightgbm_{fs}", y_true, valid,
                    pp.test_predictions["model"], quantile_levels)
            for m in postprocess["methods"]:
                _record(rows, all_y, all_preds, fold.test_task, postprocess_method_name(fs, m), y_true, valid,
                        pp.test_predictions[m], quantile_levels)
            _report_postprocess_fold(fold.test_task, fs, pp, quantile_levels, time.perf_counter() - started)

    results = pd.DataFrame(rows)
    results.insert(1, "fold_group", results["test_task"].map(lambda t: fold_group(t, dev_last)))
    first_method = method_names[0]
    hourly_tasks = np.concatenate([np.full(len(y), task) for task, y in zip(
        sorted(f.test_task for f in folds), all_y[first_method])])
    curves = {
        name: calibration_curve(np.concatenate(all_y[name]), np.concatenate(all_preds[name], axis=0), quantile_levels)
        for name in method_names
    }
    hourly_losses = {
        name: pinball_loss_per_observation(np.concatenate(all_y[name]), np.concatenate(all_preds[name], axis=0),
                                           quantile_levels)
        for name in method_names
    }
    return results, curves, quantile_levels, hourly_losses, hourly_tasks


def _report_postprocess_fold(test_task, feature_set, pp, quantile_levels, seconds):
    parts = [f"[postprocess] fold {test_task} lightgbm_{feature_set}: {seconds:.0f}s"]
    if pp.blend_weight is not None:
        parts.append(f"blend w={pp.blend_weight:.1f}")
    levels = list(quantile_levels)
    if 0.05 in levels and 0.95 in levels:
        lo, hi = levels.index(0.05), levels.index(0.95)
        if pp.additive_shifts is not None:
            parts.append(f"shift q05={pp.additive_shifts[lo]:+.1f} q95={pp.additive_shifts[hi]:+.1f}")
        if pp.mapped_levels is not None:
            parts.append(f"map 0.05->{pp.mapped_levels[lo]:.3f} 0.95->{pp.mapped_levels[hi]:.3f}")
    print(", ".join(parts), flush=True)


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
        results.groupby("method")["pinball_loss"]
        .agg(["mean", "std", "count"])
        .rename(columns={"mean": "mean_pinball_loss", "std": "std_pinball_loss", "count": "n_folds"})
        .sort_values("mean_pinball_loss")
    )


def coverage_summary(results: pd.DataFrame) -> pd.Series:
    return results.groupby("method")["coverage_90"].mean().rename("mean_coverage_90").sort_index()


def significance_vs_model(
    results: pd.DataFrame, hourly_losses: dict, model_method: str, statistics: dict | None = None
) -> pd.DataFrame:
    """One row per OTHER method: how `model_method` compares to it.

    Hourly level (see module docstring): mean loss difference (other minus
    model, positive => model better), its moving-block bootstrap 95% CI,
    the Newey-West Diebold-Mariano statistic and p-value, and the
    Holm-adjusted p-value across all rows of this table. Fold level:
    folds won by the model, and the Wilcoxon signed-rank p-value.
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
        ci_low, ci_high = moving_block_bootstrap_ci(d, block_length=statistics["bootstrap_block_hours"],
                                                    n_bootstrap=statistics["n_bootstrap"], seed=statistics["seed"])
        fold_diff = wide[other] - wide[model_method]
        rows.append({
            "model": model_method,
            "other": other,
            "mean_diff_other_minus_model": float(d.mean()),
            "bootstrap_ci95_low": ci_low,
            "bootstrap_ci95_high": ci_high,
            "dm_stat": dm_stat,
            "dm_p_value": dm_p,
            "folds_won_by_model": f"{int((fold_diff > 0).sum())}/{len(fold_diff)}",
            "wilcoxon_p_value": float(stats.wilcoxon(fold_diff).pvalue),
        })
    table = pd.DataFrame(rows)
    table.insert(table.columns.get_loc("dm_p_value") + 1, "dm_p_value_holm", holm_adjust(table["dm_p_value"]))
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


def by_fold_group(results: pd.DataFrame, hourly_losses: dict, hourly_tasks: np.ndarray, model_method: str,
                  statistics: dict | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Summary and significance tables for each fold group present
    (development, holdout) plus "all", stacked with a fold_group column --
    so the holdout result can be read on its own, untouched by any
    decision made on the development folds."""
    groups = [g for g in ("development", "holdout") if g in set(results["fold_group"])]
    if len(groups) > 1:
        groups = ["all"] + groups
    summaries, significances = [], []
    for group in groups:
        sub = results if group == "all" else results[results["fold_group"] == group]
        tasks = set(sub["test_task"])
        mask = np.isin(hourly_tasks, list(tasks))
        sub_hourly = {name: losses[mask] for name, losses in hourly_losses.items()}

        summary = summarize(sub).join(sub.groupby("method")["coverage_90"].mean().rename("mean_coverage_90"))
        summary.insert(0, "fold_group", group)
        summaries.append(summary)
        if len(tasks) >= 2:
            significance = significance_vs_model(sub, sub_hourly, model_method, statistics)
            significance.insert(0, "fold_group", group)
            significances.append(significance)
    return pd.concat(summaries), pd.concat(significances, ignore_index=True)


def primary_model_name(methods) -> str:
    return "lightgbm_full" if "lightgbm_full" in methods else "lightgbm_minimal"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--feature-set", choices=["full", "minimal", "both"], default="both",
                         help="Which LightGBM feature set(s) to include in the comparison.")
    parser.add_argument("--baseline", choices=BASELINE_CHOICES, default="all",
                         help="Which baseline(s) to include in the comparison.")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR,
                         help="Subfolder of reports/ to write results and the calibration plot to.")
    parser.add_argument("--folds", choices=FOLD_GROUP_CHOICES, default="all",
                         help="Run all folds, or only the development / holdout folds.")
    args = parser.parse_args()

    with open(args.config) as f:
        cli_config = yaml.safe_load(f)
    statistics = cli_config.get("statistics", {})

    results, curves, quantile_levels, hourly_losses, hourly_tasks = run(
        args.config, feature_set=args.feature_set, baseline=args.baseline, folds_subset=args.folds)
    pd.set_option("display.width", 160)

    pinball_table = wide_pinball_table(results)
    print("Per-fold pinball loss (rows = fold, columns = method):")
    print(pinball_table.round(2).to_string())
    print()

    primary_model = primary_model_name(results["method"].unique())
    summary, significance = by_fold_group(results, hourly_losses, hourly_tasks, primary_model, statistics)
    print("Summary per fold group (best first within each group):")
    print(summary.round(3).to_string())
    print()

    coverage = coverage_summary(results)

    print(f"Significance per fold group: {primary_model} vs each other method "
          f"(positive mean_diff / dm_stat = {primary_model} better):")
    print(significance.round(4).to_string(index=False))
    print()

    pp_significance = None
    if postprocess_settings(cli_config)["enabled"] and PRIMARY_BASELINE in set(results["method"]):
        # Test every model method (raw and post-processed) against the primary baseline.
        model_methods = [m for m in results["method"].unique() if m not in BASELINES]
        pp_rows = []
        for method in model_methods:
            _, sig = by_fold_group(results, hourly_losses, hourly_tasks, method, statistics)
            pp_rows.append(sig[sig["other"] == PRIMARY_BASELINE])
        pp_significance = pd.concat(pp_rows, ignore_index=True)
        print(f"Significance of each model method vs {PRIMARY_BASELINE} (positive mean_diff = model better):")
        print(pp_significance.round(4).to_string(index=False))
        print()

    calibration = calibration_tables(curves)
    print("Calibration (nominal -> empirical), per method:")
    for name in curves:
        print(f"\n{name}:")
        print(calibration[calibration["method"] == name].drop(columns="method").round(3).to_string(index=False))

    out = args.output_dir
    plot_path = plot_calibration(curves, REPORTS_DIR / out / "calibration.png")
    print(f"\nSaved calibration plot to {plot_path}")

    save_report(results, f"{out}/results.csv")
    save_report(pinball_table, f"{out}/pinball_by_fold.csv")
    save_report(summary, f"{out}/summary.csv")
    save_report(coverage, f"{out}/coverage.csv")
    save_report(significance, f"{out}/significance.csv")
    save_report(calibration, f"{out}/calibration_curves.csv")
    if pp_significance is not None:
        save_report(pp_significance, f"{out}/significance_postprocess.csv")
    print(f"Saved results to reports/{out}/")
