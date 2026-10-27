"""
Compares the LightGBM model of several run_comparison.py runs with
different configs -- the ablation steps and the oracle-temperature upper
bound against the final model -- using the per-hour losses each run saved
(reports/<run>/hourly_losses.csv) and the same significance tests as
run_comparison.py, per fold group.

The runs must cover exactly the same test hours (same timestamps) on the
same quantile grid, so the loss series are paired hour by hour; both are
checked (timestamps from hourly_losses.csv, grid size from run_info.json).

Usage:
    python -m src.evaluation.compare_runs --primary final=run_comparison \\
        --runs plain=ablation_plain recency_only=ablation_recency_only \\
               scenarios_only=ablation_scenarios_only oracle=run_comparison_oracle \\
        --reference oracle [--model-column lightgbm_full] [--config configs/default.yaml] [--output-dir run_ablation]

Each NAME=DIR is a reports/ subfolder written by run_comparison.py; the
baselines are taken from the primary run. Runs named in --reference
(the oracle) are reported but kept out of every Holm family.
"""
from __future__ import annotations

import argparse
import json

import pandas as pd
import yaml

from src.evaluation.report_utils import REPORTS_DIR, save_report
from src.evaluation.run_comparison import by_fold_group, fold_group, write_run_info


def _load(directory: str) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    root = REPORTS_DIR / directory
    hourly = pd.read_csv(root / "hourly_losses.csv", parse_dates=["timestamp"])
    results = pd.read_csv(root / "results.csv")
    info = json.loads((root / "run_info.json").read_text())
    return hourly, results, info


def _named(spec: str) -> tuple[str, str]:
    name, sep, directory = spec.partition("=")
    if not sep or not name or not directory:
        raise argparse.ArgumentTypeError(f"expected NAME=DIR, got {spec!r}")
    return name, directory


def combine_runs(primary: tuple[str, str], runs: list[tuple[str, str]], model_column: str
                 ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(results, hourly) with the primary run's baselines plus every run's
    `model_column` renamed to that run's name."""
    primary_name, primary_dir = primary
    hourly, results, info = _load(primary_dir)
    hourly = hourly.rename(columns={model_column: primary_name})
    results = results.replace({"method": {model_column: primary_name}})

    for name, directory in runs:
        other_hourly, other_results, other_info = _load(directory)
        if not other_hourly["timestamp"].equals(hourly["timestamp"]):
            raise ValueError(f"run {name!r} ({directory}) does not cover the same test hours as the primary run")
        if other_info.get("n_quantiles") != info.get("n_quantiles"):
            raise ValueError(f"run {name!r} ({directory}) used a different quantile grid "
                             f"({other_info.get('n_quantiles')} vs {info.get('n_quantiles')} levels)")
        hourly[name] = other_hourly[model_column].to_numpy()
        model_rows = other_results[other_results["method"] == model_column].assign(method=name)
        results = pd.concat([results, model_rows], ignore_index=True)
    return results, hourly


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--primary", type=_named, required=True, help="NAME=DIR of the run every other one is tested against")
    parser.add_argument("--runs", type=_named, nargs="+", required=True, help="NAME=DIR of each run to compare")
    parser.add_argument("--reference", nargs="*", default=[], help="Run names reported for reference only (no Holm)")
    parser.add_argument("--model-column", default="lightgbm_full", help="Model method name inside each run")
    parser.add_argument("--config", default="configs/default.yaml", help="For the statistics settings and fold split")
    parser.add_argument("--output-dir", default="run_ablation")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)
    results, hourly = combine_runs(args.primary, args.runs, args.model_column)
    dev_last = config["backtest"]["development_last_test_task"]
    results["fold_group"] = results["test_task"].map(lambda t: fold_group(t, dev_last))

    summary, significance = by_fold_group(results, hourly, args.primary[0], config.get("statistics", {}),
                                          reference_methods=args.reference)
    pd.set_option("display.width", 200)
    print("Summary per fold group (best first within each group):")
    print(summary.round(3).to_string())
    print()
    print(f"Significance per fold group: {args.primary[0]} vs each other run/method "
          f"(positive mean_diff / dm_stat = {args.primary[0]} better):")
    print(significance.round(4).to_string(index=False))

    save_report(summary, f"{args.output_dir}/summary.csv")
    save_report(significance, f"{args.output_dir}/significance.csv")
    write_run_info(args.output_dir, args.config, config,
                   {"primary": dict([args.primary]), "runs": dict(args.runs), "reference": args.reference})
    print(f"\nSaved results to reports/{args.output_dir}/")
