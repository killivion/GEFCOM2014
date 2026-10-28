"""
Merges run_comparison runs over DIFFERENT test months with the SAME
model and quantile grid -- the extended months (--folds extended) and the
task folds -- into one report, with the usual summary and significance
tests per fold group (extended, development, holdout, and "all" = every
merged month).

Checks: no test hour appears in two runs, the runs' model and quantile
settings are identical (run_info.json), and only methods present in
every run are kept (the official benchmark exists for the task folds
only, so it is dropped from a merge with the extended months).

Usage:
    python -m src.evaluation.merge_runs --runs run_extended run_comparison [--output-dir run_all_months]
"""
from __future__ import annotations

import argparse
import json

import pandas as pd

from src.evaluation.report_utils import REPORTS_DIR, save_report
from src.evaluation.run_comparison import PRIMARY_MODEL, by_fold_group, wide_pinball_table, write_run_info


def merge_runs(directories: list[str]) -> tuple[pd.DataFrame, pd.DataFrame, dict, list[str]]:
    """(results, hourly, config of the first run, dropped methods)."""
    loaded = []
    for directory in directories:
        root = REPORTS_DIR / directory
        info = json.loads((root / "run_info.json").read_text())
        loaded.append((directory, pd.read_csv(root / "results.csv"),
                       pd.read_csv(root / "hourly_losses.csv", parse_dates=["timestamp"]), info))

    first_dir, _, _, first_info = loaded[0]
    for directory, _, _, info in loaded[1:]:
        for key in ("model", "quantiles"):
            if info["config"][key] != first_info["config"][key]:
                raise ValueError(f"{directory} and {first_dir} differ in config.{key}")

    method_sets = [set(results["method"]) for _, results, _, _ in loaded]
    common = set.intersection(*method_sets)
    dropped = sorted(set.union(*method_sets) - common)
    results = pd.concat([r[r["method"].isin(common)] for _, r, _, _ in loaded], ignore_index=True)
    columns = ["timestamp", "test_task"] + [c for c in loaded[0][2].columns if c in common]
    hourly = (pd.concat([h[columns] for _, _, h, _ in loaded], ignore_index=True)
              .sort_values("timestamp", kind="stable").reset_index(drop=True))
    if hourly["timestamp"].duplicated().any():
        raise ValueError("the runs share test hours; merge only runs over different months")
    if results.duplicated(["test_task", "method"]).any():
        raise ValueError("the runs share fold numbers")
    return results, hourly, first_info["config"], dropped


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", nargs="+", required=True, help="reports/ subfolders written by run_comparison")
    parser.add_argument("--output-dir", default="run_all_months")
    args = parser.parse_args()

    results, hourly, config, dropped = merge_runs(args.runs)
    if dropped:
        print(f"Not in every run, left out: {dropped}")
    summary, significance = by_fold_group(results, hourly, PRIMARY_MODEL, config.get("statistics", {}))
    pd.set_option("display.width", 200)
    print("Summary per fold group (best first within each group):")
    print(summary.round(3).to_string())
    print()
    print(significance.round(4).to_string(index=False))

    out = args.output_dir
    save_report(results, f"{out}/results.csv")
    save_report(wide_pinball_table(results), f"{out}/pinball_by_fold.csv")
    save_report(summary, f"{out}/summary.csv")
    save_report(significance, f"{out}/significance.csv")
    write_run_info(out, "merged", config, {"runs": args.runs, "dropped_methods": dropped})
    print(f"\nSaved results to reports/{out}/")
