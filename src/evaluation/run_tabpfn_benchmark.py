"""
TabPFN vs. LightGBM: accuracy, calibration and runtime on two tasks, the
same test hours for both models (configs/tabpfn_benchmark.yaml).

- timeseries: the month-ahead forecast on the 14 task folds, reported
  per fold group: development (Oct 2010 - Apr 2011) and holdout
  (May - Nov 2011). LightGBM is the final model exactly as reported
  (all training rows, temperature scenarios, recency weighting). TabPFN
  gets the same features and scenarios and default settings, in two
  variants fixed in advance: "tabpfn" (primary) with every training row
  as context, equally weighted, and "tabpfn_recency" with a recency-
  weighted sample (TabPFN takes no sample weights).
- tabular: interpolation, not forecasting -- the history before the first
  forecast month cut into random week blocks with an embargo
  (backtest.make_week_block_folds); climatology temperature for every row.

Each model gets its best honest setup; neither ever sees the observed
temperature or load of an hour it predicts. Calibration: every forecast
is also recalibrated (conformal, src/models/conformal.py) from OTHER test
data only -- earlier test months (time series) or the other folds
(tabular). All three methods are reported; config.calibration.primary is
fixed in advance and is the one tested for significance. Seasonal
recalibration (time series, after --calibration-history): each month is
calibrated on all earlier months within config.calibration.
seasonal_window_months of its calendar month, including the extended
backtest months 2008 - Sep 2010 (forecast only as calibration history,
never scored); seasonal vs. sequential is judged on the development folds.

Each method's predictions and timings are saved per task
(reports/<output-dir>/<task>/predictions_<method>.npz, not tracked by
git), and the evaluation reads whatever methods are there -- so TabPFN
(API quota) runs once, and the report can be rebuilt without it.

Usage:
    python -m src.evaluation.run_tabpfn_benchmark --task timeseries|tabular [--methods lightgbm tabpfn tabpfn_recency] [--max-folds N] [--calibration-history] [--config configs/tabpfn_benchmark.yaml] [--output-dir tabpfn_benchmark]

--methods with no names only re-evaluates the saved predictions.
--max-folds runs the first N folds only (to check API usage first).
TabPFN needs `pip install -r requirements-tabpfn.txt` and TABPFN_TOKEN.
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.data.loader import get_data
from src.evaluation.backtest import Fold, make_monthly_folds, make_rolling_folds, make_week_block_folds
from src.evaluation.metrics import (calibration_curve, holm_adjust, interval_coverage_at_quantiles, pinball_loss,
                                    pinball_loss_per_observation)
from src.evaluation.report_utils import REPORTS_DIR, save_report
from src.evaluation.run_comparison import (calibration_tables, fold_group, git_state, plot_calibration,
                                           significance_vs_model, summarize, write_run_info)
from src.models.conformal import (CONFORMAL_METHODS, cross_fold_recalibration, seasonal_recalibration,
                                  sequential_recalibration)
from src.models.lightgbm_model import lightgbm_quantiles
from src.models.tabpfn_model import tabpfn_quantiles

TASKS = ("timeseries", "tabular")
METHODS = ("lightgbm", "tabpfn", "tabpfn_recency")
# Methods per task; the recency variant needs a forecast origin (time series only).
TASK_METHODS = {"timeseries": METHODS, "tabular": ("lightgbm", "tabpfn")}
DEFAULT_OUTPUT_DIR = "tabpfn_benchmark"
FOLD_ATTEMPTS = 3  # per fold, for transient API network errors


def load_configs(config_path: str) -> tuple[dict, dict]:
    with open(config_path) as f:
        bench = yaml.safe_load(f)
    with open(bench["base_config"]) as f:
        base = yaml.safe_load(f)
    return bench, base


def make_folds(task: str, data_df: pd.DataFrame, base: dict, bench: dict,
               calibration_history: bool = False) -> list[Fold]:
    """The scored folds of `task`; with calibration_history, instead the
    extended monthly folds before the first task fold (time series only),
    forecast only to serve as calibration history."""
    backtest = base["backtest"]
    first_test = data_df.loc[data_df["task"] == backtest["first_test_task"], "timestamp"].min()
    if calibration_history:
        return make_monthly_folds(data_df, backtest["extended_first_month"], backtest["first_test_task"])
    if task == "tabular":
        t = bench["tabular"]
        return make_week_block_folds(data_df, first_test, t["n_splits"], t["embargo_days"], t["seed"])
    folds = make_rolling_folds(data_df, backtest["first_test_task"], backtest["last_test_task"])
    if bench["timeseries"]["folds"] == "all":
        return folds
    dev_last = backtest["development_last_test_task"]
    return [f for f in folds if fold_group(f.test_task, dev_last) == bench["timeseries"]["folds"]]


def model_setup(task: str, base: dict) -> dict:
    """Inputs shared by both models: the final setup for the forecast,
    climatology temperature without scenarios or recency for the tabular
    task (no forecast origin, and never the hour's observed temperature)."""
    model = base["model"]
    if task == "tabular":
        return {"temperature_source": "climatology", "temperature_scenarios": {"enabled": False},
                "recency_half_life_years": None}
    return {"temperature_source": base["leakage"]["temperature_source"],
            "temperature_scenarios": model.get("temperature_scenarios"),
            "recency_half_life_years": model.get("recency_half_life_years"),
            "temperature_dynamics_features": model.get("temperature_dynamics")}


def predict_fold(method: str, fold: Fold, levels: list[float], setup: dict, base: dict, tabpfn: dict,
                 timings: dict) -> np.ndarray:
    start = time.perf_counter()
    if method == "lightgbm":
        preds = lightgbm_quantiles(fold.train_df, fold.test_df, levels, model_params=base["model"]["params"],
                                   **setup)
    else:
        preds = tabpfn_quantiles(fold.train_df, fold.test_df, levels, tabpfn=tabpfn, timings=timings, **setup)
    timings["total_seconds"] = time.perf_counter() - start
    return preds


def _tabpfn_api_info() -> dict:
    """API limits and usage, best effort (recorded in run_info.json)."""
    info = {}
    try:
        import tabpfn_client
        from tabpfn_client.client import ServiceClient
        tabpfn_client.init()
        info["usage"] = tabpfn_client.get_api_usage()
        limits = ServiceClient.get_model_limits()
        if limits is not None:
            info["default_model_version"] = limits.default_model_version
            info["limits"] = limits.model_limits[limits.default_model_version].model_dump()
    except Exception as exc:  # informational only; the run itself reports real failures
        info["error"] = repr(exc)
    return info


def run_method(task: str, method: str, folds: list[Fold], levels: list[float], base: dict, bench: dict,
               out_dir: Path, prefix: str = "predictions") -> dict:
    """Predicts every fold and saves <prefix>_<method>.npz; returns
    run information for run_info.json."""
    setup = model_setup(task, base)
    tabpfn = {}
    info = {}
    if method != "lightgbm":
        common = {k: v for k, v in bench["tabpfn"].items() if k != "variants"}
        variant = bench["tabpfn"]["variants"][method]
        tabpfn = {**common, "context_rows": variant["context_rows"]}
        if not variant.get("recency", False):
            setup = {**setup, "recency_half_life_years": None}
        info["api_before"] = _tabpfn_api_info()
        limits = info["api_before"].get("limits", {})
        if "train_set_max_rows" in limits:
            tabpfn["context_rows"] = min(tabpfn["context_rows"] or limits["train_set_max_rows"],
                                         limits["train_set_max_rows"])
        if "test_set_max_rows" in limits:
            tabpfn["predict_chunk_rows"] = min(tabpfn["predict_chunk_rows"], limits["test_set_max_rows"])
        info["tabpfn_settings_used"] = tabpfn
        info["recency_half_life_years"] = setup["recency_half_life_years"]

    y, preds, test_task, timestamps, per_fold, model_files = [], [], [], [], [], set()
    for fold in sorted(folds, key=lambda f: f.test_task):
        for attempt in range(1, FOLD_ATTEMPTS + 1):
            timings = {}
            try:
                p = predict_fold(method, fold, levels, setup, base, tabpfn, timings)
                break
            except Exception as exc:
                # Only network failures of the API client are retried (the
                # prediction is deterministic: fixed seeds, same context).
                if not type(exc).__module__.startswith(("httpx", "httpcore")) or attempt == FOLD_ATTEMPTS:
                    raise
                print(f"[{task}/{method}] fold {fold.test_task}: {type(exc).__name__} ({exc}); "
                      f"retry {attempt}/{FOLD_ATTEMPTS - 1} in {30 * attempt} s", flush=True)
                time.sleep(30 * attempt)
        y_fold = fold.test_df["load"].to_numpy()
        valid = ~np.isnan(y_fold)
        y.append(y_fold[valid])
        preds.append(p[valid])
        test_task.append(np.full(valid.sum(), fold.test_task))
        timestamps.append(fold.test_df["timestamp"].to_numpy()[valid].astype("datetime64[ns]").astype(np.int64))
        model_files.add(timings.pop("model_file", None))
        per_fold.append({"test_task": fold.test_task, "n_train_rows": int(fold.train_df["load"].notna().sum()), **timings})
        print(f"[{task}/{method}] fold {fold.test_task}: {timings['total_seconds']:.1f} s, "
              f"pinball {pinball_loss(y_fold[valid], p[valid], levels):.3f}", flush=True)

    out_dir.mkdir(parents=True, exist_ok=True)
    timing = pd.DataFrame(per_fold)
    np.savez_compressed(out_dir / f"{prefix}_{method}.npz", y=np.concatenate(y), preds=np.concatenate(preds),
                        test_task=np.concatenate(test_task), timestamp=np.concatenate(timestamps),
                        quantile_levels=np.asarray(levels),
                        **{f"timing_{c}": timing[c].to_numpy(dtype=float) for c in timing.columns})
    if method != "lightgbm":
        info["model_files"] = sorted(f for f in model_files if f)
        if len(info["model_files"]) > 1:
            print(f"WARNING: {method} used several model files across folds: {info['model_files']}")
        info["api_after"] = _tabpfn_api_info()
        versions = {info[k].get("default_model_version") for k in ("api_before", "api_after")}
        if len(versions - {None}) > 1:
            print(f"WARNING: the API's default TabPFN version changed during the run: {versions}")
    return info


def load_predictions(out_dir: Path) -> tuple[dict, pd.DataFrame]:
    """{method: npz} for every saved method, restricted to the folds all of
    them share and sorted by time; raises if the test hours differ."""
    loaded = {m: dict(np.load(out_dir / f"predictions_{m}.npz")) for m in METHODS
              if (out_dir / f"predictions_{m}.npz").exists()}
    if not loaded:
        raise SystemExit(f"no predictions_<method>.npz in {out_dir}")
    common = set.intersection(*(set(np.unique(z["test_task"])) for z in loaded.values()))
    out, runtime = {}, []
    for m, z in loaded.items():
        keep = np.isin(z["test_task"], list(common))
        order = np.argsort(z["timestamp"][keep], kind="stable")
        out[m] = {k: z[k][keep][order] for k in ("y", "preds", "test_task", "timestamp")}
        out[m]["quantile_levels"] = z["quantile_levels"]
        t = pd.DataFrame({k[len("timing_"):]: v for k, v in z.items() if k.startswith("timing_")})
        t.insert(0, "method", m)
        t["test_task"] = t["test_task"].astype(int)
        runtime.append(t[t["test_task"].isin(common)])
    first = next(iter(out.values()))
    for m, z in out.items():
        if not (np.array_equal(z["timestamp"], first["timestamp"]) and np.array_equal(z["y"], first["y"])):
            raise ValueError(f"{m} was scored on different test hours than {next(iter(out))}")
    if any(len(np.unique(z["test_task"])) < len(np.unique(loaded[m]["test_task"])) for m, z in out.items()):
        print(f"Note: evaluating only the {len(common)} folds every method has.")
    return out, pd.concat(runtime, ignore_index=True)


def recalibrate(task: str, y, preds, test_task, levels, method: str) -> np.ndarray:
    if task == "timeseries":
        return sequential_recalibration(y, preds, test_task, levels, method)
    return cross_fold_recalibration(y, preds, test_task, levels, method)


def operating_month(timestamps_ns: np.ndarray) -> np.ndarray:
    return (pd.to_datetime(timestamps_ns) - pd.Timedelta(hours=1)).month.to_numpy()


def seasonal_variants(out_dir: Path, method: str, z: dict, levels: list[float], window: int) -> dict:
    """{"<method>+seasonal_<conformal>": predictions} when the method's
    calibration history (calibration_<method>.npz, the extended months)
    exists: each scored month recalibrated from all earlier months --
    history and earlier scored months -- within `window` calendar months
    of its own. Empty without the history."""
    path = out_dir / f"calibration_{method}.npz"
    if not path.exists():
        return {}
    info_path = out_dir / "run_info.json"
    if method != "lightgbm" and info_path.exists():
        # The history must come from the same model as the scored forecasts.
        methods = json.loads(info_path.read_text()).get("methods", {})
        scored = methods.get(method, {}).get("model_files")
        history = methods.get(f"{method}_calibration_history", {}).get("model_files")
        if scored and history and scored != history:
            print(f"WARNING: {method}: calibration history from {history}, scored forecasts from {scored}")
    h = np.load(path)
    y_all = np.concatenate([h["y"], z["y"]])
    preds_all = np.concatenate([h["preds"], z["preds"]])
    task_all = np.concatenate([h["test_task"], z["test_task"]])
    month_all = operating_month(np.concatenate([h["timestamp"], z["timestamp"]]))
    scored = slice(len(h["y"]), None)
    return {f"{method}+seasonal_{cm}": seasonal_recalibration(
        y_all, preds_all, task_all, month_all, levels, cm, np.unique(z["test_task"]), window)[scored]
        for cm in CONFORMAL_METHODS}


def evaluate(task: str, out_dir: Path, base: dict, bench: dict) -> None:
    data, runtime = load_predictions(out_dir)
    first = next(iter(data.values()))
    levels = [float(q) for q in first["quantile_levels"]]
    y, test_task, timestamps = first["y"], first["test_task"], first["timestamp"]

    variants = {}  # name -> predictions; "<method>" raw, "<method>+<conformal>" recalibrated
    for m, z in data.items():
        variants[m] = z["preds"]
        for cm in CONFORMAL_METHODS:
            variants[f"{m}+{cm}"] = recalibrate(task, y, z["preds"], test_task, levels, cm)
        if task == "timeseries":
            variants.update(seasonal_variants(out_dir, m, z, levels, bench["calibration"]["seasonal_window_months"]))

    rows = []
    for name, p in variants.items():
        for t in np.unique(test_task):
            r = test_task == t
            rows.append({"test_task": int(t), "method": name, "pinball_loss": pinball_loss(y[r], p[r], levels),
                         "coverage_90": interval_coverage_at_quantiles(y[r], p[r], levels)})
    results = pd.DataFrame(rows)
    curves = {name: calibration_curve(y, p, levels) for name, p in variants.items()}

    # Reported per fold group (development / holdout / all, like
    # run_comparison), so the holdout can be read on its own.
    dev_last = base["backtest"]["development_last_test_task"]
    group_of = {int(t): fold_group(int(t), dev_last) if task == "timeseries" else "tabular"
                for t in np.unique(test_task)}
    results.insert(1, "fold_group", results["test_task"].map(group_of))
    groups = [g for g in ("development", "holdout", "tabular") if g in group_of.values()]
    if len(groups) > 1:
        groups.append("all")

    hourly = {name: pinball_loss_per_observation(y, p, levels) for name, p in variants.items()}
    if task == "timeseries":
        clusters = test_task  # whole test months
    else:
        day = pd.to_datetime(timestamps) - pd.Timedelta(hours=1)
        clusters = (day.normalize() - pd.to_timedelta(day.dayofweek, unit="D")).to_numpy()  # whole weeks
    primary = bench["calibration"]["primary"]
    tabpfn_methods = [m for m in data if m != "lightgbm"]
    summaries, significance = [], []
    for group in groups:
        sub = results if group == "all" else results[results["fold_group"] == group]
        rows_in = np.isin(test_task, sub["test_task"].unique())
        summary = summarize(sub)
        summary.insert(0, "fold_group", group)
        summary["mean_abs_calibration_gap"] = [
            float(np.mean(np.abs(np.subtract(calibration_curve(y[rows_in], variants[n][rows_in], levels)["empirical"],
                                             levels)))) for n in summary.index]
        summaries.append(summary)
        # Significance: each TabPFN variant vs LightGBM, raw and after the
        # primary recalibration.
        suffixes = ("", f"+{primary}", f"+seasonal_{primary}")
        for model, other in [(f"{m}{s}", f"lightgbm{s}") for m in tabpfn_methods for s in suffixes]:
            if model in variants and other in variants and sub["test_task"].nunique() >= 2:
                pair = sub[sub["method"].isin([model, other])]
                table = significance_vs_model(pair, {k: hourly[k][rows_in] for k in (model, other)}, model,
                                              base.get("statistics"), clusters=clusters[rows_in])
                table.insert(0, "fold_group", group)
                significance.append(table)
    summary = pd.concat(summaries)
    significance = pd.concat(significance, ignore_index=True) if significance else pd.DataFrame()
    if not significance.empty:
        # Holm across every TabPFN-vs-LightGBM comparison of a fold group
        # (variants x calibrations), not per single comparison.
        for _, idx in significance.groupby("fold_group").groups.items():
            significance.loc[idx, "p_conservative_holm"] = holm_adjust(significance.loc[idx, "p_conservative"])

    # Median, not mean: wall-clock time also counts a machine that slept mid-fold.
    runtime_summary = runtime.groupby("method").median(numeric_only=True).drop(columns="test_task")
    runtime_summary.insert(0, "n_folds", runtime.groupby("method").size())

    pd.set_option("display.width", 200)
    print(f"\n{task}: summary per fold group (best first)")
    print(summary.round(4).to_string())
    print("\nruntime per fold (seconds, median over folds; see runtime_by_fold.csv)")
    print(runtime_summary.round(1).to_string())
    if not significance.empty:
        print("\nsignificance (model = TabPFN variant, positive mean_diff = TabPFN better; 'month_' columns = cluster bootstrap over "
              f"{'test months' if task == 'timeseries' else 'test weeks'})")
        print(significance.round(4).to_string(index=False))

    rel = out_dir.relative_to(REPORTS_DIR) if out_dir.is_relative_to(REPORTS_DIR) else out_dir
    save_report(results, f"{rel}/results.csv")
    save_report(summary, f"{rel}/summary.csv")
    save_report(runtime, f"{rel}/runtime_by_fold.csv")
    save_report(runtime_summary, f"{rel}/runtime.csv")
    save_report(calibration_tables(curves), f"{rel}/calibration_curves.csv")
    if not significance.empty:
        save_report(significance, f"{rel}/significance.csv")
    shown = [n for n in variants if "+" not in n or n.endswith(f"+{primary}")]  # incl. +seasonal_<primary>
    subtitle = ("pooled across the test months" if task == "timeseries"
                else "pooled across the week-block folds; interpolation, not forecasting")
    plot_calibration({n: curves[n] for n in shown}, out_dir / "calibration.png", subtitle)
    print(f"\nSaved to {out_dir}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=TASKS, required=True)
    parser.add_argument("--methods", nargs="*", choices=METHODS, default=None,
                        help="Methods to (re)run (default: all of the task's); none = only evaluate the saved "
                             "predictions.")
    parser.add_argument("--max-folds", type=int, default=None, help="Run only the first N folds.")
    parser.add_argument("--calibration-history", action="store_true",
                        help="Time series: forecast the extended months (calibration history for seasonal "
                             "recalibration, never scored) instead of the scored folds.")
    parser.add_argument("--config", default="configs/tabpfn_benchmark.yaml")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="Subfolder of reports/.")
    args = parser.parse_args(argv)

    bench, base = load_configs(args.config)
    out_dir = REPORTS_DIR / args.output_dir / args.task
    if args.methods is None:
        args.methods = list(TASK_METHODS[args.task])
    unsupported = set(args.methods) - set(TASK_METHODS[args.task])
    if unsupported:
        raise SystemExit(f"{sorted(unsupported)} not available for --task {args.task}")
    if args.calibration_history and args.task != "timeseries":
        raise SystemExit("--calibration-history is for --task timeseries only")
    prefix, key_suffix = ("calibration", "_calibration_history") if args.calibration_history else ("predictions", "")
    if args.methods:
        data = get_data(base["data"]["raw_load_dir"], n_tasks=base["data"]["n_tasks"],
                        processed_path=base["data"]["processed_path"])
        folds = sorted(make_folds(args.task, data.df, base, bench, args.calibration_history),
                       key=lambda f: f.test_task)[:args.max_folds]
        levels = base["quantiles"]["levels"]
        info_path = out_dir / "run_info.json"
        rel = out_dir.relative_to(REPORTS_DIR) if out_dir.is_relative_to(REPORTS_DIR) else out_dir
        for method in args.methods:
            info = run_method(args.task, method, folds, levels, base, bench, out_dir, prefix)
            # One entry per method, kept across runs (TabPFN may run later than
            # LightGBM); re-read just before writing, after every method.
            methods_info = json.loads(info_path.read_text()).get("methods", {}) if info_path.exists() else {}
            methods_info[method + key_suffix] = {"created": datetime.now().isoformat(timespec="seconds"), "git": git_state(),
                                    "max_folds": args.max_folds, **info}
            write_run_info(f"{rel}", bench["base_config"], base,
                           {"task": args.task, "benchmark_config": bench, "methods": methods_info})
    if args.calibration_history and not any(out_dir.glob("predictions_*.npz")):
        print(f"Saved the calibration history to {out_dir}; nothing scored yet to evaluate.")
        return
    evaluate(args.task, out_dir, base, bench)


if __name__ == "__main__":
    main()
