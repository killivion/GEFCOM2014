"""
Verifies the extended monthly folds (backtest.make_monthly_folds): each
month is trained only on rows strictly before it, months follow the
hour-ending convention of the task files, the numbering continues
backwards from the first task fold, and a monthly fold equals the task
fold for the same month. Also checks merge_runs refuses overlapping runs
and drops methods missing from a run.
"""
import json

import numpy as np
import pandas as pd
import pytest

import src.evaluation.merge_runs as mr
from src.evaluation.backtest import make_monthly_folds, make_rolling_folds
from src.evaluation.run_comparison import fold_group


def _monthly_data():
    """Hourly rows Jan 2020 - May 2020 (hour-ending). Task 1 = up to the
    end of March, task 2 = April, task 3 = May."""
    ts = pd.date_range("2020-01-01 01:00", "2020-06-01 00:00", freq="h")
    month = (ts - pd.Timedelta(hours=1)).month
    task = np.select([month <= 3, month == 4], [1, 2], 3)
    return pd.DataFrame({"timestamp": ts, "load": 100.0, "task": task})


def test_monthly_folds_numbering_and_hours():
    folds = make_monthly_folds(_monthly_data(), "2020-02", first_test_task=2)
    assert [f.test_task for f in folds] == [0, 1]  # Feb, Mar; Apr is task fold 2
    feb = folds[0]
    assert feb.test_df["timestamp"].min() == pd.Timestamp("2020-02-01 01:00")
    assert feb.test_df["timestamp"].max() == pd.Timestamp("2020-03-01 00:00")
    for f in folds:
        assert f.train_df["timestamp"].max() < f.test_df["timestamp"].min()
        assert len(f.train_df) + len(f.test_df) <= len(_monthly_data())


def test_monthly_fold_equals_task_fold_for_the_same_month():
    df = _monthly_data()
    task_fold = make_rolling_folds(df, 2, 2)[0]
    # Extending to April by pretending task 3 is the first task fold.
    monthly = make_monthly_folds(df, "2020-04", first_test_task=3)[0]
    pd.testing.assert_frame_equal(task_fold.test_df.reset_index(drop=True), monthly.test_df.reset_index(drop=True))
    pd.testing.assert_frame_equal(task_fold.train_df.reset_index(drop=True), monthly.train_df.reset_index(drop=True))


def test_fold_groups():
    assert fold_group(-31, 8) == "extended"
    assert fold_group(1, 8) == "extended"
    assert fold_group(2, 8) == "development"
    assert fold_group(9, 8) == "holdout"


def _write_run(root, name, start, tasks, methods, model=None):
    d = root / name
    d.mkdir()
    n = 24 * len(tasks)
    hourly = pd.DataFrame({"timestamp": pd.date_range(start, periods=n, freq="h"), "test_task": np.repeat(tasks, 24)})
    rows = []
    for i, m in enumerate(methods):
        hourly[m] = 1.0 + i + np.random.default_rng(i).random(n)
        rows += [{"test_task": t, "method": m, "pinball_loss": 1.0 + i, "coverage_90": 0.9} for t in tasks]
    hourly.to_csv(d / "hourly_losses.csv", index=False)
    pd.DataFrame(rows).to_csv(d / "results.csv", index=False)
    config = {"model": model or {"params": {"max_depth": 4}}, "quantiles": {"levels": [0.5]}}
    (d / "run_info.json").write_text(json.dumps({"config": config}))


def test_merge_runs_keeps_common_methods(tmp_path, monkeypatch):
    monkeypatch.setattr(mr, "REPORTS_DIR", tmp_path)
    _write_run(tmp_path, "ext", "2010-08-01 01:00", [0, 1], ["seasonal_naive", "lightgbm_full"])
    _write_run(tmp_path, "main", "2010-10-01 01:00", [2, 3], ["seasonal_naive", "gefcom_benchmark", "lightgbm_full"])
    results, hourly, _, dropped = mr.merge_runs(["ext", "main"])
    assert dropped == ["gefcom_benchmark"]
    assert list(hourly.columns) == ["timestamp", "test_task", "seasonal_naive", "lightgbm_full"]
    assert sorted(results["test_task"].unique()) == [0, 1, 2, 3]


def test_merge_runs_refuses_overlap_and_different_models(tmp_path, monkeypatch):
    monkeypatch.setattr(mr, "REPORTS_DIR", tmp_path)
    _write_run(tmp_path, "a", "2010-10-01 01:00", [2], ["lightgbm_full"])
    _write_run(tmp_path, "b", "2010-10-01 01:00", [3], ["lightgbm_full"])
    _write_run(tmp_path, "c", "2010-12-01 01:00", [4], ["lightgbm_full"], model={"params": {"max_depth": 6}})
    with pytest.raises(ValueError, match="share test hours"):
        mr.merge_runs(["a", "b"])
    with pytest.raises(ValueError, match="config.model"):
        mr.merge_runs(["a", "c"])
