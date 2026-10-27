"""compare_runs.combine_runs pairs runs hour by hour and refuses runs that
cover different test hours or a different quantile grid."""
import json

import numpy as np
import pandas as pd
import pytest

import src.evaluation.compare_runs as cr


def _write_run(root, name, timestamps, tasks, model_loss, baseline_loss, n_quantiles=99):
    d = root / name
    d.mkdir()
    pd.DataFrame({"timestamp": timestamps, "test_task": tasks, "seasonal_climatology": baseline_loss,
                  "lightgbm_full": model_loss}).to_csv(d / "hourly_losses.csv", index=False)
    rows = []
    for t in sorted(set(tasks)):
        m = tasks == t
        rows += [{"test_task": t, "method": "seasonal_climatology", "pinball_loss": baseline_loss[m].mean(), "coverage_90": 0.9},
                 {"test_task": t, "method": "lightgbm_full", "pinball_loss": model_loss[m].mean(), "coverage_90": 0.9}]
    pd.DataFrame(rows).to_csv(d / "results.csv", index=False)
    (d / "run_info.json").write_text(json.dumps({"n_quantiles": n_quantiles}))


def _hours(n, start="2010-10-01 01:00"):
    return pd.date_range(start, periods=n, freq="h")


def test_combine_runs_renames_models_and_keeps_baselines(tmp_path, monkeypatch):
    monkeypatch.setattr(cr, "REPORTS_DIR", tmp_path)
    tasks = np.repeat([2, 3, 4], 10)
    rng = np.random.default_rng(0)
    _write_run(tmp_path, "final", _hours(30), tasks, rng.normal(5, 1, 30), rng.normal(8, 1, 30))
    _write_run(tmp_path, "plain", _hours(30), tasks, rng.normal(7, 1, 30), rng.normal(8, 1, 30))

    results, hourly = cr.combine_runs(("final", "final"), [("plain", "plain")], "lightgbm_full")

    assert {"seasonal_climatology", "final", "plain"} <= set(hourly.columns)
    assert set(results["method"]) == {"seasonal_climatology", "final", "plain"}
    np.testing.assert_array_equal(hourly["test_task"].to_numpy(), tasks)


def test_combine_runs_rejects_different_hours(tmp_path, monkeypatch):
    monkeypatch.setattr(cr, "REPORTS_DIR", tmp_path)
    tasks = np.repeat([2, 3], 5)
    _write_run(tmp_path, "final", _hours(10), tasks, np.ones(10), np.ones(10))
    # Same tasks, but shifted timestamps: the task vector alone would not catch this.
    _write_run(tmp_path, "other", _hours(10, start="2010-10-01 02:00"), tasks, np.ones(10), np.ones(10))
    with pytest.raises(ValueError):
        cr.combine_runs(("final", "final"), [("other", "other")], "lightgbm_full")


def test_combine_runs_rejects_different_quantile_grid(tmp_path, monkeypatch):
    monkeypatch.setattr(cr, "REPORTS_DIR", tmp_path)
    tasks = np.repeat([2, 3], 5)
    _write_run(tmp_path, "final", _hours(10), tasks, np.ones(10), np.ones(10), n_quantiles=99)
    _write_run(tmp_path, "dev", _hours(10), tasks, np.ones(10), np.ones(10), n_quantiles=20)
    with pytest.raises(ValueError):
        cr.combine_runs(("final", "final"), [("dev", "dev")], "lightgbm_full")
