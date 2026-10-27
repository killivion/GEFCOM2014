"""compare_runs.combine_runs pairs runs hour by hour and refuses runs that
cover different test hours."""
import numpy as np
import pandas as pd
import pytest

import src.evaluation.compare_runs as cr


def _write_run(root, name, tasks, model_loss, baseline_loss):
    d = root / name
    d.mkdir()
    pd.DataFrame({"test_task": tasks, "seasonal_climatology": baseline_loss, "lightgbm_full": model_loss}).to_csv(
        d / "hourly_losses.csv", index=False)
    rows = []
    for t in sorted(set(tasks)):
        m = tasks == t
        rows += [{"test_task": t, "method": "seasonal_climatology", "pinball_loss": baseline_loss[m].mean(), "coverage_90": 0.9},
                 {"test_task": t, "method": "lightgbm_full", "pinball_loss": model_loss[m].mean(), "coverage_90": 0.9}]
    pd.DataFrame(rows).to_csv(d / "results.csv", index=False)


def test_combine_runs_renames_models_and_keeps_baselines(tmp_path, monkeypatch):
    monkeypatch.setattr(cr, "REPORTS_DIR", tmp_path)
    tasks = np.repeat([2, 3, 4], 10)
    rng = np.random.default_rng(0)
    _write_run(tmp_path, "final", tasks, rng.normal(5, 1, 30), rng.normal(8, 1, 30))
    _write_run(tmp_path, "plain", tasks, rng.normal(7, 1, 30), rng.normal(8, 1, 30))

    results, hourly, out_tasks = cr.combine_runs(("final", "final"), [("plain", "plain")], "lightgbm_full")

    assert set(hourly) == {"seasonal_climatology", "final", "plain"}
    assert set(results["method"]) == {"seasonal_climatology", "final", "plain"}
    np.testing.assert_array_equal(out_tasks, tasks)


def test_combine_runs_rejects_unpaired_runs(tmp_path, monkeypatch):
    monkeypatch.setattr(cr, "REPORTS_DIR", tmp_path)
    _write_run(tmp_path, "final", np.repeat([2, 3], 5), np.ones(10), np.ones(10))
    _write_run(tmp_path, "other", np.repeat([2, 4], 5), np.ones(10), np.ones(10))
    with pytest.raises(ValueError):
        cr.combine_runs(("final", "final"), [("other", "other")], "lightgbm_full")
