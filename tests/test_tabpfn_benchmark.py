"""
Verifies the TabPFN benchmark without the network (a fake regressor
stands in for the API): TabPFN never sees the test month's load or
temperature, its context is a recency-weighted sample of training rows
only, chunked prediction equals one call, the week-block folds are
disjoint and respect the embargo, cross-fold recalibration never uses a
fold's own rows, and the CLI runs both tasks end to end.
"""
import json

import numpy as np
import pandas as pd
import pytest
import yaml

import src.evaluation.run_tabpfn_benchmark as rbm
import src.models.tabpfn_model as tm
from src.data.loader import LoadedData
from src.evaluation.backtest import make_week_block_folds
from src.features.build_features import operating_day_timestamps
from src.models.conformal import cross_fold_recalibration, seasonal_recalibration
from src.models.lightgbm_model import recency_weights
from src.models.tabpfn_model import inclusion_probabilities, recency_subsample, tabpfn_quantiles
from tests.test_temperature_scenarios import _SCENARIOS, _frame, _multi_year_dataset

LEVELS = [0.05, 0.5, 0.95]


class FakeTabPFN:
    """Deterministic stand-in: quantiles of the context load, shifted by
    the row's temperature -- so a leaked temperature would change it."""
    fitted = []  # every context it was fitted on (class-level, for inspection)

    def __init__(self, **params):
        self.params = params

    def fit(self, X, y):
        FakeTabPFN.fitted.append(X)
        self.y = np.asarray(y)
        return self

    def predict(self, X, output_type, quantiles):
        assert output_type == "quantiles"
        return [np.quantile(self.y, q) + 0.1 * X["temp_mean"].to_numpy() for q in quantiles]


@pytest.fixture(autouse=True)
def fake_tabpfn(monkeypatch):
    FakeTabPFN.fitted = []
    monkeypatch.setattr(tm, "_default_regressor_factory", lambda **p: FakeTabPFN(**p))
    monkeypatch.setattr(rbm, "_tabpfn_api_info", lambda: {})  # never touch the network


def test_recency_subsample_size_order_and_bias():
    ts = pd.Series(pd.date_range("2015-01-01", "2020-12-31", freq="h"))
    cutoff = ts.max()
    rows = recency_subsample(ts, cutoff, 2000, half_life_years=1, seed=0)
    assert len(rows) == 2000 and len(np.unique(rows)) == 2000 and (np.diff(rows) > 0).all()
    uniform = recency_subsample(ts, cutoff, 2000, half_life_years=None, seed=0)
    # Recency weighting over-represents recent rows.
    assert ts.iloc[rows].mean() > ts.iloc[uniform].mean() + pd.Timedelta(days=180)
    assert len(recency_subsample(ts.iloc[:100], cutoff, 2000, None, 0)) == 100


def test_recency_inclusion_probability_is_proportional_to_weight():
    # Half of the rows drawn: with numpy's sequential choice(p=...) the
    # most recent rows would be included far less than proportionally.
    ts = pd.Series(pd.date_range("2016-01-01", periods=400, freq="5D"))
    cutoff = ts.max()
    weights = recency_weights(ts, cutoff, 1.0)
    expected = inclusion_probabilities(weights, 200)
    assert np.isclose(expected.sum(), 200) and expected.max() <= 1
    counts = np.zeros(len(ts))
    for seed in range(2000):
        rows = recency_subsample(ts, cutoff, 200, half_life_years=1.0, seed=seed)
        assert len(rows) == 200
        counts[rows] += 1
    np.testing.assert_allclose(counts / 2000, expected, atol=0.05)


def test_tabpfn_never_sees_test_load_or_temperature():
    train_df, test_df = _multi_year_dataset()
    settings = {"context_rows": 3000}
    preds = tabpfn_quantiles(train_df, test_df, LEVELS, temperature_scenarios=_SCENARIOS,
                             recency_half_life_years=2, tabpfn=settings)

    corrupted = test_df.copy()
    corrupted["load"] = 999_999.0
    for col in [c for c in test_df.columns if c.startswith(("temp_mean", "heating", "cooling"))]:
        corrupted[col] = 999.0
    preds_corrupted = tabpfn_quantiles(train_df, corrupted, LEVELS, temperature_scenarios=_SCENARIOS,
                                       recency_half_life_years=2, tabpfn=settings)
    np.testing.assert_allclose(preds, preds_corrupted)
    assert preds.shape == (len(test_df), len(LEVELS)) and (np.diff(preds, axis=1) >= 0).all()


def test_context_is_a_sample_of_training_rows_only():
    train_df, test_df = _multi_year_dataset()
    timings = {}
    tabpfn_quantiles(train_df, test_df, LEVELS, temperature_scenarios=_SCENARIOS, recency_half_life_years=2,
                     tabpfn={"context_rows": 500}, timings=timings)
    context = FakeTabPFN.fitted[-1]
    assert len(context) == 500 and timings["context_rows"] == 500
    assert set(context.index) <= set(train_df.index)
    assert {"fit_seconds", "predict_seconds"} <= set(timings)


def test_chunked_prediction_equals_one_call():
    train_df, test_df = _multi_year_dataset()
    one = tabpfn_quantiles(train_df, test_df, LEVELS, temperature_scenarios=_SCENARIOS,
                           tabpfn={"context_rows": 500, "predict_chunk_rows": 10**9})
    chunked = tabpfn_quantiles(train_df, test_df, LEVELS, temperature_scenarios=_SCENARIOS,
                               tabpfn={"context_rows": 500, "predict_chunk_rows": 7})
    np.testing.assert_allclose(one, chunked)


def _hourly(start, end):
    ts = pd.date_range(start, end, freq="h")
    return pd.DataFrame({"timestamp": ts, "load": 1.0, "task": 1})


@pytest.mark.parametrize("embargo_days", [0, 3, 7])
def test_week_blocks_disjoint_complete_and_embargoed(embargo_days):
    df = _hourly("2020-01-01 01:00", "2021-06-01 00:00")
    df.loc[5, "load"] = np.nan
    end = pd.Timestamp("2021-03-01 01:00")
    folds = make_week_block_folds(df, end, n_splits=4, embargo_days=embargo_days, seed=1)
    tested = pd.concat([f.test_df for f in folds])
    eligible = df[(df["timestamp"] < end) & df["load"].notna()]
    assert sorted(tested.index) == sorted(eligible.index)  # every eligible hour tested exactly once
    for f in folds:
        assert f.test_df["timestamp"].max() < end and f.train_df["timestamp"].max() < end
        test_t = operating_day_timestamps(f.test_df["timestamp"]).to_numpy()
        train_t = operating_day_timestamps(f.train_df["timestamp"]).to_numpy()
        gap = np.min(np.abs(train_t[:, None] - test_t[None, :]), axis=1)
        assert (gap > np.timedelta64(embargo_days, "D")).all()
        # Whole weeks: every test week's operating days are all in the test set.
        day = operating_day_timestamps(f.test_df["timestamp"]).dt.normalize()
        assert (day.groupby(day - pd.to_timedelta(day.dt.dayofweek, unit="D")).nunique() <= 7).all()


def test_cross_fold_recalibration_ignores_the_folds_own_outcomes():
    rng = np.random.default_rng(0)
    n = 300
    fold = np.repeat([1, 2, 3], n // 3)
    y = rng.normal(0, 1, n)
    preds = np.sort(rng.normal(0, 1, (n, 3)), axis=1)
    out = cross_fold_recalibration(y, preds, fold, LEVELS, "additive")
    y_changed = y.copy()
    y_changed[fold == 2] += 100  # fold 2's own outcomes
    out_changed = cross_fold_recalibration(y_changed, preds, fold, LEVELS, "additive")
    np.testing.assert_allclose(out[fold == 2], out_changed[fold == 2])
    assert not np.allclose(out[fold == 1], out_changed[fold == 1])  # but they calibrate the other folds


def _synthetic_data():
    """Hourly 2019 - Mar 2022 with temperatures; task 1 = up to 2021,
    then one task per month (2 = Jan 2022 ... 4 = Mar 2022)."""
    rng = np.random.default_rng(0)
    df = _frame(pd.Timestamp("2019-01-01 01:00"), 24 * (365 * 3 + 31 + 28 + 31), rng)
    month_start = (df["timestamp"] - pd.Timedelta(hours=1))
    df["task"] = np.where(month_start.dt.year < 2022, 1, month_start.dt.month + 1)
    return LoadedData(df=df, temp_cols=[])


@pytest.fixture
def cli(tmp_path, monkeypatch):
    monkeypatch.setattr(rbm, "get_data", lambda *a, **k: _synthetic_data())
    base = {
        "data": {"raw_load_dir": "unused", "processed_path": "unused", "n_tasks": 4},
        "quantiles": {"levels": LEVELS},
        "leakage": {"temperature_source": "climatology"},
        "statistics": {"hac_max_lag": 5, "hac_max_lag_long": 20, "bootstrap_block_hours": 5, "n_bootstrap": 50,
                       "seed": 0},
        "model": {"params": {"n_estimators": 10, "max_depth": 3, "num_leaves": 7},
                  "temperature_scenarios": {**_SCENARIOS, "shift_days": [0]}, "recency_half_life_years": 2},
        "backtest": {"first_test_task": 2, "last_test_task": 4, "development_last_test_task": 3,
                     "extended_first_month": "2021-10"},
    }
    (tmp_path / "base.yaml").write_text(yaml.safe_dump(base))
    bench = {"base_config": str(tmp_path / "base.yaml"), "timeseries": {"folds": "all"},
             "tabular": {"n_splits": 3, "embargo_days": 1, "seed": 0},
             "tabpfn": {"predict_chunk_rows": 1000, "seed": 0, "regressor_params": {},
                        "variants": {"tabpfn": {"context_rows": None},
                                     "tabpfn_recency": {"context_rows": 400, "recency": True}}},
             "calibration": {"primary": "additive", "seasonal_window_months": 1}}
    (tmp_path / "bench.yaml").write_text(yaml.safe_dump(bench))

    def call(*args):
        rbm.main(["--config", str(tmp_path / "bench.yaml"), "--output-dir", str(tmp_path / "out"), *args])
        return tmp_path / "out"
    return call


@pytest.mark.parametrize("task", ["timeseries", "tabular"])
def test_cli_runs_both_methods_and_evaluates(cli, task):
    out = cli("--task", task) / task
    summary = pd.read_csv(out / "summary.csv")
    methods = rbm.TASK_METHODS[task]
    expected = {f"{m}{s}" for m in methods for s in ("", "+additive", "+scaled", "+pit")}
    groups = {"timeseries": {"development": 2, "holdout": 1, "all": 3}, "tabular": {"tabular": 3}}[task]
    for group, n_folds in groups.items():
        rows = summary[summary["fold_group"] == group]
        assert set(rows["method"]) == expected and (rows["n_folds"] == n_folds).all()
    significance = pd.read_csv(out / "significance.csv")
    assert set(significance["model"]) == {f"{m}{s}" for m in methods if m != "lightgbm" for s in ("", "+additive")}
    assert "holdout" not in set(significance["fold_group"])  # one holdout fold: nothing to test
    runtime = pd.read_csv(out / "runtime.csv")
    assert set(runtime["method"]) == set(methods) and (runtime["total_seconds"] > 0).all()
    info = json.loads((out / "run_info.json").read_text())["methods"]
    assert set(info) == set(methods) and info["tabpfn"]["recency_half_life_years"] is None
    if task == "timeseries":
        assert info["tabpfn_recency"]["recency_half_life_years"] == 2
        # Primary variant: every training row is context; the recency variant: 400.
        assert max(len(x) for x in FakeTabPFN.fitted) > 400 and min(len(x) for x in FakeTabPFN.fitted) == 400


def test_cli_rejects_recency_variant_for_tabular(cli):
    with pytest.raises(SystemExit):
        cli("--task", "tabular", "--methods", "tabpfn_recency")


def test_cli_evaluates_only_shared_folds_and_keeps_run_info(cli):
    cli("--task", "timeseries", "--methods", "lightgbm")
    out = cli("--task", "timeseries", "--methods", "tabpfn", "--max-folds", "1") / "timeseries"
    summary = pd.read_csv(out / "summary.csv")
    assert (summary["n_folds"] == 1).all()
    assert set(json.loads((out / "run_info.json").read_text())["methods"]) == {"lightgbm", "tabpfn"}
    assert not (out / "significance.csv").exists()  # one fold: nothing to test


def _network_error():
    # Stands in for httpx.ReadError without importing httpx.
    return type("ReadError", (Exception,), {"__module__": "httpx"})("connection reset")


def test_cli_retries_a_fold_after_a_network_error(cli, monkeypatch):
    monkeypatch.setattr(rbm.time, "sleep", lambda s: None)
    calls = {"n": 0}
    real_fit = FakeTabPFN.fit

    def flaky_fit(self, X, y):
        calls["n"] += 1
        if calls["n"] == 1:
            raise _network_error()
        return real_fit(self, X, y)

    monkeypatch.setattr(FakeTabPFN, "fit", flaky_fit)
    out = cli("--task", "tabular", "--methods", "tabpfn") / "tabular"
    assert calls["n"] == 3 + 1  # three folds, one of them twice
    assert (out / "predictions_tabpfn.npz").exists()


def test_cli_does_not_retry_other_errors(cli, monkeypatch):
    def broken_fit(self, X, y):
        raise ValueError("bad input")

    monkeypatch.setattr(FakeTabPFN, "fit", broken_fit)
    with pytest.raises(ValueError):
        cli("--task", "tabular", "--methods", "tabpfn")


def _seasonal_case():
    # Tasks 1..6 = Nov, Dec, Jan, Feb, Mar, Apr; 100 rows each.
    rng = np.random.default_rng(0)
    task = np.repeat(np.arange(1, 7), 100)
    month = np.repeat([11, 12, 1, 2, 3, 4], 100)
    y = rng.normal(0, 1, len(task))
    preds = np.sort(rng.normal(0, 1, (len(task), 3)), axis=1)
    return y, preds, task, month


def test_seasonal_recalibration_uses_only_earlier_months_of_the_season():
    y, preds, task, month = _seasonal_case()
    adjust = [4]  # February: window 1 -> Jan (task 3) only among earlier tasks; Mar is later
    base = seasonal_recalibration(y, preds, task, month, LEVELS, "additive", adjust, month_window=1)
    for changed_task, should_move in ((4, False), (5, False), (2, False), (3, True)):
        y2 = y.copy()
        y2[task == changed_task] += 50
        out = seasonal_recalibration(y2, preds, task, month, LEVELS, "additive", adjust, month_window=1)
        moved = not np.allclose(out[task == 4], base[task == 4])
        assert moved == should_move, changed_task
    # Only the tasks to adjust change.
    np.testing.assert_allclose(base[task != 4], preds[task != 4])


def test_seasonal_window_wraps_december_to_january():
    y, preds, task, month = _seasonal_case()
    base = seasonal_recalibration(y, preds, task, month, LEVELS, "additive", [3], month_window=1)
    y2 = y.copy()
    y2[task == 2] += 50  # December calibrates January
    out = seasonal_recalibration(y2, preds, task, month, LEVELS, "additive", [3], month_window=1)
    assert not np.allclose(out[task == 3], base[task == 3])


def test_cli_seasonal_calibration_from_history(cli):
    cli("--task", "timeseries", "--calibration-history")
    out = cli("--task", "timeseries") / "timeseries"
    assert {f"calibration_{m}.npz" for m in rbm.TASK_METHODS["timeseries"]} <= {f.name for f in out.iterdir()}
    summary = pd.read_csv(out / "summary.csv")
    for m in rbm.TASK_METHODS["timeseries"]:
        assert {f"{m}+seasonal_{cm}" for cm in ("additive", "scaled", "pit")} <= set(summary["method"])
    assert "tabpfn+seasonal_additive" in set(pd.read_csv(out / "significance.csv")["model"])
    # Holm runs across every comparison of a fold group (variants x calibrations).
    sig = pd.read_csv(out / "significance.csv")
    for _, g in sig.groupby("fold_group"):
        assert len(g) == 6  # 2 TabPFN variants x (raw, +additive, +seasonal_additive)
        assert (g["p_conservative_holm"] >= g["p_conservative"] - 1e-12).all()
        assert g["p_conservative_holm"].max() <= 1
    info = json.loads((out / "run_info.json").read_text())["methods"]
    assert {"lightgbm_calibration_history", "lightgbm"} <= set(info)
