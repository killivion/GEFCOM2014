import json

import numpy as np
import pandas as pd
import pytest
import yaml

import src.evaluation.run_comparison as rc
from src.data.loader import LoadedData
from src.evaluation.run_comparison import is_oracle_config, significance_vs_model


def _two_method_case():
    # Model is better than "base" by ~1 on every hour and in every fold.
    rng = np.random.default_rng(0)
    model_losses = rng.uniform(1, 2, 400)
    hourly = {"model": model_losses, "base": model_losses + 1.0 + rng.normal(0, 0.1, 400)}
    results = pd.DataFrame({
        "test_task": list(range(1, 6)) * 2,
        "method": ["model"] * 5 + ["base"] * 5,
        "pinball_loss": [1.0, 1.1, 1.2, 1.3, 1.4] + [2.0, 2.2, 2.1, 2.5, 2.3],
    })
    clusters = np.repeat(np.arange(1, 6), 80)
    return results, hourly, clusters


_STATS = {"hac_max_lag": 5, "hac_max_lag_long": 20, "bootstrap_block_hours": 5, "n_bootstrap": 200}


def test_significance_vs_model_better_model_wins():
    results, hourly, clusters = _two_method_case()
    table = significance_vs_model(results, hourly, "model", _STATS, clusters=clusters)

    assert len(table) == 1
    for col in ("model", "other", "family", "mean_diff_other_minus_model", "relative_improvement",
                "block_ci95_low", "block_ci95_high", "dm_stat", "dm_p_value", "dm_p_value_long_lag",
                "month_ci95_low", "month_p_value", "folds_won_by_model", "wilcoxon_p_value",
                "p_conservative", "p_conservative_holm"):
        assert col in table.columns
    row = table.iloc[0]
    assert row["other"] == "base"
    assert row["mean_diff_other_minus_model"] > 0
    assert row["block_ci95_low"] > 0 and row["month_ci95_low"] > 0
    assert row["folds_won_by_model"] == "5/5"
    # The conservative p is the largest of the individual p-values.
    assert row["p_conservative"] == max(row["dm_p_value"], row["dm_p_value_long_lag"], row["block_p_value"],
                                        row["month_p_value"], row["wilcoxon_p_value"])


def test_reference_methods_are_kept_out_of_holm():
    results, hourly, clusters = _two_method_case()
    table = significance_vs_model(results, hourly, "model", _STATS, clusters=clusters, reference_methods=["base"])
    assert table.iloc[0]["family"] == "reference"
    assert np.isnan(table.iloc[0]["p_conservative_holm"])


def test_oracle_config_is_detected():
    honest = {"leakage": {"temperature_source": "climatology"}, "model": {"temperature_scenarios": {"enabled": True}}}
    scenarios_with_actual_key = {"leakage": {"temperature_source": "actual"},
                                 "model": {"temperature_scenarios": {"enabled": True}}}
    oracle = {"leakage": {"temperature_source": "actual"}, "model": {"temperature_scenarios": {"enabled": False}}}
    assert not is_oracle_config(honest)
    assert not is_oracle_config(scenarios_with_actual_key)
    assert is_oracle_config(oracle)


# --- CLI (main) on synthetic data: --baseline none and --skip-stats --------


def _synthetic_loaded_data():
    """Hourly load Jan - Jun 2021 (hour-ending); task 1 = Jan-Feb, then one
    task per month (2 = Mar ... 5 = Jun)."""
    ts = pd.date_range("2021-01-01 01:00", "2021-07-01 00:00", freq="h")
    month = (ts - pd.Timedelta(hours=1)).month
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"timestamp": ts, "task": np.maximum(month - 1, 1),
                       "load": 100 + 10 * np.sin(2 * np.pi * ts.hour / 24) + rng.normal(0, 3, len(ts))})
    return LoadedData(df=df, temp_cols=[])


def _fake_lightgbm(train_df, test_df, quantile_levels, **kwargs):
    # Stands in for the model: the training load's quantiles, every hour.
    return np.tile(np.quantile(train_df["load"], quantile_levels), (len(test_df), 1))


@pytest.fixture
def cli(tmp_path, monkeypatch):
    monkeypatch.setattr(rc, "get_data", lambda *a, **k: _synthetic_loaded_data())
    monkeypatch.setattr(rc, "lightgbm_quantiles", _fake_lightgbm)
    config = {
        "data": {"raw_load_dir": "unused", "processed_path": "unused", "n_tasks": 5},
        "quantiles": {"levels": [0.05, 0.5, 0.95]},
        "leakage": {"temperature_source": "climatology"},
        "statistics": {**_STATS, "seed": 0},
        "model": {"params": {}, "temperature_scenarios": {"enabled": True}},
        "backtest": {"first_test_task": 2, "last_test_task": 5, "development_last_test_task": 5},
    }
    config_path = tmp_path / "tiny.yaml"
    config_path.write_text(yaml.safe_dump(config))

    def call(name, *flags):
        out = tmp_path / name  # absolute, so REPORTS_DIR / out == out
        rc.main(["--config", str(config_path), "--folds", "development", "--output-dir", str(out), *flags])
        return out
    return call


def test_baseline_none_runs_only_the_model(cli):
    out = cli("none", "--baseline", "none")
    summary = pd.read_csv(out / "summary.csv")
    assert summary["method"].tolist() == ["lightgbm_full"]
    assert summary["n_folds"].tolist() == [4]
    assert not (out / "significance.csv").exists()
    assert list(pd.read_csv(out / "pinball_by_fold.csv").columns) == ["test_task", "lightgbm_full"]
    info = json.loads((out / "run_info.json").read_text())
    assert info["baseline"] == "none" and info["skip_stats"] is False


def test_skip_stats_gives_the_same_summary(cli):
    full = cli("full", "--baseline", "climatology")
    fast = cli("fast", "--baseline", "climatology", "--skip-stats")
    for name in ("summary.csv", "pinball_by_fold.csv", "results.csv"):
        pd.testing.assert_frame_equal(pd.read_csv(full / name), pd.read_csv(fast / name))
    assert (full / "significance.csv").exists() and (full / "hourly_losses.csv").exists()
    for skipped in ("significance.csv", "calibration.png", "calibration_curves.csv", "hourly_losses.csv"):
        assert not (fast / skipped).exists()
    info = json.loads((fast / "run_info.json").read_text())
    assert info["skip_stats"] is True and info["baseline"] == "climatology"


def test_run_rejects_unknown_baseline():
    with pytest.raises(ValueError, match="baseline must be one of"):
        rc.run("unused.yaml", baseline="nope")
