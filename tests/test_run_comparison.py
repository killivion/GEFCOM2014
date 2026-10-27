import numpy as np
import pandas as pd

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
