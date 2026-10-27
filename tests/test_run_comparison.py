import numpy as np
import pandas as pd

from src.evaluation.run_comparison import significance_vs_model


def test_significance_vs_model_better_model_wins():
    # Model is better than "base" by ~1 on every hour and in every fold.
    rng = np.random.default_rng(0)
    model_losses = rng.uniform(1, 2, 400)
    hourly = {"model": model_losses, "base": model_losses + 1.0 + rng.normal(0, 0.1, 400)}
    results = pd.DataFrame({
        "test_task": list(range(1, 6)) * 2,
        "method": ["model"] * 5 + ["base"] * 5,
        "pinball_loss": [1.0, 1.1, 1.2, 1.3, 1.4] + [2.0, 2.2, 2.1, 2.5, 2.3],
    })
    stats = {"hac_max_lag": 5, "bootstrap_block_hours": 5, "n_bootstrap": 200}
    table = significance_vs_model(results, hourly, "model", stats)

    assert len(table) == 1
    for col in ("model", "other", "mean_diff_other_minus_model", "bootstrap_ci95_low",
                "bootstrap_ci95_high", "dm_stat", "dm_p_value", "dm_p_value_holm",
                "folds_won_by_model", "wilcoxon_p_value"):
        assert col in table.columns
    row = table.iloc[0]
    assert row["other"] == "base"
    assert row["mean_diff_other_minus_model"] > 0
    assert row["bootstrap_ci95_low"] > 0
    assert row["folds_won_by_model"] == "5/5"
