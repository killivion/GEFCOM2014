"""
Verifies the temperature-dynamics features: their values on a known
path, that a scenario's path before the month comes from the same
earlier-year scenario (still strictly before the cutoff), and that the
test month's own temperatures and post-cutoff rows cannot change a
prediction when the features are on.
"""
import numpy as np
import pandas as pd
import pytest

from src.models.lightgbm_model import (DYNAMICS_CONTEXT_HOURS, TEMPERATURE_DYNAMICS, build_temperature_scenarios,
                                       lightgbm_quantiles, temperature_dynamics)
from tests.test_lightgbm_no_leakage import _FAST_PARAMS
from tests.test_temperature_scenarios import _SCENARIOS, TEST_START, _multi_year_dataset


def test_dynamics_values_on_a_known_path():
    ts = pd.Series(pd.date_range("2020-01-01 01:00", periods=96, freq="h"))
    temp = np.arange(96, dtype=float)
    out = temperature_dynamics(ts, temp, list(TEMPERATURE_DYNAMICS))
    assert out["temp_ma24"].iloc[:23].isna().all()
    assert out["temp_ma24"].iloc[23] == pytest.approx(np.mean(temp[:24]))
    assert out["temp_ma72"].iloc[71] == pytest.approx(np.mean(temp[:72]))
    assert out["temp_lag24"].iloc[:24].isna().all()
    assert out["temp_lag24"].iloc[30] == temp[6]
    # Hour-ending days: 01:00 .. 00:00 of the next date is one day.
    assert out["temp_day_max"].iloc[0] == temp[23] and out["temp_day_min"].iloc[23] == temp[0]
    assert out["temp_day_min"].iloc[24] == temp[24]


def test_unknown_dynamics_name_raises():
    with pytest.raises(ValueError):
        temperature_dynamics(pd.Series(pd.date_range("2020", periods=3, freq="h")), [1, 2, 3], ["temp_ma7"])


def test_scenario_context_is_the_same_scenario_before_the_month():
    train_df, test_df = _multi_year_dataset()
    plain, labels = build_temperature_scenarios(train_df, test_df["timestamp"], shift_days=[0, 2])
    with_context, labels_context = build_temperature_scenarios(train_df, test_df["timestamp"], shift_days=[0, 2],
                                                               context_hours=DYNAMICS_CONTEXT_HOURS)
    # The oldest year's path would start before the history (2019-01-01
    # minus 72 hours), so it is dropped; every other scenario is kept as is.
    assert set(labels_context) == set(labels) - {(3, 0), (3, 2)}
    keep = [labels.index(label) for label in labels_context]
    np.testing.assert_allclose(with_context[:, DYNAMICS_CONTEXT_HOURS:], plain[keep])
    lookup = train_df.set_index("timestamp")["temp_mean_actual"]
    context_ts = TEST_START - pd.to_timedelta(np.arange(DYNAMICS_CONTEXT_HOURS, 0, -1), unit="h")
    for row, (k, s) in zip(with_context, labels_context):
        source = context_ts - pd.DateOffset(years=k) + pd.Timedelta(days=s)
        assert (source < TEST_START).all()
        np.testing.assert_allclose(row[:DYNAMICS_CONTEXT_HOURS], lookup.reindex(source).to_numpy())


@pytest.mark.parametrize("scenarios", [_SCENARIOS, None])
def test_dynamics_predictions_ignore_the_test_month_and_post_cutoff_rows(scenarios):
    train_df, test_df = _multi_year_dataset()
    dynamics = list(TEMPERATURE_DYNAMICS)
    kwargs = dict(model_params=_FAST_PARAMS, temperature_scenarios=scenarios, temperature_dynamics_features=dynamics)
    preds = lightgbm_quantiles(train_df, test_df, [0.1, 0.5, 0.9], **kwargs)

    temp_cols = [c for c in test_df.columns if c.startswith(("temp_mean", "heating", "cooling"))]
    if scenarios:  # with scenarios no test temperature may be read at all
        corrupted_test = test_df.copy()
        corrupted_test[temp_cols] = 999.0
    else:  # without, the climatology input is used, never the actual one
        corrupted_test = test_df.copy()
        corrupted_test[[c for c in temp_cols if c.endswith("_actual")]] = 999.0
    post_cutoff = test_df.copy()
    post_cutoff["load"] = np.nan
    post_cutoff[temp_cols] = -999.0
    leaky_train = pd.concat([train_df, post_cutoff], ignore_index=True)
    preds_corrupted = lightgbm_quantiles(leaky_train, corrupted_test, [0.1, 0.5, 0.9], **kwargs)
    np.testing.assert_allclose(preds, preds_corrupted)


def test_dynamics_change_the_predictions():
    train_df, test_df = _multi_year_dataset()
    base = lightgbm_quantiles(train_df, test_df, [0.5], model_params=_FAST_PARAMS, temperature_scenarios=_SCENARIOS)
    dyn = lightgbm_quantiles(train_df, test_df, [0.5], model_params=_FAST_PARAMS, temperature_scenarios=_SCENARIOS,
                             temperature_dynamics_features=["temp_ma24", "temp_day_max"])
    assert not np.allclose(base, dyn)
