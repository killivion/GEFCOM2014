"""
Verifies the temperature-scenario ("weather ensemble") option of the
LightGBM model: scenarios are built only from temperatures strictly
before the fold's training cutoff (the test month's own temperature and
anything after the cutoff cannot change a prediction), the scenario
count matches the years x shifts that fit inside the history, and both
scenario combinations give monotone quantiles.
"""
import numpy as np
import pandas as pd
import pytest

from src.features.build_features import add_calendar_features
from src.models.lightgbm_model import (
    build_temperature_scenarios,
    combine_scenario_quantiles,
    lightgbm_quantiles,
)
from tests.test_lightgbm_no_leakage import _FAST_PARAMS

HISTORY_START = pd.Timestamp("2019-01-01 00:00")
TEST_START = pd.Timestamp("2022-01-01 00:00")


def _frame(start, n_hours, rng):
    ts = pd.date_range(start, periods=n_hours, freq="h")
    df = add_calendar_features(pd.DataFrame({"timestamp": ts}))
    temp = 50 + 25 * np.sin(2 * np.pi * (df["dayofyear"] - 100) / 365) + rng.normal(0, 5, len(df))
    df["load"] = 100 + 1.5 * np.abs(temp - 60) + 5 * np.sin(2 * np.pi * df["hour"] / 24) + rng.normal(0, 1, len(df))
    for variant in ("actual", "last_year", "climatology"):
        df[f"temp_mean_{variant}"] = temp
        df[f"heating_degrees_{variant}"] = np.clip(65 - temp, 0, None)
        df[f"cooling_degrees_{variant}"] = np.clip(temp - 65, 0, None)
    return df


def _multi_year_dataset(test_days=5, seed=0):
    """Three years of hourly training history (2019-2021) and a short
    test period starting 2022-01-01, all synthetic."""
    rng = np.random.default_rng(seed)
    n_train = int((TEST_START - HISTORY_START) / pd.Timedelta(hours=1))
    return _frame(HISTORY_START, n_train, rng), _frame(TEST_START, 24 * test_days, rng)


_SCENARIOS = {"enabled": True, "shift_days": [-2, 0, 2], "combine": "mixture"}


def test_scenario_count_matches_years_and_shifts_inside_history():
    # Years 2021, 2020, 2019 x shifts -2/0/+2; the only scenario not fully
    # inside the history is 2019 shifted -2 days (starts 2018-12-30).
    train_df, test_df = _multi_year_dataset()
    temps, labels = build_temperature_scenarios(train_df, test_df["timestamp"], shift_days=[-2, 0, 2])
    assert temps.shape == (8, len(test_df))
    assert (3, -2) not in labels
    assert set(labels) == {(k, s) for k in (1, 2, 3) for s in (-2, 0, 2)} - {(3, -2)}

    temps_capped, labels_capped = build_temperature_scenarios(
        train_df, test_df["timestamp"], shift_days=[-2, 0, 2], max_years=2)
    assert temps_capped.shape[0] == 6
    assert {k for k, _ in labels_capped} == {1, 2}


def test_scenario_values_are_the_same_calendar_hour_of_earlier_years():
    train_df, test_df = _multi_year_dataset()
    temps, labels = build_temperature_scenarios(train_df, test_df["timestamp"], shift_days=[0, 2])
    lookup = train_df.set_index("timestamp")["temp_mean_actual"]
    for row, (k, s) in zip(temps, labels):
        expected = lookup.reindex(test_df["timestamp"] - pd.DateOffset(years=k) + pd.Timedelta(days=s)).to_numpy()
        np.testing.assert_allclose(row, expected)


def test_scenario_never_reaches_past_the_cutoff():
    # Last year shifted +370 days lands inside the test period itself, so
    # that scenario must be dropped; every kept scenario reads only
    # timestamps strictly before the cutoff.
    train_df, test_df = _multi_year_dataset()
    _, labels = build_temperature_scenarios(train_df, test_df["timestamp"], shift_days=[0, 370])
    assert (1, 370) not in labels
    assert (2, 370) in labels
    for k, s in labels:
        source = test_df["timestamp"] - pd.DateOffset(years=k) + pd.Timedelta(days=s)
        assert (source < TEST_START).all()


def test_predictions_ignore_test_month_and_post_cutoff_temperatures():
    train_df, test_df = _multi_year_dataset()
    preds = lightgbm_quantiles(train_df, test_df, [0.1, 0.5, 0.9], model_params=_FAST_PARAMS,
                               temperature_scenarios=_SCENARIOS)

    # Corrupt every temperature column of the test month itself ...
    corrupted_test = test_df.copy()
    for col in [c for c in test_df.columns if c.startswith(("temp_mean", "heating", "cooling"))]:
        corrupted_test[col] = 999.0
    # ... and slip post-cutoff rows (no load, so they are never trained
    # on) with absurd temperatures into the training frame.
    post_cutoff = test_df.copy()
    post_cutoff["load"] = np.nan
    for col in [c for c in test_df.columns if c.startswith(("temp_mean", "heating", "cooling"))]:
        post_cutoff[col] = -999.0
    leaky_train = pd.concat([train_df, post_cutoff], ignore_index=True)

    preds_corrupted = lightgbm_quantiles(leaky_train, corrupted_test, [0.1, 0.5, 0.9], model_params=_FAST_PARAMS,
                                         temperature_scenarios=_SCENARIOS)
    np.testing.assert_allclose(preds, preds_corrupted)


def test_scenarios_change_predictions_versus_climatology_input():
    # Sanity check that the option is actually wired in.
    train_df, test_df = _multi_year_dataset()
    test_df["temp_mean_climatology"] = 95.0
    test_df["cooling_degrees_climatology"] = 30.0
    p_clim = lightgbm_quantiles(train_df, test_df, [0.5], model_params=_FAST_PARAMS)
    p_scen = lightgbm_quantiles(train_df, test_df, [0.5], model_params=_FAST_PARAMS, temperature_scenarios=_SCENARIOS)
    assert not np.allclose(p_clim, p_scen)


@pytest.mark.parametrize("combine", ["mixture", "quantile_average"])
def test_scenario_predictions_are_monotone(combine):
    train_df, test_df = _multi_year_dataset()
    levels = [0.05, 0.25, 0.5, 0.75, 0.95]
    preds = lightgbm_quantiles(train_df, test_df, levels, model_params=_FAST_PARAMS,
                               temperature_scenarios={**_SCENARIOS, "combine": combine})
    assert preds.shape == (len(test_df), len(levels))
    assert (np.diff(preds, axis=1) >= 0).all()


def test_mixture_combination_is_monotone_and_spans_the_scenarios():
    rng = np.random.default_rng(0)
    levels = [0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99]
    # Unsorted (crossing) inputs with scenarios at very different levels.
    scenario_preds = rng.normal(0, 1, (6, 50, len(levels))) + np.arange(6)[:, None, None] * 10
    mixed = combine_scenario_quantiles(scenario_preds, levels, method="mixture")
    assert mixed.shape == (50, len(levels))
    assert (np.diff(mixed, axis=1) >= 0).all()
    # The mixture's outer quantiles reach the lowest / highest scenarios,
    # far wider than quantile averaging.
    averaged = combine_scenario_quantiles(scenario_preds, levels, method="quantile_average")
    assert ((mixed[:, -1] - mixed[:, 0]) > (averaged[:, -1] - averaged[:, 0])).all()


def test_identical_scenarios_combine_to_the_scenario_itself():
    levels = np.linspace(0.01, 0.99, 99)
    single = np.sort(np.random.default_rng(1).normal(100, 10, (20, 99)), axis=1)
    stacked = np.repeat(single[None], 4, axis=0)
    np.testing.assert_allclose(combine_scenario_quantiles(stacked, levels, "quantile_average"), single)
    np.testing.assert_allclose(combine_scenario_quantiles(stacked, levels, "mixture"), single, atol=1e-9)


def test_invalid_combine_raises():
    train_df, test_df = _multi_year_dataset()
    with pytest.raises(ValueError):
        lightgbm_quantiles(train_df, test_df, [0.5], model_params=_FAST_PARAMS,
                           temperature_scenarios={**_SCENARIOS, "combine": "bogus"})
