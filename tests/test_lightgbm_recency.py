"""
Tests for the optional recency weighting of the LightGBM model: the
sample weights decay with age before the training cutoff, actually change
the fit, and are off by default.
"""
import numpy as np
import pandas as pd
import pytest

from src.features.build_features import add_calendar_features
from src.models.lightgbm_model import lightgbm_quantiles, recency_weights

_FAST_PARAMS = {"n_estimators": 20, "max_depth": 3, "learning_rate": 0.3, "num_leaves": 7}

# Hour-ending timestamps: training covers 2020-01-01 01:00 .. 2021-03-01
# 00:00 (cutoff), the test fold is March 2021.
_START = pd.Timestamp("2020-01-01 01:00")
_CUTOFF = pd.Timestamp("2021-03-01 00:00")
_TEST_END = pd.Timestamp("2021-04-01 00:00")


def _frame(timestamps, load):
    df = pd.DataFrame({"timestamp": timestamps, "load": load})
    df = add_calendar_features(df)
    for variant in ("actual", "last_year", "climatology"):
        df[f"temp_mean_{variant}"] = 50.0
        df[f"heating_degrees_{variant}"] = 0.0
        df[f"cooling_degrees_{variant}"] = 0.0
    return df


def _synthetic_fold(level_fn=lambda ts: 100.0, seed=0, start=_START):
    rng = np.random.default_rng(seed)
    ts = pd.Series(pd.date_range(start, _TEST_END, freq="h"))
    daily = 1 + 0.2 * np.sin(2 * np.pi * ts.dt.hour.to_numpy() / 24)
    load = np.asarray([level_fn(t) for t in ts], dtype=float) * daily * (1 + rng.normal(0, 0.01, len(ts)))
    df = _frame(ts, load)
    return df[df["timestamp"] <= _CUTOFF].reset_index(drop=True), df[df["timestamp"] > _CUTOFF].reset_index(drop=True)


def test_recency_weights_are_monotone_in_recency():
    ts = pd.Series(pd.date_range("2015-01-01", _CUTOFF, freq="D"))
    w = recency_weights(ts, _CUTOFF, half_life_years=2.0)
    assert np.all(np.diff(w) > 0)
    assert w[-1] == pytest.approx(1.0)
    two_years_back = recency_weights(pd.Series([_CUTOFF - pd.Timedelta(days=2 * 365.25)]), _CUTOFF, 2.0)
    assert two_years_back[0] == pytest.approx(0.5)
    with pytest.raises(ValueError):
        recency_weights(ts, _CUTOFF, half_life_years=0)


def test_recency_weights_pull_predictions_towards_recent_load():
    # Three years of history; load is 100 before March 2020 and 200 after,
    # so two of the three earlier Marches sit at 100. Unweighted, the
    # March median stays near 100; with a short half-life it moves to 200.
    train_df, test_df = _synthetic_fold(
        level_fn=lambda ts: 200.0 if ts > pd.Timestamp("2020-03-01") else 100.0,
        start=pd.Timestamp("2018-01-01 01:00"))
    plain = lightgbm_quantiles(train_df, test_df, [0.5], model_params=_FAST_PARAMS)
    weighted = lightgbm_quantiles(train_df, test_df, [0.5], model_params=_FAST_PARAMS, recency_half_life_years=0.25)
    assert weighted.mean() > plain.mean() + 50


def test_recency_weighting_is_off_by_default():
    train_df, test_df = _synthetic_fold()
    default = lightgbm_quantiles(train_df, test_df, [0.5], model_params=_FAST_PARAMS)
    explicit = lightgbm_quantiles(train_df, test_df, [0.5], model_params=_FAST_PARAMS, recency_half_life_years=None)
    np.testing.assert_array_equal(default, explicit)
