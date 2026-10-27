"""
Tests for the optional load-level normalization and recency weighting of
the LightGBM model: the level only ever reads load at or before each
row's forecast origin (and only training load), predictions come back on
the original load scale, and the sample weights decay with age.
"""
import numpy as np
import pandas as pd
import pytest

from src.features.build_features import add_calendar_features
from src.models.lightgbm_model import (
    forecast_origin,
    lightgbm_level_options,
    lightgbm_quantiles,
    load_level,
    recency_weights,
)

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


def test_forecast_origin_follows_hour_ending_convention():
    ts = pd.Series(pd.to_datetime(["2021-03-01 00:00", "2021-03-01 01:00", "2021-03-31 23:00", "2021-04-01 00:00"]))
    expected = pd.to_datetime(["2021-02-01", "2021-03-01", "2021-03-01", "2021-03-01"])
    assert list(forecast_origin(ts)) == list(expected)


@pytest.mark.parametrize("kind", ["trailing_year", "same_month_last_year"])
def test_level_only_reads_load_at_or_before_origin(kind):
    train_df, _ = _synthetic_fold(level_fn=lambda ts: 100.0 + ts.dayofyear)
    # Rows of February 2021: origin 2021-02-01 00:00.
    origin = pd.Timestamp("2021-02-01")
    rows = train_df.loc[forecast_origin(train_df["timestamp"]) == origin, "timestamp"]
    level = load_level(rows, train_df, kind)
    assert len(rows) == 28 * 24 and np.isfinite(level).all()

    corrupted = train_df.copy()
    corrupted.loc[corrupted["timestamp"] > origin, "load"] = 1e9
    np.testing.assert_allclose(load_level(rows, corrupted, kind), level)


def test_trailing_year_level_includes_the_hour_ending_at_the_origin():
    train_df, _ = _synthetic_fold()
    rows = pd.Series([pd.Timestamp("2021-02-15 12:00")])  # full 8760-hour window before its origin
    base = load_level(rows, train_df, "trailing_year")
    bumped = train_df.copy()
    bumped.loc[bumped["timestamp"] == pd.Timestamp("2021-02-01 00:00"), "load"] += 8760.0
    assert load_level(rows, bumped, "trailing_year")[0] == pytest.approx(base[0] + 1.0)


def test_level_is_nan_without_enough_history():
    train_df, _ = _synthetic_fold()
    rows = pd.Series([pd.Timestamp("2020-06-15 12:00")])  # < 1 year of history before its origin
    assert np.isnan(load_level(rows, train_df, "trailing_year")).all()


@pytest.mark.parametrize("kind", ["trailing_year", "same_month_last_year"])
def test_normalized_predictions_ignore_test_load(kind):
    train_df, test_df = _synthetic_fold(level_fn=lambda ts: 100.0 + ts.dayofyear)
    preds = lightgbm_quantiles(train_df, test_df, [0.1, 0.5, 0.9], model_params=_FAST_PARAMS,
                               level_normalization=kind)
    corrupted = test_df.copy()
    corrupted["load"] = 1e9
    preds_corrupted = lightgbm_quantiles(train_df, corrupted, [0.1, 0.5, 0.9], model_params=_FAST_PARAMS,
                                         level_normalization=kind)
    np.testing.assert_allclose(preds, preds_corrupted)


def test_normalized_predictions_are_on_the_original_scale():
    # Load is a constant 500 x daily shape: whatever L is, the median
    # prediction must come back near 500 x shape, not near the ratio ~1.
    train_df, test_df = _synthetic_fold(level_fn=lambda ts: 500.0)
    preds = lightgbm_quantiles(train_df, test_df, [0.5], model_params=_FAST_PARAMS,
                               level_normalization="trailing_year")
    np.testing.assert_allclose(preds[:, 0], test_df["load"].to_numpy(), rtol=0.05)


def test_trailing_year_level_tracks_a_load_trend():
    # Load grows 50% per year; a tree model on the raw target cannot go
    # above the training maximum, the level-normalized one follows the
    # trend into the test month.
    growth = lambda ts: 100.0 * 1.5 ** ((ts - _START) / pd.Timedelta(days=365.25))
    train_df, test_df = _synthetic_fold(level_fn=growth)
    y = test_df["load"].to_numpy()
    raw = lightgbm_quantiles(train_df, test_df, [0.5], model_params=_FAST_PARAMS)
    normed = lightgbm_quantiles(train_df, test_df, [0.5], model_params=_FAST_PARAMS,
                                level_normalization="trailing_year")
    assert np.abs(normed[:, 0] - y).mean() < np.abs(raw[:, 0] - y).mean()


def test_defaults_are_off_and_match_explicit_none():
    train_df, test_df = _synthetic_fold()
    default = lightgbm_quantiles(train_df, test_df, [0.5], model_params=_FAST_PARAMS)
    explicit = lightgbm_quantiles(train_df, test_df, [0.5], model_params=_FAST_PARAMS,
                                  level_normalization=None, recency_half_life_years=None)
    np.testing.assert_array_equal(default, explicit)
    assert lightgbm_level_options({"model": {"params": {}}}) == {
        "level_normalization": None, "recency_half_life_years": None}


def test_invalid_level_normalization_raises():
    train_df, test_df = _synthetic_fold()
    with pytest.raises(ValueError):
        lightgbm_quantiles(train_df, test_df, [0.5], model_params=_FAST_PARAMS, level_normalization="bogus")


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
