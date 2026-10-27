"""
Verifies that the LightGBM model never uses real data from the period it
is forecasting: test_df's own "load" column is never read, and the
temperature input is the configured (by default prior-years climatology)
variant rather than the forecast month's observed temperature.
"""
import numpy as np
import pandas as pd
import pytest

from src.features.build_features import add_calendar_features
from src.models.lightgbm_model import lightgbm_quantiles


def _synthetic_dataset(n_train_days=60, n_test_days=5, start=pd.Timestamp("2021-01-01"), seed=0):
    """A small, fully synthetic dataset with every column lightgbm_model.py
    expects, so these tests don't depend on the real raw CSVs."""
    rng = np.random.default_rng(seed)

    def build(n_days, day_offset):
        ts = [start + pd.Timedelta(days=day_offset, hours=h) for h in range(24 * n_days)]
        df = pd.DataFrame({"timestamp": ts})
        df["load"] = 100 + 10 * np.sin(np.arange(len(df)) * 2 * np.pi / 24) + rng.normal(0, 1, len(df))
        df = add_calendar_features(df)
        for variant in ("actual", "last_year", "climatology"):
            df[f"temp_mean_{variant}"] = 50.0
            df[f"heating_degrees_{variant}"] = 0.0
            df[f"cooling_degrees_{variant}"] = 0.0
        return df

    return build(n_train_days, 0), build(n_test_days, n_train_days)


_FAST_PARAMS = {"n_estimators": 20, "max_depth": 3, "learning_rate": 0.3, "num_leaves": 7}


def test_full_feature_predictions_invariant_to_test_load_column():
    # End-to-end (real, tiny, fast) LightGBM models: corrupting test_df's
    # own "load" column -- the only place the real forecast-period answer
    # lives -- must not change a single prediction, since it is never used
    # as a feature.
    train_df, test_df = _synthetic_dataset()

    preds_original = lightgbm_quantiles(train_df, test_df, [0.5], model_params=_FAST_PARAMS, feature_set="full")

    corrupted = test_df.copy()
    corrupted["load"] = 999_999.0
    preds_corrupted = lightgbm_quantiles(train_df, corrupted, [0.5], model_params=_FAST_PARAMS, feature_set="full")

    np.testing.assert_allclose(preds_original, preds_corrupted)


def test_minimal_feature_predictions_invariant_to_test_load_column():
    train_df, test_df = _synthetic_dataset()

    preds_original = lightgbm_quantiles(train_df, test_df, [0.5], model_params=_FAST_PARAMS, feature_set="minimal")

    corrupted = test_df.copy()
    corrupted["load"] = 999_999.0
    preds_corrupted = lightgbm_quantiles(train_df, corrupted, [0.5], model_params=_FAST_PARAMS, feature_set="minimal")

    np.testing.assert_allclose(preds_original, preds_corrupted)


def test_temperature_source_switch_changes_the_column_used():
    # Training temperatures are random and load follows them, so the model
    # relies on temperature. In test_df the "actual" and "climatology"
    # columns are set to very different values; the two sources must then
    # give different predictions (the switch really selects the column).
    train_df, test_df = _synthetic_dataset()
    temp = np.random.default_rng(1).uniform(20, 90, len(train_df))
    train_df["load"] = 100 + 2 * (temp - 50)
    for variant in ("actual", "last_year", "climatology"):
        train_df[f"temp_mean_{variant}"] = temp
    test_df["temp_mean_actual"] = 90.0
    test_df["temp_mean_climatology"] = 20.0

    kwargs = dict(model_params=_FAST_PARAMS, feature_set="minimal")
    p_actual = lightgbm_quantiles(train_df, test_df, [0.5], temperature_source="actual", **kwargs)
    p_clim = lightgbm_quantiles(train_df, test_df, [0.5], temperature_source="climatology", **kwargs)

    assert p_actual.mean() > p_clim.mean() + 20


def test_invalid_temperature_source_raises():
    # Anything outside actual/last_year/climatology is rejected.
    train_df, test_df = _synthetic_dataset()
    with pytest.raises(ValueError):
        lightgbm_quantiles(train_df, test_df, [0.5], temperature_source="bogus", model_params=_FAST_PARAMS)


def test_full_feature_set_works_without_median_quantile():
    # "full" no longer needs 0.5 in quantile_levels.
    train_df, test_df = _synthetic_dataset()
    preds = lightgbm_quantiles(train_df, test_df, [0.1, 0.9], model_params=_FAST_PARAMS, feature_set="full")
    assert preds.shape == (len(test_df), 2)
    assert np.isfinite(preds).all()
