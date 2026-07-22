"""
Verifies that the LightGBM model never uses real data from the period it
is forecasting -- only real pre-cutoff history, or values it has already
generated itself for an earlier part of the same forecast window.
"""
import numpy as np
import pandas as pd
import pytest

from src.features.build_features import add_calendar_features, add_load_lag_features
from src.models.lightgbm_model import FULL_FEATURE_COLUMNS, _predict_recursive, lightgbm_quantiles


def _synthetic_dataset(n_train_days=60, n_test_days=5, start=pd.Timestamp("2021-01-01"), seed=0):
    """A small, fully synthetic dataset with every column lightgbm_model.py
    expects, so these tests don't depend on the real raw CSVs."""
    rng = np.random.default_rng(seed)

    def build(n_days, day_offset):
        ts = [start + pd.Timedelta(days=day_offset, hours=h) for h in range(24 * n_days)]
        df = pd.DataFrame({"timestamp": ts})
        df["load"] = 100 + 10 * np.sin(np.arange(len(df)) * 2 * np.pi / 24) + rng.normal(0, 1, len(df))
        df = add_calendar_features(df)
        df["temp_mean_actual"] = 50.0
        df["temp_mean_last_year"] = 50.0
        df["heating_degrees_actual"] = 0.0
        df["cooling_degrees_actual"] = 0.0
        df["heating_degrees_last_year"] = 0.0
        df["cooling_degrees_last_year"] = 0.0
        return df

    return build(n_train_days, 0), build(n_test_days, n_train_days)


_FAST_PARAMS = {"n_estimators": 20, "max_depth": 3, "learning_rate": 0.3, "num_leaves": 7}


def test_train_lag_features_never_reference_future_rows():
    # add_load_lag_features is only ever applied to genuinely-known
    # historical data (train_df), so it should be a plain backward shift:
    # row i's lag_24h/168h must equal the load exactly 24/168 rows earlier,
    # never a later (future-relative-to-that-row) value.
    df = pd.DataFrame({
        "timestamp": pd.date_range("2021-01-01", periods=200, freq="h"),
        "load": np.arange(200, dtype=float),
    })
    out = add_load_lag_features(df)

    assert (out["load_lag_24h"].iloc[24:].to_numpy() == out["load"].iloc[:-24].to_numpy()).all()
    assert (out["load_lag_168h"].iloc[168:].to_numpy() == out["load"].iloc[:-168].to_numpy()).all()
    # Rolling mean at row i must only cover rows [i-168, i-1] -- not row i itself.
    assert out["load_rolling_mean_7d"].iloc[168] == pytest.approx(out["load"].iloc[0:168].mean())


def test_recursive_lag_uses_generated_prediction_not_real_test_load():
    # Stub "models": the 0.5-quantile model always predicts a fixed,
    # distinctive value (777) regardless of input, simulating day 1's
    # generated forecast. The 0.1-quantile model instead echoes back
    # whatever load_lag_24h it was given, so we can directly inspect what
    # value the SECOND test day's lag feature actually received.
    class _ConstantModel:
        def __init__(self, value):
            self.value = value

        def predict(self, X):
            return np.full(len(X), self.value)

    class _EchoLag24Model:
        def predict(self, X):
            return X["load_lag_24h"].to_numpy()

    models = {0.1: _EchoLag24Model(), 0.5: _ConstantModel(777.0)}
    quantile_levels = [0.1, 0.5]

    train_start = pd.Timestamp("2021-01-01")
    train_df = pd.DataFrame({
        "timestamp": [train_start + pd.Timedelta(hours=h) for h in range(24 * 10)],
        "load": 1.0,  # distinctive real historical value
    })

    test_start = train_start + pd.Timedelta(days=10)
    test_df = pd.DataFrame({"timestamp": [test_start + pd.Timedelta(hours=h) for h in range(24 * 2)]})
    # Deliberately wrong/distinctive "real" load for the forecast period --
    # must never be read, since it isn't even a feature column.
    test_df["load"] = -999.0
    test_df = add_calendar_features(test_df)
    for col in ["temp_mean_actual", "temp_mean_last_year", "heating_degrees_actual",
                "cooling_degrees_actual", "heating_degrees_last_year", "cooling_degrees_last_year"]:
        test_df[col] = 0.0

    preds = _predict_recursive(models, quantile_levels, train_df, test_df, FULL_FEATURE_COLUMNS,
                                use_actual_future_temperature=True)

    day1_echoed_lag24 = preds[:24, 0]
    day2_echoed_lag24 = preds[24:, 0]

    # Day 1's lag_24h correctly comes from train_df's real history (1.0).
    assert np.allclose(day1_echoed_lag24, 1.0)
    # Day 2's lag_24h comes from day 1's own GENERATED median prediction
    # (777.0) -- not train's 1.0, and absolutely not test_df's real -999.0.
    assert np.allclose(day2_echoed_lag24, 777.0)


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
