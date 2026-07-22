"""
LightGBM quantile regression: one gradient-boosted-tree regressor per
quantile level, trained on calendar (+ optionally actual temperature)
features from src/features/build_features.py.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

from src.features.build_features import add_calendar_features, add_temperature_feature

FEATURE_COLUMNS = ["hour", "dayofweek", "month", "is_weekend", "temp_mean"]


def _build_features(
    df: pd.DataFrame, train_df: pd.DataFrame, temp_cols: list[str], use_actual_future_temperature: bool
) -> pd.DataFrame:
    out = add_calendar_features(df)
    out = add_temperature_feature(out, train_df, temp_cols, use_actual_future_temperature)
    return out


def lightgbm_quantiles(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    quantile_levels: list[float],
    temp_cols: list[str],
    use_actual_future_temperature: bool = True,
    model_params: dict | None = None,
    on_quantile_done: callable | None = None,
) -> np.ndarray:
    """Trains one LightGBM quantile regressor per quantile level on
    train_df and predicts quantiles for test_df.

    Returns an array of shape (len(test_df), len(quantile_levels)).

    If given, on_quantile_done(index, total) is called after each of the
    `len(quantile_levels)` models finishes training, for progress reporting.
    """
    model_params = model_params or {}

    # `train_df` is passed as the temperature history source for BOTH train
    # and test feature building, so the same-hour-last-year substitute (see
    # add_temperature_feature) never reaches into the period being forecast.
    train_feat = _build_features(train_df, train_df, temp_cols, use_actual_future_temperature)
    test_feat = _build_features(test_df, train_df, temp_cols, use_actual_future_temperature)

    # Task 1's earliest history has no released load (see loader.py); drop
    # those rows from training rather than fitting against NaN targets.
    train_mask = train_feat["load"].notna()
    X_train = train_feat.loc[train_mask, FEATURE_COLUMNS]
    y_train = train_feat.loc[train_mask, "load"]
    X_test = test_feat[FEATURE_COLUMNS]

    preds = np.zeros((len(test_df), len(quantile_levels)))
    for i, q in enumerate(quantile_levels):
        model = LGBMRegressor(objective="quantile", alpha=q, verbosity=-1, **model_params)
        model.fit(X_train, y_train)
        preds[:, i] = model.predict(X_test)
        if on_quantile_done is not None:
            on_quantile_done(i + 1, len(quantile_levels))

    return preds
