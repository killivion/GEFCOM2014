"""
LightGBM quantile regression: one gradient-boosted-tree regressor per
quantile level.

Every feature is something that is genuinely known when a month-ahead
forecast is issued at the fold's cutoff:

- calendar features (precomputed once, see build_features.py);
- a temperature input chosen by `temperature_source` -- "climatology"
  (default: prior-years same hour / day-of-year mean), "last_year", or
  "actual" (the observed temperature of the forecast month itself: future
  information, used ONLY as an explicitly labelled oracle upper bound).
  Training rows use the same variant as test rows, so the model learns
  the relationship it will actually be given at forecast time.

No feature depends on load at all, so the whole test month is predicted
in one batch. Two load-based designs were tried and dropped because,
with honest temperature inputs, both did worse than this one: recursive
day-by-day 24h/168h lags fed by the model's own median (errors compound
over the month), and load-level means anchored at the forecast origin
(one value per month, so the trees overfit to individual months).

Two feature sets:
- "full": calendar (incl. cyclic encodings, day of year) + holiday +
  temperature + heating/cooling degrees.
- "minimal": hour, day of week, month, weekend flag, temperature -- a
  check of whether the extra features in "full" actually help.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

from src.features.build_features import TEMPERATURE_VARIANTS

MINIMAL_FEATURE_COLUMNS = ["hour", "dayofweek", "month", "is_weekend", "temp_mean"]

FULL_FEATURE_COLUMNS = [
    "hour", "dayofweek", "month", "dayofyear", "is_weekend",
    "hour_sin", "hour_cos", "dayofweek_sin", "dayofweek_cos", "is_holiday",
    "temp_mean", "heating_degrees", "cooling_degrees",
]

FEATURE_SETS = {"minimal": MINIMAL_FEATURE_COLUMNS, "full": FULL_FEATURE_COLUMNS}


def _enforce_monotonic_quantiles(preds: np.ndarray, quantile_levels: list[float]) -> np.ndarray:
    """Rearrangement fix for quantile crossing (Chernozhukov, Fernandez-Val
    & Galichon, 2010). Each of the 99 quantile levels is trained as an
    independent LightGBM model, so nothing guarantees
    pred(tau=0.05) <= pred(tau=0.10) <= ... for a given row -- checked
    directly on this data, ~27% of rows had at least one crossing
    violation before this fix. Sorting each row's predicted values into
    non-decreasing order (matching quantile_levels' order) is the
    standard, simple correction. Requires quantile_levels to be sorted
    ascending.
    """
    if list(quantile_levels) != sorted(quantile_levels):
        raise ValueError("quantile_levels must be sorted ascending for the monotonic rearrangement fix")
    return np.sort(preds, axis=1)


def _select_temperature_columns(df: pd.DataFrame, temperature_source: str) -> pd.DataFrame:
    """Renames whichever precomputed temperature variant applies to the
    canonical temp_mean/heating_degrees/cooling_degrees columns the model
    features expect (see build_features.add_temperature_variants)."""
    if temperature_source not in TEMPERATURE_VARIANTS:
        raise ValueError(f"temperature_source must be one of {TEMPERATURE_VARIANTS}, got {temperature_source!r}")
    out = df.copy()
    out["temp_mean"] = out[f"temp_mean_{temperature_source}"]
    out["heating_degrees"] = out[f"heating_degrees_{temperature_source}"]
    out["cooling_degrees"] = out[f"cooling_degrees_{temperature_source}"]
    return out


def lightgbm_quantiles(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    quantile_levels: list[float],
    temperature_source: str = "climatology",
    model_params: dict | None = None,
    on_quantile_done: callable | None = None,
    feature_set: str = "full",
) -> np.ndarray:
    """Trains one LightGBM quantile regressor per quantile level on
    train_df and predicts test_df. test_df's own "load" column is dropped
    before any feature is built, so it can never be read.

    Returns an array of shape (len(test_df), len(quantile_levels)).

    If given, on_quantile_done(index, total) is called after each of the
    `len(quantile_levels)` models finishes training, for progress reporting.
    """
    if feature_set not in FEATURE_SETS:
        raise ValueError(f"feature_set must be one of {list(FEATURE_SETS)}, got {feature_set!r}")
    model_params = model_params or {}
    feature_columns = FEATURE_SETS[feature_set]

    train_feat = _select_temperature_columns(train_df, temperature_source)
    test_feat = _select_temperature_columns(test_df.drop(columns=["load"], errors="ignore"), temperature_source)

    train_mask = train_feat[feature_columns + ["load"]].notna().all(axis=1)
    X_train = train_feat.loc[train_mask, feature_columns]
    y_train = train_feat.loc[train_mask, "load"]
    X_test = test_feat[feature_columns]

    preds = np.zeros((len(test_df), len(quantile_levels)))
    for i, q in enumerate(quantile_levels):
        model = LGBMRegressor(objective="quantile", alpha=q, verbosity=-1, **model_params)
        model.fit(X_train, y_train)
        preds[:, i] = model.predict(X_test)
        if on_quantile_done is not None:
            on_quantile_done(i + 1, len(quantile_levels))

    return _enforce_monotonic_quantiles(preds, quantile_levels)
