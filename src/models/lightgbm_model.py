"""
LightGBM quantile regression: one gradient-boosted-tree regressor per
quantile level.

Two feature sets:
- "full": calendar + temperature + load-lag features (the intended
  model). Calendar and temperature are precomputed once on the full
  dataset (see src/data/loader.py:get_data and
  src/features/build_features.py) and simply selected here. Load-lag
  features (load_lag_24h, load_lag_168h, load_rolling_mean_7d/std_7d) are
  NOT precomputed globally, because what's "known" at forecast time
  differs between training rows (their own real history) and a test
  fold's rows (only real history up to the fold's cutoff, plus whatever
  the model has already predicted earlier in the SAME forecast month).
  Training uses the former (vectorized, see add_load_lag_features);
  prediction uses the latter, via _predict_recursive below, which
  forecasts one day at a time and feeds each day's median prediction
  back in as "known" load for the next day's lag features -- so no
  lag/rolling feature, at any point, is ever built from real data
  belonging to the period being forecast.
- "minimal": just the original calendar + temperature features, no load
  lags. Since it has no features that depend on recursively-generated
  data, it can predict the whole test month in one batch (much faster)
  and serves as a baseline to check whether the lag/rolling features in
  "full" actually help.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

from src.features.build_features import add_load_lag_features

MINIMAL_FEATURE_COLUMNS = ["hour", "dayofweek", "month", "is_weekend", "temp_mean"]

FULL_FEATURE_COLUMNS = [
    "hour", "dayofweek", "month", "dayofyear", "is_weekend",
    "hour_sin", "hour_cos", "dayofweek_sin", "dayofweek_cos", "is_holiday",
    "temp_mean", "heating_degrees", "cooling_degrees",
    "load_lag_24h", "load_lag_168h", "load_rolling_mean_7d", "load_rolling_std_7d",
]

FEATURE_SETS = {"minimal": MINIMAL_FEATURE_COLUMNS, "full": FULL_FEATURE_COLUMNS}

_ROLLING_WINDOW_HOURS = 168


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


def _select_temperature_columns(df: pd.DataFrame, use_actual_future_temperature: bool) -> pd.DataFrame:
    """Renames whichever precomputed temperature variant applies to the
    canonical temp_mean/heating_degrees/cooling_degrees columns the model
    features expect (see build_features.add_temperature_variants)."""
    out = df.copy()
    suffix = "actual" if use_actual_future_temperature else "last_year"
    out["temp_mean"] = out[f"temp_mean_{suffix}"]
    out["heating_degrees"] = out[f"heating_degrees_{suffix}"]
    out["cooling_degrees"] = out[f"cooling_degrees_{suffix}"]
    return out


def lightgbm_quantiles(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    quantile_levels: list[float],
    use_actual_future_temperature: bool = True,
    model_params: dict | None = None,
    on_quantile_done: callable | None = None,
    feature_set: str = "full",
) -> np.ndarray:
    """Trains one LightGBM quantile regressor per quantile level on
    train_df and predicts test_df.

    feature_set:
    - "full": calendar + temperature + load-lag features, predicted one
      day at a time (see _predict_recursive) so load-lag features never
      use real data from the period being forecast.
    - "minimal": calendar + temperature only, predicted in one batch
      (no recursion needed since nothing depends on generated data) --
      a faster baseline to check whether the "full" feature set actually
      helps.

    Returns an array of shape (len(test_df), len(quantile_levels)).

    If given, on_quantile_done(index, total) is called after each of the
    `len(quantile_levels)` models finishes training, for progress reporting.
    """
    if feature_set not in FEATURE_SETS:
        raise ValueError(f"feature_set must be one of {list(FEATURE_SETS)}, got {feature_set!r}")
    if feature_set == "full" and 0.5 not in quantile_levels:
        raise ValueError(
            "quantile_levels must include 0.5 -- the recursive day-by-day "
            "prediction feeds the median forecast forward as the next "
            "day's 'known' load for lag features."
        )
    model_params = model_params or {}
    feature_columns = FEATURE_SETS[feature_set]

    train_feat = _select_temperature_columns(train_df, use_actual_future_temperature)
    if feature_set == "full":
        # train_df is genuine, gap-free historical data, so a plain
        # vectorized shift/rolling is leakage-safe here.
        train_feat = add_load_lag_features(train_feat)

    train_mask = train_feat[feature_columns + ["load"]].notna().all(axis=1)
    X_train = train_feat.loc[train_mask, feature_columns]
    y_train = train_feat.loc[train_mask, "load"]

    models = {}
    for i, q in enumerate(quantile_levels):
        model = LGBMRegressor(objective="quantile", alpha=q, verbosity=-1, **model_params)
        model.fit(X_train, y_train)
        models[q] = model
        if on_quantile_done is not None:
            on_quantile_done(i + 1, len(quantile_levels))

    if feature_set == "minimal":
        return _predict_batch(models, quantile_levels, test_df, feature_columns, use_actual_future_temperature)
    return _predict_recursive(models, quantile_levels, train_df, test_df, feature_columns, use_actual_future_temperature)


def _predict_batch(
    models: dict[float, LGBMRegressor],
    quantile_levels: list[float],
    test_df: pd.DataFrame,
    feature_columns: list[str],
    use_actual_future_temperature: bool,
) -> np.ndarray:
    """Predicts the whole test set in one shot -- only valid for feature
    sets with no dependency on recursively-generated data (e.g. "minimal",
    calendar + temperature only)."""
    test_feat = _select_temperature_columns(test_df, use_actual_future_temperature)
    X_test = test_feat[feature_columns]

    preds = np.zeros((len(test_df), len(quantile_levels)))
    for i, q in enumerate(quantile_levels):
        preds[:, i] = models[q].predict(X_test)
    return _enforce_monotonic_quantiles(preds, quantile_levels)


def _predict_recursive(
    models: dict[float, LGBMRegressor],
    quantile_levels: list[float],
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_columns: list[str],
    use_actual_future_temperature: bool,
) -> np.ndarray:
    """Forecasts test_df one calendar day at a time. Each day's
    load_lag_24h/168h and rolling mean/std are built from `known_load` --
    initialized to train_df's real load and, after each day is predicted,
    extended with THAT day's own median (0.5-quantile) prediction. Since
    lag_24h/168h always reference a day strictly before the one currently
    being predicted, and the rolling window is anchored at the end of the
    previous day, no feature for day D is ever built from real data
    belonging to day D or later -- only real pre-cutoff history and the
    model's own earlier-day predictions.

    Each day's raw predictions are also passed through
    _enforce_monotonic_quantiles before being stored or fed forward, so
    the "known" load used for later days is the crossing-corrected
    median, not the raw (possibly inconsistent) one.
    """
    known_load = train_df.set_index("timestamp")["load"].sort_index()
    known_load = known_load[~known_load.index.duplicated(keep="last")]

    test_sorted = test_df.sort_values("timestamp").reset_index(drop=True)
    day_of = test_sorted["timestamp"].dt.normalize()
    test_days = day_of.unique()
    median_idx = quantile_levels.index(0.5)

    preds_by_ts: dict[pd.Timestamp, list[float]] = {}

    for day in test_days:
        day_df = test_sorted.loc[day_of == day]
        day_feat = _select_temperature_columns(day_df, use_actual_future_temperature)

        ts = day_feat["timestamp"]
        day_feat["load_lag_24h"] = (ts - pd.Timedelta(hours=24)).map(known_load)
        day_feat["load_lag_168h"] = (ts - pd.Timedelta(hours=168)).map(known_load)

        window_end = pd.Timestamp(day) - pd.Timedelta(hours=1)
        window_start = window_end - pd.Timedelta(hours=_ROLLING_WINDOW_HOURS - 1)
        window = known_load.loc[window_start:window_end]
        day_feat["load_rolling_mean_7d"] = window.mean() if len(window) else np.nan
        day_feat["load_rolling_std_7d"] = window.std() if len(window) else np.nan

        X_day = day_feat[feature_columns]

        day_preds_raw = np.column_stack([models[q].predict(X_day) for q in quantile_levels])
        day_preds = _enforce_monotonic_quantiles(day_preds_raw, quantile_levels)

        day_timestamps = day_feat["timestamp"].to_numpy()
        for i, ts_val in enumerate(day_timestamps):
            preds_by_ts[ts_val] = day_preds[i, :].tolist()

        # Feed this day's (corrected) median prediction forward as "known"
        # load for subsequent days' lag/rolling features -- never the real value.
        median_series = pd.Series(day_preds[:, median_idx], index=day_timestamps)
        known_load = pd.concat([known_load, median_series]).sort_index()
        known_load = known_load[~known_load.index.duplicated(keep="last")]

    preds = np.zeros((len(test_df), len(quantile_levels)))
    for row_i, ts_val in enumerate(test_df["timestamp"].to_numpy()):
        preds[row_i, :] = preds_by_ts[ts_val]
    return preds
