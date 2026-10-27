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

Two optional, config-driven ways to handle load level / trend and data
recency (both OFF by default, which reproduces the plain model exactly):

- level_normalization: the target is load / L(row) and predictions are
  multiplied back by L, where L is a slowly varying load level known at
  the row's forecast origin (00:00 on day 1 of the row's calendar month;
  timestamps are hour-ending, so a month runs day-1 01:00 .. next month's
  day-1 00:00, and for a test fold the origin is the training cutoff). L
  is a NORMALIZER, never a feature, so the trees cannot split on it (the
  failure mode of the earlier origin-anchored level features). L only
  ever reads training load at or before the origin -- see load_level.
  "trailing_year": mean load over the 365 days ending at the origin.
  "same_month_last_year": mean load of the same calendar month one year
  before.
- recency_half_life_years: LightGBM sample weights 0.5 ** (age / h),
  with age measured from the training cutoff, so older years count less.
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


LEVEL_NORMALIZATIONS = (None, "trailing_year", "same_month_last_year")

# A level is only used when at least this fraction of its window's hours
# have observed load (strict, so a partial window cannot bias the mean
# seasonally); rows whose level is undefined (e.g. the first year of load
# history for "trailing_year") are dropped from training.
LEVEL_MIN_COVERAGE = 0.98


def forecast_origin(timestamps: pd.Series) -> pd.Series:
    """00:00 on day 1 of each row's calendar month. Timestamps are
    hour-ending, so the hour ending at day-1 00:00 belongs to the PREVIOUS
    month (and is itself known at that month's origin)."""
    return (timestamps - pd.Timedelta(hours=1)).dt.to_period("M").dt.start_time


def _level_window(origin: pd.Timestamp, kind: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    """(start, end] window of hour-ending timestamps L averages over; end
    is always <= origin."""
    if kind == "trailing_year":
        return origin - pd.Timedelta(days=365), origin
    if kind == "same_month_last_year":
        start = origin - pd.DateOffset(years=1)
        return start, start + pd.DateOffset(months=1)
    raise ValueError(f"level_normalization must be one of {LEVEL_NORMALIZATIONS}, got {kind!r}")


def load_level(timestamps: pd.Series, history: pd.DataFrame, kind: str,
               min_coverage: float = LEVEL_MIN_COVERAGE) -> np.ndarray:
    """L for each row of `timestamps`: mean of history["load"] over the
    row's level window, which ends at or before the row's forecast origin.
    `history` must be TRAINING data only (timestamp, load); nothing after
    each row's origin is ever read. NaN where coverage < min_coverage."""
    hist = history[["timestamp", "load"]].dropna().sort_values("timestamp")
    hist_ts = hist["timestamp"].to_numpy()
    cum = np.concatenate([[0.0], np.cumsum(hist["load"].to_numpy(dtype=float))])

    origins = forecast_origin(timestamps)
    level_by_origin = {}
    for origin in pd.DatetimeIndex(origins.unique()):
        start, end = _level_window(origin, kind)
        lo = np.searchsorted(hist_ts, np.datetime64(start), side="right")
        hi = np.searchsorted(hist_ts, np.datetime64(end), side="right")
        expected_hours = (end - start) / pd.Timedelta(hours=1)
        n = hi - lo
        level_by_origin[origin] = (cum[hi] - cum[lo]) / n if n >= min_coverage * expected_hours else np.nan
    return origins.map(pd.Series(level_by_origin, dtype=float)).to_numpy(dtype=float)


def recency_weights(timestamps: pd.Series, cutoff: pd.Timestamp, half_life_years: float) -> np.ndarray:
    """0.5 ** (age / half_life), age = years between a row and the
    training cutoff (1.0 at the cutoff, halving every half_life_years)."""
    if half_life_years <= 0:
        raise ValueError("recency_half_life_years must be positive")
    age_years = (cutoff - timestamps).dt.total_seconds().to_numpy() / (365.25 * 24 * 3600)
    return 0.5 ** (age_years / half_life_years)


def lightgbm_level_options(config: dict) -> dict:
    """The optional level/recency keyword arguments of lightgbm_quantiles
    from a config's `model` section (both absent => both off)."""
    model = config.get("model", {})
    return {
        "level_normalization": model.get("level_normalization"),
        "recency_half_life_years": model.get("recency_half_life_years"),
    }


def lightgbm_quantiles(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    quantile_levels: list[float],
    temperature_source: str = "climatology",
    model_params: dict | None = None,
    on_quantile_done: callable | None = None,
    feature_set: str = "full",
    level_normalization: str | None = None,
    recency_half_life_years: float | None = None,
) -> np.ndarray:
    """Trains one LightGBM quantile regressor per quantile level on
    train_df and predicts test_df. test_df's own "load" column is dropped
    before any feature is built, so it can never be read.

    level_normalization / recency_half_life_years: see module docstring
    (None = off). The level of every row, train or test, is computed from
    train_df's load only.

    Returns an array of shape (len(test_df), len(quantile_levels)).

    If given, on_quantile_done(index, total) is called after each of the
    `len(quantile_levels)` models finishes training, for progress reporting.
    """
    if feature_set not in FEATURE_SETS:
        raise ValueError(f"feature_set must be one of {list(FEATURE_SETS)}, got {feature_set!r}")
    if level_normalization not in LEVEL_NORMALIZATIONS:
        raise ValueError(f"level_normalization must be one of {LEVEL_NORMALIZATIONS}, got {level_normalization!r}")
    model_params = model_params or {}
    feature_columns = FEATURE_SETS[feature_set]

    train_feat = _select_temperature_columns(train_df, temperature_source)
    test_feat = _select_temperature_columns(test_df.drop(columns=["load"], errors="ignore"), temperature_source)
    cutoff = train_df["timestamp"].max()

    if level_normalization is not None:
        if (forecast_origin(test_feat["timestamp"]) > cutoff).any():
            raise ValueError("a test row's forecast origin lies after the training cutoff")
        train_level = load_level(train_feat["timestamp"], train_df, level_normalization)
        test_level = load_level(test_feat["timestamp"], train_df, level_normalization)
        if np.isnan(test_level).any():
            raise ValueError(f"not enough training load history for a {level_normalization!r} level")
    else:
        train_level = np.ones(len(train_feat))
        test_level = np.ones(len(test_feat))

    train_mask = (train_feat[feature_columns + ["load"]].notna().all(axis=1) & ~np.isnan(train_level)).to_numpy()
    X_train = train_feat.loc[train_mask, feature_columns]
    y_train = train_feat.loc[train_mask, "load"].to_numpy() / train_level[train_mask]
    X_test = test_feat[feature_columns]

    sample_weight = None
    if recency_half_life_years is not None:
        sample_weight = recency_weights(train_feat.loc[train_mask, "timestamp"], cutoff, recency_half_life_years)

    preds = np.zeros((len(test_df), len(quantile_levels)))
    for i, q in enumerate(quantile_levels):
        model = LGBMRegressor(objective="quantile", alpha=q, verbosity=-1, **model_params)
        model.fit(X_train, y_train, sample_weight=sample_weight)
        # L > 0, so scaling by it preserves the quantile order.
        preds[:, i] = model.predict(X_test) * test_level
        if on_quantile_done is not None:
            on_quantile_done(i + 1, len(quantile_levels))

    return _enforce_monotonic_quantiles(preds, quantile_levels)
