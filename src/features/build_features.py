"""
Feature engineering.

Two safety tiers, driven by how each feature is sourced:

1. Calendar and temperature features (`add_calendar_features`,
   `add_temperature_variants`) only ever look up a fixed point relative
   to a row's own timestamp (the row itself, or exactly one year
   earlier), so they're computed ONCE on the full dataset in
   src/data/loader.py:get_data() -- the lookup is always historical
   relative to that row regardless of which rolling-origin fold the row
   ends up in.

2. Load-derived features (lag_24h, lag_168h, rolling mean/std) depend on
   *which* load values are actually known at forecast time. For training
   rows that's trivial (their own real history, see
   `add_load_lag_features`). For a test fold's rows, "yesterday" or "last
   week" may fall INSIDE the very month being forecast, which isn't
   really known yet in the one-shot monthly forecast setting.
   src/models/lightgbm_model.py handles this by forecasting the test
   month one day at a time and feeding each day's own (median) prediction
   back in as "known" load for the next day's lag features -- these
   features are therefore NOT computed here on the full dataframe, and
   `add_load_lag_features` must only ever be applied to genuinely known
   (real, historical) data.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar

HEATING_COOLING_BASE_TEMP = 65.0


def add_calendar_features(df: pd.DataFrame, timestamp_col: str = "timestamp") -> pd.DataFrame:
    out = df.copy()
    ts = out[timestamp_col]
    out["hour"] = ts.dt.hour
    out["dayofweek"] = ts.dt.dayofweek
    out["month"] = ts.dt.month
    out["dayofyear"] = ts.dt.dayofyear
    out["is_weekend"] = out["dayofweek"].isin([5, 6]).astype(int)

    # Cyclical encodings so hour 23 / hour 0 (and Sun/Mon) read as
    # adjacent rather than maximally far apart.
    out["hour_sin"] = np.sin(2 * np.pi * out["hour"] / 24)
    out["hour_cos"] = np.cos(2 * np.pi * out["hour"] / 24)
    out["dayofweek_sin"] = np.sin(2 * np.pi * out["dayofweek"] / 7)
    out["dayofweek_cos"] = np.cos(2 * np.pi * out["dayofweek"] / 7)

    holidays = USFederalHolidayCalendar().holidays(start=ts.min(), end=ts.max())
    out["is_holiday"] = ts.dt.normalize().isin(holidays).astype(int)

    return out


def add_temperature_variants(df: pd.DataFrame, temp_cols: list[str], timestamp_col: str = "timestamp") -> pd.DataFrame:
    """Adds BOTH a real ("actual") and a leakage-free ("last_year")
    temperature feature, plus heating/cooling degree-day features derived
    from each. Both variants are per-row lookups (a row's own timestamp,
    or exactly one year earlier) that never depend on which fold a row
    ends up in, so this can run once on the full dataset; downstream code
    (lightgbm_model.py) selects whichever variant matches
    config.leakage.use_actual_future_temperature.
    """
    out = df.copy()
    out["temp_mean_actual"] = out[temp_cols].mean(axis=1)

    history_series = out.set_index(timestamp_col)["temp_mean_actual"].sort_index()
    history_series = history_series[~history_series.index.duplicated(keep="last")]

    lookup_ts = out[timestamp_col] - pd.DateOffset(years=1)
    out["temp_mean_last_year"] = lookup_ts.map(history_series)
    out["temp_mean_last_year"] = out["temp_mean_last_year"].fillna(history_series.mean())

    for suffix in ("actual", "last_year"):
        temp_col = f"temp_mean_{suffix}"
        out[f"heating_degrees_{suffix}"] = (HEATING_COOLING_BASE_TEMP - out[temp_col]).clip(lower=0)
        out[f"cooling_degrees_{suffix}"] = (out[temp_col] - HEATING_COOLING_BASE_TEMP).clip(lower=0)

    return out


def add_load_lag_features(df: pd.DataFrame, timestamp_col: str = "timestamp") -> pd.DataFrame:
    """Vectorized lag/rolling load features for a SINGLE, fully-known,
    continuous-hourly historical dataframe (e.g. a fold's train_df, which
    is genuine past data with no gaps).

    NOT safe to apply directly to a test fold's rows -- a naive shift
    would pull the real, not-yet-known load from later in the same
    forecast month for most of it. See lightgbm_model.py's recursive
    day-by-day prediction for how test-time lag features are built
    instead, using only real history plus each earlier day's own
    generated prediction.
    """
    out = df.sort_values(timestamp_col).reset_index(drop=True)
    out["load_lag_24h"] = out["load"].shift(24)
    out["load_lag_168h"] = out["load"].shift(168)
    out["load_rolling_mean_7d"] = out["load"].shift(1).rolling(168).mean()
    out["load_rolling_std_7d"] = out["load"].shift(1).rolling(168).std()
    return out
