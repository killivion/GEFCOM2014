"""
Simple baselines every fancier model must beat. All three only use the
fold's training data, and all three account for the time-of-day /
day-of-week pattern; the first two also account for the time of YEAR,
which matters for a month-ahead forecast (load in July and in April
differ far more than any weekday pattern):

1. seasonal_climatology_quantiles (the primary baseline): for each
   (month, hour-of-day, weekday/weekend) bucket, the empirical quantiles
   of historical load in that bucket.

2. seasonal_naive_quantiles: point anchor = load at the same hour 52
   weeks (364 days, so the weekday matches) earlier, with quantiles from
   the empirical distribution of 52-week-ahead residuals.

3. climatology_quantiles: for each (hour-of-day, day-of-week) bucket,
   the empirical quantiles of historical load, pooled over ALL months --
   kept only as a deliberately naive reference point, since it ignores
   seasonality entirely.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _bucket_key(ts: pd.Series) -> pd.Series:
    return ts.dt.hour.astype(str) + "_" + ts.dt.dayofweek.astype(str)


def climatology_quantiles(train_df: pd.DataFrame, test_df: pd.DataFrame, quantile_levels: list[float]) -> np.ndarray:
    """
    Pure climatology baseline: predicts the same set of quantiles for every
    test row in a given (hour, day-of-week) bucket, based only on training
    history in that bucket. No temperature, no trend.
    """
    train = train_df.copy()
    train["bucket"] = _bucket_key(train["timestamp"])

    bucket_quantiles = (
        train.groupby("bucket")["load"]
        .quantile(quantile_levels)
        .unstack(level=-1)
    )
    bucket_quantiles.columns = quantile_levels

    test_bucket = _bucket_key(test_df["timestamp"])
    global_fallback = train["load"].quantile(quantile_levels)

    preds = np.zeros((len(test_df), len(quantile_levels)))
    for i, b in enumerate(test_bucket):
        if b in bucket_quantiles.index:
            preds[i, :] = bucket_quantiles.loc[b].values
        else:
            preds[i, :] = global_fallback.values
    return preds


YEARLY_SEASON_HOURS = 24 * 364  # 52 weeks: same weekday, ~one year back


def _seasonal_bucket_key(ts: pd.Series) -> pd.Series:
    return ts.dt.month.astype(str) + "_" + ts.dt.hour.astype(str) + "_" + (ts.dt.dayofweek >= 5).astype(int).astype(str)


def seasonal_climatology_quantiles(train_df: pd.DataFrame, test_df: pd.DataFrame, quantile_levels: list[float]) -> np.ndarray:
    """
    Seasonal climatology: predicts, for every test row, the empirical
    quantiles of training load in its (month, hour, weekday/weekend)
    bucket -- "what usually happens at this hour, on this kind of day, at
    this time of year". Weekday/weekend rather than full day-of-week so
    each bucket pools ~20 days per year of history instead of ~4. Falls
    back to the global training quantiles for an unseen bucket.
    """
    train = train_df.dropna(subset=["load"]).copy()
    train["bucket"] = _seasonal_bucket_key(train["timestamp"])

    bucket_quantiles = train.groupby("bucket")["load"].quantile(quantile_levels).unstack(level=-1)
    bucket_quantiles.columns = quantile_levels
    global_fallback = train["load"].quantile(quantile_levels).to_numpy()

    preds = bucket_quantiles.reindex(_seasonal_bucket_key(test_df["timestamp"])).to_numpy()
    missing = np.isnan(preds).any(axis=1)
    preds[missing] = global_fallback
    return preds


def seasonal_naive_quantiles(train_df: pd.DataFrame, test_df: pd.DataFrame, quantile_levels: list[float],
                              season_hours: int = YEARLY_SEASON_HOURS) -> np.ndarray:
    """
    Point anchor = most recent observation available in training data that
    is a whole number of seasons (default: 52 weeks, so the same hour and
    weekday roughly one year earlier) before the test hour. Spread = empirical quantiles of the (recent) residuals
    between actual load and its own lagged seasonal-naive prediction,
    added on top of the anchor. This gives a naive but genuinely
    probabilistic forecast, not just a repeated point value.

    With the yearly default the anchor is always in training data for a
    month-ahead fold. With a shorter season (e.g. season_hours=168, one
    week) "one season ago" can fall inside the forecast month itself (not
    yet known); the anchor then keeps stepping back whole seasons until it
    lands on a timestamp that IS in training data, so it's still the same
    hour-of-day/day-of-week, just from further back.
    """
    train = train_df.sort_values("timestamp").reset_index(drop=True)
    load_series = train.set_index("timestamp")["load"]

    # Residuals of seasonal-naive on the training set itself (in-sample,
    # but only using data available in `train_df`, so this is still
    # leakage-safe with respect to the test fold).
    # Time-based (not positional) shift, so a gap in the hourly series
    # can't silently pair an hour with the wrong one.
    lagged_index = load_series.index - pd.Timedelta(hours=season_hours)
    shifted = pd.Series(load_series.reindex(lagged_index).to_numpy(), index=load_series.index)
    residuals = (load_series - shifted).dropna()
    residual_quantiles = residuals.quantile(quantile_levels).values  # shape (n_q,)

    known_index = load_series.dropna().index
    earliest_known = known_index.min()
    max_seasons_back = int(np.ceil((test_df["timestamp"].max() - earliest_known) / pd.Timedelta(hours=season_hours))) + 1

    preds = np.zeros((len(test_df), len(quantile_levels)))
    for i, ts in enumerate(test_df["timestamp"]):
        seasons_back = 1
        anchor_ts = ts - pd.Timedelta(hours=season_hours * seasons_back)
        while anchor_ts not in known_index:
            seasons_back += 1
            if seasons_back > max_seasons_back:
                raise ValueError(
                    f"No training observation a whole number of seasons before test "
                    f"timestamp {ts} within {max_seasons_back} seasons of lookback; "
                    f"training data does not extend far enough back."
                )
            anchor_ts = ts - pd.Timedelta(hours=season_hours * seasons_back)
        anchor = load_series.loc[anchor_ts]
        preds[i, :] = anchor + residual_quantiles

    return preds


BASELINES = {
    "seasonal_climatology": seasonal_climatology_quantiles,
    "seasonal_naive": seasonal_naive_quantiles,
    "climatology": climatology_quantiles,
}

PRIMARY_BASELINE = "seasonal_climatology"
