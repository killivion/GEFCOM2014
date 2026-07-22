"""
Two simple, strong baselines every fancier model must beat:

1. seasonal_naive_quantiles: point forecast = load from the same hour one
   week earlier, with quantiles built from the empirical distribution of
   recent same-hour-of-week residuals.

2. climatology_quantiles: for each (hour-of-day, day-of-week) bucket,
   use the empirical quantiles of historical load in that bucket,
   ignoring temperature and recent trend entirely.
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


def seasonal_naive_quantiles(train_df: pd.DataFrame, test_df: pd.DataFrame, quantile_levels: list[float],
                              season_hours: int = 24 * 7) -> np.ndarray:
    """
    Point anchor = most recent same-hour-of-week observation available in
    training data. Spread = empirical quantiles of the (recent) residuals
    between actual load and its own lagged seasonal-naive prediction,
    added on top of the anchor. This gives a naive but genuinely
    probabilistic forecast, not just a repeated point value.
    """
    train = train_df.sort_values("timestamp").reset_index(drop=True)
    load_series = train.set_index("timestamp")["load"]

    # Residuals of seasonal-naive on the training set itself (in-sample,
    # but only using data available in `train_df`, so this is still
    # leakage-safe with respect to the test fold).
    shifted = load_series.shift(season_hours)
    residuals = (load_series - shifted).dropna()
    residual_quantiles = residuals.quantile(quantile_levels).values  # shape (n_q,)

    last_known = load_series

    preds = np.zeros((len(test_df), len(quantile_levels)))
    for i, ts in enumerate(test_df["timestamp"]):
        anchor_ts = ts - pd.Timedelta(hours=season_hours)
        if anchor_ts in last_known.index:
            anchor = last_known.loc[anchor_ts]
        else:
            anchor = last_known.iloc[-1]  # fallback: most recent known load
        preds[i, :] = anchor + residual_quantiles

    return preds
