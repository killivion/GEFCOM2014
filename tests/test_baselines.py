import numpy as np
import pandas as pd
import pytest

from src.models.baselines import climatology_quantiles, seasonal_naive_quantiles


def test_climatology_quantiles_matches_bucket_empirical_quantile():
    # Only the Monday-00:00 bucket has data: loads 10, 20, 30 across three
    # different Mondays. A test row on a (different) Monday 00:00 should
    # get exactly that bucket's empirical quantile.
    start = pd.Timestamp("2021-01-04")  # a Monday
    train_df = pd.DataFrame({
        "timestamp": [start + pd.Timedelta(weeks=w) for w in range(3)],
        "load": [10.0, 20.0, 30.0],
    })
    test_df = pd.DataFrame({"timestamp": [start + pd.Timedelta(weeks=3)]})

    preds = climatology_quantiles(train_df, test_df, [0.5])

    assert preds[0, 0] == pytest.approx(np.median([10.0, 20.0, 30.0]))


def test_climatology_quantiles_falls_back_to_global_for_unseen_bucket():
    # Training data only has Monday-00:00 observations; the test row falls
    # in a different (hour, day-of-week) bucket never seen in training, so
    # the global-quantile fallback should kick in instead of crashing.
    start = pd.Timestamp("2021-01-04")
    train_df = pd.DataFrame({
        "timestamp": [start + pd.Timedelta(weeks=w) for w in range(3)],
        "load": [10.0, 20.0, 30.0],
    })
    test_df = pd.DataFrame({"timestamp": [start + pd.Timedelta(hours=5)]})  # Monday 05:00, unseen bucket

    preds = climatology_quantiles(train_df, test_df, [0.5])

    assert preds[0, 0] == pytest.approx(np.median([10.0, 20.0, 30.0]))


def test_seasonal_naive_tracks_weekly_pattern_beyond_first_test_week():
    # Perfectly periodic weekly pattern, no noise: load depends only on
    # hour-of-week and repeats identically every 7 days.
    def pattern(ts):
        return 10.0 * ts.hour + 100.0 * ts.dayofweek

    start = pd.Timestamp("2021-01-04")  # Monday 00:00
    n_train_hours = 24 * 7 * 4  # 4 full weeks
    train_df = pd.DataFrame({"timestamp": [start + pd.Timedelta(hours=h) for h in range(n_train_hours)]})
    train_df["load"] = train_df["timestamp"].map(pattern)

    # Test period is 3 full weeks right after training ends -- long enough
    # that most rows' "one week ago" falls inside the test period itself,
    # exercising the multi-week-lookback fix rather than the first week only.
    test_start = start + pd.Timedelta(hours=n_train_hours)
    n_test_hours = 24 * 7 * 3
    test_df = pd.DataFrame({"timestamp": [test_start + pd.Timedelta(hours=h) for h in range(n_test_hours)]})

    preds = seasonal_naive_quantiles(train_df, test_df, [0.5])

    expected = test_df["timestamp"].map(pattern).to_numpy()
    assert preds[:, 0] == pytest.approx(expected)

    # Regression guard: predictions late in the test period must still vary
    # with hour-of-week, not collapse to one flat value (the old bug's
    # fallback-to-last-known-value behavior for the un-anchorable tail of
    # each month).
    last_week = preds[-24 * 7:, 0]
    assert len(np.unique(last_week)) > 1


def test_seasonal_naive_raises_if_no_anchor_ever_available():
    # Training data too sparse/short to ever provide a same-hour-of-week
    # anchor for the test row -> should raise rather than silently return
    # a meaningless prediction.
    train_df = pd.DataFrame({
        "timestamp": pd.date_range("2021-01-04 00:00", periods=5, freq="h"),
        "load": [1.0, 2.0, 3.0, 4.0, 5.0],
    })
    # Offset by 7 hours so no multiple-of-168h lookback from this test
    # timestamp ever lands on one of the 5 known training hours.
    test_df = pd.DataFrame({"timestamp": [pd.Timestamp("2021-02-01 07:00")]})

    with pytest.raises(ValueError):
        seasonal_naive_quantiles(train_df, test_df, [0.5])
