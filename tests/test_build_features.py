"""
Tests for the temperature climatology and temperature variants in
src/features/build_features.py.
"""
import numpy as np
import pandas as pd
import pytest

from src.features.build_features import add_calendar_features, add_temperature_variants, prior_years_temperature_climatology


def _daily_noon(start, end):
    return pd.Series(pd.date_range(start, end, freq="D") + pd.Timedelta(hours=12))


def test_climatology_constant_per_year_and_first_year_nan():
    # Years 2019/2020/2021 have constant temps 10/20/30: 2019 has no
    # earlier year (NaN), 2020 sees only 2019 (10), 2021 sees mean(10, 20).
    ts = _daily_noon("2019-01-01", "2021-12-31")
    temp = pd.Series(ts.dt.year.map({2019: 10.0, 2020: 20.0, 2021: 30.0}).to_numpy())
    clim = prior_years_temperature_climatology(ts, temp)
    year = ts.dt.year.to_numpy()
    assert clim[year == 2019].isna().all()
    np.testing.assert_allclose(clim[year == 2020], 10.0)
    np.testing.assert_allclose(clim[year == 2021], 15.0)


def test_climatology_only_depends_on_earlier_years():
    # Perturbing year Y's (or later) temperatures must not change year Y's values.
    ts = _daily_noon("2019-01-01", "2021-12-31")
    temp = pd.Series(np.random.default_rng(0).normal(50, 10, len(ts)))
    base = prior_years_temperature_climatology(ts, temp)
    year = ts.dt.year.to_numpy()

    perturbed = temp.copy()
    perturbed[year >= 2021] += 100.0
    after = prior_years_temperature_climatology(ts, perturbed)
    np.testing.assert_allclose(base.to_numpy()[year <= 2020], after.to_numpy()[year <= 2020], equal_nan=True)

    perturbed2 = temp.copy()
    perturbed2[year == 2020] += 100.0
    after2 = prior_years_temperature_climatology(ts, perturbed2)
    np.testing.assert_allclose(base.to_numpy()[year <= 2020], after2.to_numpy()[year <= 2020], equal_nan=True)
    # ...while 2021 (which does look at 2020) does change.
    assert not np.allclose(base.to_numpy()[year == 2021], after2.to_numpy()[year == 2021])


def test_temperature_variants_last_year_is_nan_without_history_not_filled():
    ts = _daily_noon("2019-01-01", "2020-12-31")
    df = pd.DataFrame({"timestamp": ts, "t1": np.arange(len(ts), dtype=float)})
    df["t2"] = df["t1"] + 2.0
    out = add_temperature_variants(df, ["t1", "t2"])

    first_year = out["timestamp"].dt.year == 2019
    assert out.loc[first_year, "temp_mean_last_year"].isna().all()
    assert out.loc[first_year, "temp_mean_climatology"].isna().all()
    # 2020-06-15 -> value of 2019-06-15
    row = out[out["timestamp"] == pd.Timestamp("2020-06-15 12:00")].iloc[0]
    prev = out[out["timestamp"] == pd.Timestamp("2019-06-15 12:00")].iloc[0]
    assert row["temp_mean_last_year"] == pytest.approx(prev["temp_mean_actual"])
    assert row["temp_mean_actual"] == pytest.approx(row["t1"] + 1.0)
    for v in ("actual", "last_year", "climatology"):
        assert f"heating_degrees_{v}" in out.columns
        assert f"cooling_degrees_{v}" in out.columns


def test_heating_cooling_degrees_are_clipped_differences_to_65():
    df = pd.DataFrame({"timestamp": pd.to_datetime(["2020-01-01 01:00", "2020-01-01 02:00"]), "t1": [50.0, 80.0]})
    out = add_temperature_variants(df, ["t1"])
    assert out["heating_degrees_actual"].tolist() == [15.0, 0.0]
    assert out["cooling_degrees_actual"].tolist() == [0.0, 15.0]


def test_calendar_features_follow_hour_ending_convention():
    # 2010-11-01 00:00 is hour 24 of Sunday 2010-10-31: it belongs to
    # October and to a Sunday, not to Monday 1 November.
    df = pd.DataFrame({"timestamp": pd.to_datetime(["2010-11-01 00:00", "2010-11-01 01:00"])})
    out = add_calendar_features(df)
    assert out["month"].tolist() == [10, 11]
    assert out["dayofweek"].tolist() == [6, 0]
    assert out["is_weekend"].tolist() == [1, 0]
    assert out["hour"].tolist() == [0, 1]


def test_climatology_year_follows_the_operating_day():
    # 2021-01-01 00:00 is hour 24 of 31 Dec 2020: its climatology may only
    # use years before 2020, i.e. only 2019 here (constant 10).
    ts = pd.Series(pd.date_range("2019-01-01 01:00", "2021-01-01 00:00", freq="h"))
    day_year = (ts - pd.Timedelta(hours=1)).dt.year
    temp = pd.Series(day_year.map({2019: 10.0, 2020: 20.0}).to_numpy())
    clim = prior_years_temperature_climatology(ts, temp)
    assert clim.iloc[-1] == pytest.approx(10.0)


def test_noleap_dayofyear_maps_same_date_to_same_key():
    from src.features.build_features import noleap_dayofyear
    days = pd.Series(pd.to_datetime(["2011-03-01", "2012-03-01", "2012-02-29", "2012-02-28", "2012-12-31"]))
    assert noleap_dayofyear(days).tolist() == [60, 60, 59, 59, 365]
