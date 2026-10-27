"""
Feature engineering.

Calendar and temperature features (`add_calendar_features`,
`add_temperature_variants`) only ever look at a row's own timestamp or
at strictly EARLIER timestamps whose position relative to that row is
fixed (exactly one year earlier, or the same hour / day of year in
earlier operating-day years), so they're computed ONCE on the full
dataset in src/data/loader.py:get_data() -- the lookup is always
historical relative to that row regardless of which rolling-origin fold
the row ends up in. The one exception is the "actual" temperature
variant, which IS the row's own observed temperature: as a model input
for a forecast month it is future information. The final model only
uses it on TRAINING rows (with temperature scenarios for the test month,
see lightgbm_model.py); the oracle config uses it on test rows as a
labelled upper bound.

There are no load-derived features (see README, "Unsuccessful approaches").
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar

HEATING_COOLING_BASE_TEMP = 65.0

# Half-width (days) of the day-of-year window the temperature climatology
# averages over, so each value pools ~15 days x every earlier year rather
# than one noisy reading per year.
TEMP_CLIMATOLOGY_HALF_WINDOW_DAYS = 7

TEMPERATURE_VARIANTS = ("actual", "last_year", "climatology")


def operating_day_timestamps(timestamps: pd.Series) -> pd.Series:
    """Timestamps are hour-ENDING: the reading at 00:00 is the last hour
    (hour 24) of the PREVIOUS day. Shifting back one hour puts every
    reading on the day, weekday and month it actually belongs to; all
    day-level calendar keys (here and in the baselines) are taken from
    this, while the hour of day keeps the raw 0..23 value (0 = hour 24)."""
    return timestamps - pd.Timedelta(hours=1)


def add_calendar_features(df: pd.DataFrame, timestamp_col: str = "timestamp") -> pd.DataFrame:
    out = df.copy()
    ts = out[timestamp_col]
    day = operating_day_timestamps(ts)
    out["hour"] = ts.dt.hour
    out["dayofweek"] = day.dt.dayofweek
    out["month"] = day.dt.month
    out["dayofyear"] = day.dt.dayofyear
    out["is_weekend"] = out["dayofweek"].isin([5, 6]).astype(int)

    # Cyclical encodings so hour 23 / hour 0 (and Sun/Mon) read as
    # adjacent rather than maximally far apart.
    out["hour_sin"] = np.sin(2 * np.pi * out["hour"] / 24)
    out["hour_cos"] = np.cos(2 * np.pi * out["hour"] / 24)
    out["dayofweek_sin"] = np.sin(2 * np.pi * out["dayofweek"] / 7)
    out["dayofweek_cos"] = np.cos(2 * np.pi * out["dayofweek"] / 7)

    holidays = USFederalHolidayCalendar().holidays(start=day.min(), end=day.max())
    out["is_holiday"] = day.dt.normalize().isin(holidays).astype(int)

    return out


def noleap_dayofyear(day: pd.Series) -> pd.Series:
    """Day of year on a 365-day calendar: 29 Feb shares 28 Feb's value and
    every later day of a leap year is shifted back by one, so the same
    calendar date gets the same key in every year."""
    doy = day.dt.dayofyear
    return doy - (day.dt.is_leap_year & (doy > 59)).astype(int)


def prior_years_temperature_climatology(
    timestamps: pd.Series, temperature: pd.Series, half_window_days: int = TEMP_CLIMATOLOGY_HALF_WINDOW_DAYS
) -> pd.Series:
    """For each row in year Y: the mean temperature at the same hour of
    day, within +/- half_window_days of the same (365-day) day of year,
    averaged over every year strictly before Y. Year and day are those of
    the operating day (hour-ending timestamps: 1 Jan 00:00 is the last hour
    of 31 Dec), so a year's key never borrows a reading from the next year.
    Only ever reads earlier years, so it is known in advance for any
    forecast month. Rows in the first year of data have no earlier year and
    get NaN.
    """
    day = operating_day_timestamps(timestamps)
    frame = pd.DataFrame({
        "year": day.dt.year.to_numpy(),
        "hour": timestamps.dt.hour.to_numpy(),
        "doy": noleap_dayofyear(day).to_numpy(),
        "temp": temperature.to_numpy(dtype=float),
    })

    # (year, hour) x day-of-year table, smoothed along day of year within
    # each year (no wrap into the neighbouring year, so a year's values
    # never borrow from a later year).
    table = frame.groupby(["year", "hour", "doy"])["temp"].mean().unstack("doy")
    table = table.reindex(columns=range(1, 366))
    smoothed = table.T.rolling(2 * half_window_days + 1, center=True, min_periods=1).mean().T

    years = sorted(frame["year"].unique())
    per_year = []
    for year in years:
        prior = smoothed[smoothed.index.get_level_values("year") < year]
        if prior.empty:
            continue
        clim = prior.groupby(level="hour").mean()  # hour x doy, NaN-aware mean over earlier years
        clim = clim.stack().rename("temp_clim").reset_index()
        clim["year"] = year
        per_year.append(clim)

    if not per_year:
        return pd.Series(np.nan, index=timestamps.index)
    lookup = pd.concat(per_year, ignore_index=True).set_index(["year", "hour", "doy"])["temp_clim"]
    keys = pd.MultiIndex.from_arrays([frame["year"], frame["hour"], frame["doy"]])
    return pd.Series(lookup.reindex(keys).to_numpy(), index=timestamps.index)


def add_temperature_variants(df: pd.DataFrame, temp_cols: list[str], timestamp_col: str = "timestamp") -> pd.DataFrame:
    """Adds three temperature variants (see TEMPERATURE_VARIANTS), each
    with heating/cooling degree features:

    - "actual": the row's own observed station-mean temperature. For a
      forecast month this is future information -- only for the oracle
      upper bound, never the reported model.
    - "last_year": the same hour exactly one year earlier.
    - "climatology": the prior-years, same hour / day-of-year mean (see
      prior_years_temperature_climatology) -- the honest point input.

    Every variant is a per-row lookup that never depends on which fold a
    row ends up in, so this runs once on the full dataset; downstream code
    (lightgbm_model.py) selects whichever variant
    config.leakage.temperature_source names when temperature scenarios
    are off.

    Raises if any station reading is missing: the station mean would
    otherwise silently change composition hour to hour.
    """
    out = df.copy()
    if out[temp_cols].isna().any().any():
        raise ValueError("missing station temperatures -- the station mean would change composition")
    out["temp_mean_actual"] = out[temp_cols].mean(axis=1)

    history_series = out.set_index(timestamp_col)["temp_mean_actual"].sort_index()
    history_series = history_series[~history_series.index.duplicated(keep="last")]

    # No fill for the first year (nothing one year earlier exists): a
    # global-mean fill would read temperatures from the future.
    lookup_ts = out[timestamp_col] - pd.DateOffset(years=1)
    out["temp_mean_last_year"] = lookup_ts.map(history_series)

    out["temp_mean_climatology"] = prior_years_temperature_climatology(out[timestamp_col], out["temp_mean_actual"])

    for suffix in TEMPERATURE_VARIANTS:
        temp_col = f"temp_mean_{suffix}"
        out[f"heating_degrees_{suffix}"] = (HEATING_COOLING_BASE_TEMP - out[temp_col]).clip(lower=0)
        out[f"cooling_degrees_{suffix}"] = (out[temp_col] - HEATING_COOLING_BASE_TEMP).clip(lower=0)

    return out
