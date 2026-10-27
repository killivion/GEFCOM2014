"""
LightGBM quantile regression: one gradient-boosted-tree regressor per
quantile level.

Every input is something that is genuinely known when a month-ahead
forecast is issued at the fold's cutoff: calendar features (precomputed
once, see build_features.py) and temperature. No feature depends on
load, so the whole test month is predicted in one batch.

Temperature, two ways:
- Temperature scenarios ("weather ensemble", Hong & Fan; used by top
  GEFCom2014 entries; config.model.temperature_scenarios, used by the
  final model): the models are trained on the training rows' ACTUAL
  observed temperatures (known history) and the test month is predicted
  under K temperature scenarios, each the observed temperature at the
  same month/day/hour of an EARLIER year (optionally date-shifted by a
  few days), read only from training rows strictly before the test
  month. The K predictive distributions are combined into one per hour
  (see combine_scenario_quantiles). Calendar features always stay those
  of the real test timestamps; only temp_mean / heating_degrees /
  cooling_degrees are swapped. K varies by fold (more earlier years are
  available for later folds).
- Otherwise one point input chosen by `temperature_source` --
  "climatology" (prior-years same hour / day-of-year mean), "last_year",
  or "actual" (the observed temperature of the forecast month itself:
  future information, used ONLY by the oracle upper bound). Training rows
  use the same variant as test rows.

Recency weighting (config.model.recency_half_life_years): each training
row gets LightGBM sample weight 0.5 ** (age / half_life), age in years
before the training cutoff, so recent years count more.

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

from src.features.build_features import HEATING_COOLING_BASE_TEMP, TEMPERATURE_VARIANTS

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


SCENARIO_COMBINE_METHODS = ("mixture", "quantile_average")

DEFAULT_TEMPERATURE_SCENARIOS = {"enabled": False, "shift_days": [0], "combine": "mixture", "max_years": None}


def build_temperature_scenarios(
    train_df: pd.DataFrame,
    test_timestamps: pd.Series,
    shift_days: list[int] = (0,),
    max_years: int | None = None,
) -> tuple[np.ndarray, list[tuple[int, int]]]:
    """Temperature scenarios for the test timestamps, from EARLIER years.

    Scenario (k, s) is, for every test timestamp t, the observed station-
    mean temperature (temp_mean_actual) at t - k years + s days, for
    k = 1, 2, ... (at most max_years) and every s in shift_days. The
    source temperatures are read ONLY from train_df rows strictly before
    the first test timestamp (the fold's training cutoff), so neither the
    test month's own temperature nor anything after the cutoff can enter.
    A scenario is kept only if every one of its timestamps is present in
    that history with a non-missing temperature ("its dates are fully
    inside the training data"); otherwise it is dropped whole, so e.g. a
    +4-day shift of last year never reaches into the test month.

    Feb 29 of a test year maps to Feb 28 of a non-leap source year (pandas
    DateOffset semantics).

    Returns (temperatures of shape (K, len(test_timestamps)), [(k, s), ...]).
    Raises ValueError if no scenario is fully inside the history.
    """
    test_ts = pd.to_datetime(pd.Series(test_timestamps)).reset_index(drop=True)
    cutoff = test_ts.min()

    history = train_df.loc[train_df["timestamp"] < cutoff, ["timestamp", "temp_mean_actual"]].dropna()
    history = history.drop_duplicates("timestamp", keep="last").set_index("timestamp")["temp_mean_actual"]
    if history.empty:
        raise ValueError("No training temperature history before the test period to build scenarios from")
    history_start = history.index.min()

    temps, labels = [], []
    k = 1
    while max_years is None or k <= max_years:
        base = test_ts - pd.DateOffset(years=k)
        if base.max() + pd.Timedelta(days=max(shift_days)) < history_start:
            break  # every shift of this and any earlier year is before the history
        for s in shift_days:
            source_ts = base + pd.Timedelta(days=s)
            if (source_ts >= cutoff).any():
                continue
            values = history.reindex(source_ts).to_numpy(dtype=float)
            if np.isnan(values).any():
                continue
            temps.append(values)
            labels.append((k, int(s)))
        k += 1

    if not temps:
        raise ValueError("No temperature scenario is fully inside the training history")
    return np.vstack(temps), labels


def _scenario_level_weights(quantile_levels: list[float]) -> np.ndarray:
    """Probability mass each quantile level stands for when its predicted
    value is treated as one draw of a pooled sample: the width of the
    interval between the midpoints to its neighbouring levels (0 and 1 at
    the ends). For an evenly spaced grid the interior weights are equal,
    so the mixture is plain pooling; for an uneven grid (e.g. the
    20-level dev grid, which skips 0.90) it stops densely spaced levels
    from being over-counted."""
    levels = np.asarray(quantile_levels, dtype=float)
    edges = np.concatenate([[0.0], (levels[1:] + levels[:-1]) / 2, [1.0]])
    return np.diff(edges)


def combine_scenario_quantiles(
    scenario_preds: np.ndarray, quantile_levels: list[float], method: str = "mixture"
) -> np.ndarray:
    """Combines per-scenario quantile predictions of shape
    (K, n_rows, n_quantiles) into one (n_rows, n_quantiles) array.

    - "mixture": the predictive distribution is the equal-weight mixture
      of the K scenario distributions. Per row, all K x n_quantiles
      predicted values are pooled (each weighted by the probability mass
      of its level, see _scenario_level_weights) and the weighted
      empirical quantiles (midpoint / Hazen positions) are read off at
      quantile_levels. Wider than any single scenario: it carries the
      weather uncertainty across scenarios.
    - "quantile_average": the mean, per level, of the K scenarios'
      predicted quantiles (Vincentization). Narrower than the mixture;
      ignores most of the between-scenario spread.

    Each scenario is rearranged to be monotone first; the result is
    monotone by construction, and the caller still applies the standard
    rearrangement.
    """
    if method not in SCENARIO_COMBINE_METHODS:
        raise ValueError(f"combine must be one of {SCENARIO_COMBINE_METHODS}, got {method!r}")
    scenario_preds = np.sort(np.asarray(scenario_preds, dtype=float), axis=2)
    n_scenarios, n_rows, n_q = scenario_preds.shape
    if method == "quantile_average":
        return scenario_preds.mean(axis=0)

    levels = np.asarray(quantile_levels, dtype=float)
    pooled = scenario_preds.transpose(1, 0, 2).reshape(n_rows, n_scenarios * n_q)
    weights = np.tile(_scenario_level_weights(quantile_levels), n_scenarios)
    order = np.argsort(pooled, axis=1, kind="stable")
    sorted_values = np.take_along_axis(pooled, order, axis=1)
    sorted_weights = weights[order]
    positions = (np.cumsum(sorted_weights, axis=1) - sorted_weights / 2) / sorted_weights.sum(axis=1, keepdims=True)

    # np.interp clamps outside the pooled range, so the extreme levels never
    # go beyond the most extreme scenario quantile.
    out = np.empty((n_rows, n_q))
    for r in range(n_rows):
        out[r] = np.interp(levels, positions[r], sorted_values[r])
    return out


def _with_scenario_temperature(test_feat: pd.DataFrame, temperature: np.ndarray) -> pd.DataFrame:
    """test_feat (real calendar features) with only the temperature-derived
    features replaced by one scenario's temperatures."""
    out = test_feat.copy()
    out["temp_mean"] = temperature
    out["heating_degrees"] = np.clip(HEATING_COOLING_BASE_TEMP - temperature, 0, None)
    out["cooling_degrees"] = np.clip(temperature - HEATING_COOLING_BASE_TEMP, 0, None)
    return out


def recency_weights(timestamps: pd.Series, cutoff: pd.Timestamp, half_life_years: float) -> np.ndarray:
    """0.5 ** (age / half_life), age = years between a row and the
    training cutoff (1.0 at the cutoff, halving every half_life_years)."""
    if half_life_years <= 0:
        raise ValueError("recency_half_life_years must be positive")
    age_years = (cutoff - timestamps).dt.total_seconds().to_numpy() / (365.25 * 24 * 3600)
    return 0.5 ** (age_years / half_life_years)


def lightgbm_quantiles(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    quantile_levels: list[float],
    temperature_source: str = "climatology",
    model_params: dict | None = None,
    on_quantile_done: callable | None = None,
    feature_set: str = "full",
    temperature_scenarios: dict | None = None,
    recency_half_life_years: float | None = None,
) -> np.ndarray:
    """Trains one LightGBM quantile regressor per quantile level on
    train_df and predicts test_df. test_df's own "load" column is dropped
    before any feature is built, so it can never be read.

    temperature_scenarios (config.model.temperature_scenarios; off when
    None or enabled is false): {enabled, shift_days, combine, max_years}.
    When enabled, training rows use their actual observed temperature,
    the test month is predicted under every scenario from
    build_temperature_scenarios(train_df, ..., shift_days, max_years) --
    test_df's temperature columns are never read -- and the scenarios are
    combined with combine_scenario_quantiles(..., combine).
    temperature_source is then ignored.

    recency_half_life_years (config.model.recency_half_life_years; None =
    off): training rows are weighted by recency_weights relative to the
    training cutoff (the last training timestamp).

    Returns an array of shape (len(test_df), len(quantile_levels)).

    If given, on_quantile_done(index, total) is called after each of the
    `len(quantile_levels)` models finishes training, for progress reporting.
    """
    if feature_set not in FEATURE_SETS:
        raise ValueError(f"feature_set must be one of {list(FEATURE_SETS)}, got {feature_set!r}")
    model_params = model_params or {}
    feature_columns = FEATURE_SETS[feature_set]
    scenarios = {**DEFAULT_TEMPERATURE_SCENARIOS, **(temperature_scenarios or {})}
    use_scenarios = bool(scenarios["enabled"])

    train_source = "actual" if use_scenarios else temperature_source
    train_feat = _select_temperature_columns(train_df, train_source)
    test_input = test_df.drop(columns=["load"], errors="ignore")

    if use_scenarios:
        if scenarios["combine"] not in SCENARIO_COMBINE_METHODS:
            raise ValueError(f"combine must be one of {SCENARIO_COMBINE_METHODS}, got {scenarios['combine']!r}")
        scenario_temps, _ = build_temperature_scenarios(
            train_df, test_input["timestamp"], shift_days=list(scenarios["shift_days"]),
            max_years=scenarios["max_years"],
        )
        # Calendar columns only (no temperature column of test_df at all);
        # the temperature features come from each scenario.
        calendar = test_input[[c for c in feature_columns if c not in ("temp_mean", "heating_degrees", "cooling_degrees")]]
        X_test = pd.concat(
            [_with_scenario_temperature(calendar, temps)[feature_columns] for temps in scenario_temps],
            ignore_index=True,
        )
    else:
        X_test = _select_temperature_columns(test_input, temperature_source)[feature_columns]

    train_mask = train_feat[feature_columns + ["load"]].notna().all(axis=1)
    X_train = train_feat.loc[train_mask, feature_columns]
    y_train = train_feat.loc[train_mask, "load"]
    sample_weight = None
    if recency_half_life_years is not None:
        sample_weight = recency_weights(train_feat.loc[train_mask, "timestamp"], train_df["timestamp"].max(),
                                        recency_half_life_years)

    preds = np.zeros((len(X_test), len(quantile_levels)))
    for i, q in enumerate(quantile_levels):
        model = LGBMRegressor(objective="quantile", alpha=q, verbosity=-1, **model_params)
        model.fit(X_train, y_train, sample_weight=sample_weight)
        preds[:, i] = model.predict(X_test)
        if on_quantile_done is not None:
            on_quantile_done(i + 1, len(quantile_levels))

    if use_scenarios:
        preds = combine_scenario_quantiles(
            preds.reshape(len(scenario_temps), len(test_df), len(quantile_levels)), quantile_levels,
            method=scenarios["combine"],
        )
    return _enforce_monotonic_quantiles(preds, quantile_levels)
