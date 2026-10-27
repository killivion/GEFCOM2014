"""
Post-processing of quantile forecasts, fitted strictly INSIDE each
backtest fold: nothing here ever sees the fold's test month.

Inner validation
----------------
The fold's training data is split by calendar month (hour-ending
timestamps, so 00:00 on the 1st belongs to the previous month, matching
the GEFCom task files). The inner-validation months are either the last
`n_months` months with observed load (scheme "recent") or the same
calendar month as the test month in each of the `n_months` previous years
(scheme "same_month": the error structure of a month-ahead load forecast
is strongly seasonal -- a shift learned on December cold spells does not
transfer to March). Each one is forecast exactly the way the
real test month is: every base forecaster is refit on the data strictly
BEFORE that month and predicts that month only (a rolling month-ahead
refit, not one fit with a longer horizon) -- so the inner-validation
errors have the same horizon and the same "most recent month unseen"
structure as the test-month errors they are used to correct. Cost: one
extra fit per inner month per base forecaster.

Post-processors fitted on those inner-validation predictions
------------------------------------------------------------
- additive quantile correction (conformalized-quantile style): per level
  tau, the shift c_tau = tau-quantile of (y - q_tau) on inner validation,
  i.e. the constant that minimises tau's pinball loss there; applied to
  the test month, then monotonic rearrangement (row-wise sort).
- recalibration map: the empirical calibration curve F(tau) = share of
  y <= q_tau on inner validation is inverted, so the test forecast for
  level tau is the raw forecast's quantile function read at the level
  tau' with F(tau') = tau. Interpolation / extrapolation is done in
  probit (normal-score) space, so levels beyond the outermost grid points
  can be reached with a Gaussian-like tail rather than a flat clip. Scale
  free, so it transfers across seasons better than additive shifts.
- quantile-averaging blend: w * model + (1 - w) * baseline at every level,
  w picked on a grid by inner-validation pinball loss.
- blend followed by the additive correction (both fitted on the same
  inner-validation months; the correction is fitted on the blended
  inner-validation forecasts).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd
from scipy.stats import norm

from src.evaluation.metrics import pinball_loss

Forecaster = Callable[[pd.DataFrame, pd.DataFrame], np.ndarray]

POSTPROCESS_METHODS = ("calibrated", "recalmap", "blend", "blend_calibrated")

_PROBIT_CLIP = 1e-4


def month_period(timestamps: pd.Series) -> pd.Series:
    """Calendar month of each hour-ending timestamp (00:00 on the 1st is
    the last hour of the previous month)."""
    return (pd.to_datetime(timestamps) - pd.Timedelta(hours=1)).dt.to_period("M")


INNER_VALIDATION_SCHEMES = ("recent", "same_month")


def inner_validation_months(train_df: pd.DataFrame, n_months: int, scheme: str = "recent") -> list[pd.Period]:
    """Inner-validation months of train_df, oldest first.

    "recent": the last `n_months` months with observed load.
    "same_month": the same calendar month as the month right after
        train_df (the fold's test month) in each of the `n_months`
        previous years -- so the correction is learned in the same season
        it is applied in. The test month itself is never among them.
    Every month returned has observed load and observed load before it."""
    if n_months < 1:
        raise ValueError("n_months must be >= 1")
    if scheme not in INNER_VALIDATION_SCHEMES:
        raise ValueError(f"scheme must be one of {INNER_VALIDATION_SCHEMES}, got {scheme!r}")
    observed = sorted(month_period(train_df.loc[train_df["load"].notna(), "timestamp"]).unique())
    if scheme == "recent":
        months = observed[-n_months:]
    else:
        target = month_period(train_df["timestamp"]).max() + 1
        months = sorted(target - 12 * k for k in range(1, n_months + 1))
    if len(observed) <= n_months or any(m not in observed for m in months) or months[0] <= observed[0]:
        raise ValueError(f"not enough observed history in train_df for {n_months} {scheme!r} inner months")
    return months


def inner_validation_predictions(
    train_df: pd.DataFrame, n_months: int, forecasters: dict[str, Forecaster], scheme: str = "recent"
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Rolling month-ahead forecasts of the last `n_months` months of
    train_df: for each such month, every forecaster is fit on the rows of
    train_df strictly before it and predicts that month's rows. Only
    train_df is ever touched.

    Returns (y_val, {name: predictions}) with hours with missing load
    dropped, concatenated in time order.
    """
    periods = month_period(train_df["timestamp"])
    y_parts, pred_parts = [], {name: [] for name in forecasters}
    for month in inner_validation_months(train_df, n_months, scheme):
        inner_train = train_df[periods < month]
        inner_val = train_df[periods == month]
        y = inner_val["load"].to_numpy(dtype=float)
        valid = ~np.isnan(y)
        y_parts.append(y[valid])
        for name, fn in forecasters.items():
            pred_parts[name].append(np.asarray(fn(inner_train, inner_val))[valid])
    return np.concatenate(y_parts), {name: np.concatenate(p, axis=0) for name, p in pred_parts.items()}


# --- additive correction -------------------------------------------------

def fit_additive_correction(y: np.ndarray, preds: np.ndarray, quantile_levels) -> np.ndarray:
    """Per level tau: the tau-quantile of the residuals y - q_tau, the
    pinball-optimal constant shift for that level. Shape (n_quantiles,)."""
    residuals = np.asarray(y, dtype=float).reshape(-1, 1) - np.asarray(preds, dtype=float)
    return np.array([np.quantile(residuals[:, j], tau) for j, tau in enumerate(quantile_levels)])


def apply_additive_correction(preds: np.ndarray, shifts: np.ndarray) -> np.ndarray:
    return np.sort(np.asarray(preds, dtype=float) + np.asarray(shifts, dtype=float).reshape(1, -1), axis=1)


# --- recalibration map ---------------------------------------------------

def _probit(p) -> np.ndarray:
    return norm.ppf(np.clip(np.asarray(p, dtype=float), _PROBIT_CLIP, 1 - _PROBIT_CLIP))


def _interp_extrap(x_new: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """np.interp with linear extrapolation from the two outermost points
    on each side instead of a flat clip. x must be strictly increasing."""
    out = np.interp(x_new, x, y)
    lo, hi = x_new < x[0], x_new > x[-1]
    if lo.any():
        slope = (y[1] - y[0]) / (x[1] - x[0])
        out[lo] = y[0] + slope * (x_new[lo] - x[0])
    if hi.any():
        slope = (y[-1] - y[-2]) / (x[-1] - x[-2])
        out[hi] = y[-1] + slope * (x_new[hi] - x[-1])
    return out


def fit_recalibration_map(y: np.ndarray, preds: np.ndarray, quantile_levels) -> np.ndarray:
    """Returns, for every nominal level tau, the raw-forecast level tau'
    whose empirical coverage on (y, preds) is tau. Shape (n_quantiles,).

    The empirical curve F(tau_j) is made strictly increasing (running max
    plus a tiny ramp) and clipped away from 0/1 by half an observation, so
    it can be inverted in probit space."""
    y = np.asarray(y, dtype=float).reshape(-1, 1)
    n = y.shape[0]
    empirical = (y <= np.asarray(preds, dtype=float)).mean(axis=0)
    empirical = np.clip(empirical, 0.5 / n, 1 - 0.5 / n)
    empirical = np.maximum.accumulate(empirical) + np.arange(len(empirical)) * 1e-9
    z_nominal = _probit(quantile_levels)
    # Direct ppf (no extra clip): empirical is already strictly inside (0, 1)
    # and strictly increasing, so its probit is too.
    z_new = _interp_extrap(z_nominal, norm.ppf(empirical), z_nominal)
    return norm.cdf(z_new)


def apply_recalibration_map(preds: np.ndarray, quantile_levels, mapped_levels: np.ndarray) -> np.ndarray:
    """Reads each row's forecast quantile function (piecewise linear in
    probit space, linearly extrapolated) at the mapped levels."""
    preds = np.sort(np.asarray(preds, dtype=float), axis=1)
    z = _probit(quantile_levels)
    z_new = _probit(mapped_levels)
    # Vectorised per-row interpolation on a shared x grid.
    idx = np.clip(np.searchsorted(z, z_new) - 1, 0, len(z) - 2)
    w = (z_new - z[idx]) / (z[idx + 1] - z[idx])
    out = preds[:, idx] + w * (preds[:, idx + 1] - preds[:, idx])
    return np.sort(out, axis=1)


# --- blend ---------------------------------------------------------------

def blend(preds_model: np.ndarray, preds_baseline: np.ndarray, weight: float) -> np.ndarray:
    """Quantile averaging: weight * model + (1 - weight) * baseline, per level."""
    return np.sort(weight * np.asarray(preds_model, dtype=float)
                   + (1 - weight) * np.asarray(preds_baseline, dtype=float), axis=1)


def fit_blend_weight(y, preds_model, preds_baseline, quantile_levels, grid) -> float:
    """Grid weight with the lowest pinball loss (ties -> first in grid)."""
    losses = [pinball_loss(y, blend(preds_model, preds_baseline, w), quantile_levels) for w in grid]
    return float(grid[int(np.argmin(losses))])


# --- fold-level wrapper --------------------------------------------------

@dataclass
class PostprocessResult:
    test_predictions: dict[str, np.ndarray]
    blend_weight: float | None = None
    additive_shifts: np.ndarray | None = None
    mapped_levels: np.ndarray | None = None
    extra: dict = field(default_factory=dict)


def postprocessed_forecasts(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    quantile_levels,
    model_fn: Forecaster,
    baseline_fn: Forecaster,
    n_months: int = 3,
    methods=POSTPROCESS_METHODS,
    blend_grid=None,
    scheme: str = "recent",
) -> PostprocessResult:
    """Everything for one fold. Returns test-month predictions keyed by
    "model", "baseline" (the raw base forecasts, fit on all of train_df)
    and each requested post-processing method.

    The post-processors are fitted only on inner_validation_predictions
    of train_df; test_df is used only to produce the final forecasts (and
    its "load" column is never read here -- model_fn/baseline_fn must not
    read it either, which is true of every forecaster in src/models)."""
    unknown = set(methods) - set(POSTPROCESS_METHODS)
    if unknown:
        raise ValueError(f"unknown postprocess methods {sorted(unknown)}; choose from {POSTPROCESS_METHODS}")
    if blend_grid is None:
        blend_grid = np.round(np.linspace(0, 1, 11), 10)
    test_df = test_df.drop(columns=["load"], errors="ignore")

    y_val, val = inner_validation_predictions(train_df, n_months, {"model": model_fn, "baseline": baseline_fn},
                                              scheme)
    test_model = np.asarray(model_fn(train_df, test_df))
    test_baseline = np.asarray(baseline_fn(train_df, test_df))

    result = PostprocessResult(test_predictions={"model": test_model, "baseline": test_baseline})
    out = result.test_predictions

    if "calibrated" in methods:
        result.additive_shifts = fit_additive_correction(y_val, val["model"], quantile_levels)
        out["calibrated"] = apply_additive_correction(test_model, result.additive_shifts)
    if "recalmap" in methods:
        result.mapped_levels = fit_recalibration_map(y_val, val["model"], quantile_levels)
        out["recalmap"] = apply_recalibration_map(test_model, quantile_levels, result.mapped_levels)
    if "blend" in methods or "blend_calibrated" in methods:
        w = fit_blend_weight(y_val, val["model"], val["baseline"], quantile_levels, blend_grid)
        result.blend_weight = w
        if "blend" in methods:
            out["blend"] = blend(test_model, test_baseline, w)
        if "blend_calibrated" in methods:
            shifts = fit_additive_correction(y_val, blend(val["model"], val["baseline"], w), quantile_levels)
            result.extra["blend_additive_shifts"] = shifts
            out["blend_calibrated"] = apply_additive_correction(blend(test_model, test_baseline, w), shifts)
    return result
