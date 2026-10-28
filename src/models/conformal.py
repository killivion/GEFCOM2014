"""
Sequential conformal recalibration of the model's quantile forecasts.

The backtest months follow each other, and once a month has been
forecast and observed, its forecast errors are known history. For test
month t, the model's own out-of-sample forecasts of the EARLIER test
months (each made from data before that month) are a calibration set:
if they show that, say, 44.8% of loads fell below the predicted median,
month t's median is moved up accordingly. Nothing from month t or later
is used; the first test month has no calibration set and is unchanged.

Three ways to adjust each level tau (all fitted on the calibration hours):
- "additive": add delta_tau, the empirical tau-quantile of the residuals
  y - q_tau (conformalized quantile regression, one level at a time).
- "scaled": the same with residuals divided by the hour's 90% interval
  width (q_0.95 - q_0.05), so the shift grows with the forecast's spread
  (winter months are both wider and more variable).
- "pit": quantile recalibration (Kuleshov et al., 2018): the calibration
  hours' PIT values u = F(y) give the level tau' whose empirical coverage
  is tau, and month t reads its forecast at tau' instead of tau.

With n >= 672 calibration hours, the finite-sample (n+1)/n conformal
correction is negligible and is left out. Outputs are sorted per row, so
the quantiles stay monotone.
"""
from __future__ import annotations

import numpy as np

CONFORMAL_METHODS = ("additive", "scaled", "pit")


def _interval_width(preds: np.ndarray, quantile_levels) -> np.ndarray:
    levels = np.asarray(quantile_levels)
    lo, hi = int(np.argmin(np.abs(levels - 0.05))), int(np.argmin(np.abs(levels - 0.95)))
    return np.maximum(preds[:, hi] - preds[:, lo], 1e-6)


def _pit(y: np.ndarray, preds: np.ndarray, quantile_levels) -> np.ndarray:
    """Predictive CDF at the observation, linear between the levels and
    clipped to the outermost levels outside them."""
    levels = np.asarray(quantile_levels, dtype=float)
    return np.array([np.interp(y_i, q_i, levels) for y_i, q_i in zip(y, preds)])


def fit_recalibration(y: np.ndarray, preds: np.ndarray, quantile_levels, method: str) -> np.ndarray:
    """Per-level adjustment fitted on calibration hours: delta_tau for
    "additive" / "scaled", the replacement level tau' for "pit"."""
    levels = np.asarray(quantile_levels, dtype=float)
    if method == "additive":
        residuals = y[:, None] - preds
    elif method == "scaled":
        residuals = (y[:, None] - preds) / _interval_width(preds, quantile_levels)[:, None]
    elif method == "pit":
        return np.quantile(_pit(y, preds, quantile_levels), levels)
    else:
        raise ValueError(f"method must be one of {CONFORMAL_METHODS}, got {method!r}")
    return np.array([np.quantile(residuals[:, j], tau) for j, tau in enumerate(levels)])


def apply_recalibration(preds: np.ndarray, quantile_levels, method: str, adjustment: np.ndarray) -> np.ndarray:
    levels = np.asarray(quantile_levels, dtype=float)
    if method == "additive":
        out = preds + adjustment[None, :]
    elif method == "scaled":
        out = preds + adjustment[None, :] * _interval_width(preds, quantile_levels)[:, None]
    elif method == "pit":
        out = np.array([np.interp(adjustment, levels, q_i) for q_i in preds])
    else:
        raise ValueError(f"method must be one of {CONFORMAL_METHODS}, got {method!r}")
    return np.sort(out, axis=1)


def sequential_recalibration(y: np.ndarray, preds: np.ndarray, test_task: np.ndarray, quantile_levels,
                             method: str, window: int | None = None) -> np.ndarray:
    """Recalibrates every test month from the earlier test months only.

    y, preds, test_task: one row per scored hour (all folds, any order).
    window: use only the last `window` earlier test months (None = all).
    """
    out = preds.copy()
    tasks = np.unique(test_task)
    for i, task in enumerate(tasks):
        earlier = tasks[:i] if window is None else tasks[max(0, i - window):i]
        if len(earlier) == 0:
            continue
        cal = np.isin(test_task, earlier)
        adjustment = fit_recalibration(y[cal], preds[cal], quantile_levels, method)
        rows = test_task == task
        out[rows] = apply_recalibration(preds[rows], quantile_levels, method, adjustment)
    return out
