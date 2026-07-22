"""
Core probabilistic forecasting metrics: pinball (quantile) loss,
interval coverage for calibration checks, and the Diebold-Mariano test
for comparing two forecasters' per-fold loss series.
"""
from __future__ import annotations

import numpy as np
from scipy import stats


def pinball_loss(y_true: np.ndarray, y_pred_quantiles: np.ndarray, quantile_levels: np.ndarray) -> float:
    """
    Mean pinball loss across all observations and all quantile levels.

    y_true: shape (n,)
    y_pred_quantiles: shape (n, n_quantiles) — predicted value at each quantile level
    quantile_levels: shape (n_quantiles,) — the tau values, e.g. [0.01, ..., 0.99]

    Returns a single scalar (lower is better).
    """
    y_true = np.asarray(y_true, dtype=float).reshape(-1, 1)
    y_pred_quantiles = np.asarray(y_pred_quantiles, dtype=float)
    quantile_levels = np.asarray(quantile_levels, dtype=float).reshape(1, -1)

    if y_pred_quantiles.shape[0] != y_true.shape[0]:
        raise ValueError("y_true and y_pred_quantiles must have the same number of rows")
    if y_pred_quantiles.shape[1] != quantile_levels.shape[1]:
        raise ValueError("y_pred_quantiles columns must match number of quantile_levels")

    diff = y_true - y_pred_quantiles  # (n, n_quantiles)
    loss = np.maximum(quantile_levels * diff, (quantile_levels - 1) * diff)
    return float(loss.mean())


def pinball_loss_per_fold(y_true: np.ndarray, y_pred_quantiles: np.ndarray, quantile_levels: np.ndarray) -> float:
    """Same as pinball_loss but kept as a distinct name for clarity when
    called once per backtest fold (see evaluation/backtest.py)."""
    return pinball_loss(y_true, y_pred_quantiles, quantile_levels)


def interval_coverage(y_true: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> float:
    """
    Empirical coverage: fraction of y_true that falls within [lower, upper].
    Use with, e.g., the 5th/95th percentile predictions to check whether a
    nominal 90% interval actually contains ~90% of outcomes.
    """
    y_true = np.asarray(y_true, dtype=float)
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    inside = (y_true >= lower) & (y_true <= upper)
    return float(inside.mean())


def calibration_curve(y_true: np.ndarray, y_pred_quantiles: np.ndarray, quantile_levels: np.ndarray) -> dict:
    """
    For each nominal quantile level tau, computes the empirical fraction of
    y_true observations that fall at or below the predicted tau-quantile.
    A well-calibrated model has empirical fraction ≈ tau for every level.

    Returns dict: {"nominal": [...], "empirical": [...]}
    """
    y_true = np.asarray(y_true, dtype=float).reshape(-1, 1)
    y_pred_quantiles = np.asarray(y_pred_quantiles, dtype=float)
    quantile_levels = np.asarray(quantile_levels, dtype=float)

    empirical = (y_true <= y_pred_quantiles).mean(axis=0)
    return {"nominal": quantile_levels.tolist(), "empirical": empirical.tolist()}


def diebold_mariano_test(
    loss_a: np.ndarray, loss_b: np.ndarray, h: int = 1, small_sample_correction: bool = True
) -> tuple[float, float]:
    """
    Diebold-Mariano test comparing two forecasters' loss series, paired by
    fold (e.g. per-fold pinball loss for a model vs. a baseline). Tests
    H0: equal predictive accuracy (mean loss differential = 0) against a
    two-sided alternative.

    loss_a, loss_b: one loss value per fold, same length, same fold order.
    h: forecast horizon in "steps" between successive loss observations.
        Determines how many autocorrelation lags (h-1, Bartlett-weighted,
        Newey-West style) are folded into the long-run variance estimate.
        h=1 (default) assumes the loss differential has no serial
        correlation across folds -- appropriate here, since each
        GEFCom2014 fold is a distinct, non-overlapping forecast month
        rather than an overlapping multi-step-ahead horizon.
    small_sample_correction: applies the Harvey-Leybourne-Newbold (1997)
        correction and compares against a Student's t distribution
        (df=T-1) instead of the asymptotic standard normal -- recommended
        given the small number of folds typical here (e.g. 14).

    Returns (dm_stat, p_value). Positive dm_stat means `loss_a` has the
    higher (worse) mean loss, i.e. `loss_b` is more accurate on average.
    """
    d = np.asarray(loss_a, dtype=float) - np.asarray(loss_b, dtype=float)
    t = len(d)
    if t < 2:
        raise ValueError("Need at least 2 paired observations for the Diebold-Mariano test")
    d_bar = d.mean()

    # Long-run variance of the loss differential: sample variance (lag 0)
    # plus Bartlett-weighted autocovariance terms up to lag h-1.
    var_d = np.sum((d - d_bar) ** 2) / t
    for lag in range(1, h):
        gamma_lag = np.sum((d[lag:] - d_bar) * (d[:-lag] - d_bar)) / t
        weight = 1 - lag / h
        var_d += 2 * weight * gamma_lag

    if var_d <= 0:
        raise ValueError("Non-positive long-run variance estimate -- loss differential has no variation")

    dm_stat = d_bar / np.sqrt(var_d / t)

    if small_sample_correction:
        correction = np.sqrt((t + 1 - 2 * h + h * (h - 1) / t) / t)
        dm_stat *= correction
        p_value = 2 * (1 - stats.t.cdf(abs(dm_stat), df=t - 1))
    else:
        p_value = 2 * (1 - stats.norm.cdf(abs(dm_stat)))

    return float(dm_stat), float(p_value)
