"""
Core probabilistic forecasting metrics: pinball (quantile) loss and
interval coverage for calibration checks.
"""
from __future__ import annotations

import numpy as np


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
