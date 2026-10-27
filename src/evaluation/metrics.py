"""
Core probabilistic forecasting metrics: pinball (quantile) loss,
interval coverage for calibration checks, and the statistical tools used
to compare two forecasters' loss series -- a Diebold-Mariano test with a
Newey-West (HAC) variance, a circular block bootstrap and a cluster
(whole-month) bootstrap of the mean loss difference, and the Holm
correction for testing against several methods at once.
"""
from __future__ import annotations

import numpy as np
from scipy import stats


def pinball_loss_per_observation(
    y_true: np.ndarray, y_pred_quantiles: np.ndarray, quantile_levels: np.ndarray
) -> np.ndarray:
    """
    Pinball loss of each observation, averaged over its quantile levels.

    y_true: shape (n,)
    y_pred_quantiles: shape (n, n_quantiles) — predicted value at each quantile level
    quantile_levels: shape (n_quantiles,) — the tau values, e.g. [0.01, ..., 0.99]

    Returns shape (n,) -- the per-hour loss series the significance tests
    below operate on.
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
    return loss.mean(axis=1)


def pinball_loss(y_true: np.ndarray, y_pred_quantiles: np.ndarray, quantile_levels: np.ndarray) -> float:
    """
    Mean pinball loss across all observations and all quantile levels
    (same shapes as pinball_loss_per_observation). Lower is better.
    """
    return float(pinball_loss_per_observation(y_true, y_pred_quantiles, quantile_levels).mean())


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


def interval_coverage_at_quantiles(
    y_true: np.ndarray, y_pred_quantiles: np.ndarray, quantile_levels, lo: float = 0.05, hi: float = 0.95
) -> float:
    """Convenience wrapper around interval_coverage: picks the lo/hi
    quantile columns out of a full y_pred_quantiles array by their
    quantile level (e.g. lo=0.05, hi=0.95 for nominal 90% coverage)."""
    quantile_levels = list(quantile_levels)
    lo_idx = quantile_levels.index(lo)
    hi_idx = quantile_levels.index(hi)
    return interval_coverage(y_true, y_pred_quantiles[:, lo_idx], y_pred_quantiles[:, hi_idx])


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


def newey_west_variance(d: np.ndarray, max_lag: int) -> float:
    """Long-run variance of a series: sample variance plus Bartlett-weighted
    autocovariances up to `max_lag` (Newey-West). max_lag=0 is the plain
    sample variance (divisor T)."""
    d = np.asarray(d, dtype=float)
    t = len(d)
    centered = d - d.mean()
    var = np.sum(centered ** 2) / t
    for lag in range(1, min(max_lag, t - 1) + 1):
        gamma = np.sum(centered[lag:] * centered[:-lag]) / t
        var += 2 * (1 - lag / (max_lag + 1)) * gamma
    return float(var)


def diebold_mariano_test(loss_a: np.ndarray, loss_b: np.ndarray, max_lag: int = 0) -> tuple[float, float]:
    """
    Diebold-Mariano test of H0: equal expected loss, against a two-sided
    alternative, on two loss series paired in time.

    Intended for the per-HOUR pinball losses of two forecasters over the
    whole backtest (thousands of observations), not for a handful of
    per-fold averages -- on fold averages with max_lag=0 it reduces to a
    paired t-test and adds nothing. Hourly loss differentials are strongly
    autocorrelated (a cold week hurts every hour of it), so their variance
    is estimated with Newey-West over `max_lag` lags (e.g. 168 = one week);
    ignoring that would overstate significance. The statistic is compared
    to a Student-t with T-1 degrees of freedom.

    Returns (dm_stat, p_value). Positive dm_stat means `loss_a` has the
    higher (worse) mean loss, i.e. `loss_b` is more accurate on average.
    """
    d = np.asarray(loss_a, dtype=float) - np.asarray(loss_b, dtype=float)
    t = len(d)
    if t < 2:
        raise ValueError("Need at least 2 paired observations for the Diebold-Mariano test")
    var_d = newey_west_variance(d, max_lag)
    if var_d <= 0:
        raise ValueError("Non-positive long-run variance estimate -- loss differential has no variation")

    dm_stat = d.mean() / np.sqrt(var_d / t)
    p_value = 2 * (1 - stats.t.cdf(abs(dm_stat), df=t - 1))
    return float(dm_stat), float(p_value)


def _percentile_ci_and_p(means: np.ndarray, alpha: float) -> tuple[float, float, float]:
    """Percentile CI of bootstrap means, plus the two-sided bootstrap
    p-value for H0: mean = 0 (twice the smaller tail mass beyond 0)."""
    lower, upper = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    p_value = min(1.0, 2 * min((means <= 0).mean(), (means >= 0).mean()))
    return float(lower), float(upper), float(p_value)


def circular_block_bootstrap_ci(
    d: np.ndarray, block_length: int, n_bootstrap: int = 2000, alpha: float = 0.05, seed: int = 0
) -> tuple[float, float, float]:
    """Bootstrap of the mean of a time-ordered series by resampling
    contiguous blocks of `block_length` observations (so within-block
    autocorrelation is kept), wrapping around the end so every observation
    is equally likely to be drawn. Returns (ci_lower, ci_upper, p_value)."""
    d = np.asarray(d, dtype=float)
    t = len(d)
    block_length = max(1, min(block_length, t))
    n_blocks = int(np.ceil(t / block_length))
    rng = np.random.default_rng(seed)

    # Cumulative sums over the wrapped series let each block's sum be read
    # off in O(1).
    wrapped = np.concatenate([d, d[:block_length]])
    cumsum = np.concatenate([[0.0], np.cumsum(wrapped)])
    starts = rng.integers(0, t, size=(n_bootstrap, n_blocks))
    block_sums = cumsum[starts + block_length] - cumsum[starts]
    # The last block is truncated so every resample has exactly t points.
    last_len = t - (n_blocks - 1) * block_length
    last = starts[:, -1]
    block_sums[:, -1] = cumsum[last + last_len] - cumsum[last]
    means = block_sums.sum(axis=1) / t
    return _percentile_ci_and_p(means, alpha)


def cluster_bootstrap_ci(
    d: np.ndarray, clusters: np.ndarray, n_bootstrap: int = 2000, alpha: float = 0.05, seed: int = 0
) -> tuple[float, float, float]:
    """Bootstrap of the mean of d that resamples whole clusters (here: test
    months) with replacement, so ANY dependence within a month -- a
    month-long weather anomaly, a model mis-specified for that month -- is
    kept intact. More conservative than block bootstraps with blocks
    shorter than a month; with few clusters (e.g. 7 holdout months) the
    distribution is coarse. Returns (ci_lower, ci_upper, p_value)."""
    d = np.asarray(d, dtype=float)
    clusters = np.asarray(clusters)
    labels = np.unique(clusters)
    sums = np.array([d[clusters == c].sum() for c in labels])
    counts = np.array([(clusters == c).sum() for c in labels])
    rng = np.random.default_rng(seed)
    picks = rng.integers(0, len(labels), size=(n_bootstrap, len(labels)))
    means = sums[picks].sum(axis=1) / counts[picks].sum(axis=1)
    return _percentile_ci_and_p(means, alpha)


def holm_adjust(p_values) -> np.ndarray:
    """Holm-Bonferroni adjusted p-values (family-wise error control) for a
    set of simultaneous tests, e.g. one model against several baselines."""
    p = np.asarray(p_values, dtype=float)
    m = len(p)
    order = np.argsort(p)
    adjusted = np.empty(m)
    running_max = 0.0
    for rank, idx in enumerate(order):
        running_max = max(running_max, (m - rank) * p[idx])
        adjusted[idx] = min(1.0, running_max)
    return adjusted
