"""
Verifies the sequential conformal recalibration: a test month is only
ever adjusted from EARLIER months (changing the current or a later
month's observations cannot change it), the first month is untouched,
the quantiles stay monotone, and each method removes a known bias.
"""
import numpy as np
import pytest
from scipy.special import erfinv

from src.models.conformal import CONFORMAL_METHODS, sequential_recalibration
from src.evaluation.metrics import calibration_curve, pinball_loss

LEVELS = [round(0.01 * i, 2) for i in range(1, 100)]
Z = np.sqrt(2) * erfinv(2 * np.array(LEVELS) - 1)  # standard normal quantiles


def _biased_forecasts(n_months=6, hours=720, bias=-3.0, seed=0):
    """Forecasts of N(100, 10) that are too low by `bias` and too narrow."""
    rng = np.random.default_rng(seed)
    y = 100 + 10 * rng.standard_normal(n_months * hours)
    preds = np.tile(100 + bias + 7 * Z, (len(y), 1))
    test_task = np.repeat(np.arange(2, 2 + n_months), hours)
    return y, preds, test_task


@pytest.mark.parametrize("method", CONFORMAL_METHODS)
def test_first_month_unchanged_and_monotone(method):
    y, preds, task = _biased_forecasts()
    out = sequential_recalibration(y, preds, task, LEVELS, method)
    first = task == task.min()
    np.testing.assert_array_equal(out[first], preds[first])
    assert (np.diff(out, axis=1) >= -1e-12).all()


@pytest.mark.parametrize("method", CONFORMAL_METHODS)
@pytest.mark.parametrize("window", [None, 2])
def test_uses_only_earlier_months(method, window):
    y, preds, task = _biased_forecasts()
    out = sequential_recalibration(y, preds, task, LEVELS, method, window=window)
    month = 4
    y_changed = y.copy()
    y_changed[task >= month] += 50  # current and later observations
    out_changed = sequential_recalibration(y_changed, preds, task, LEVELS, method, window=window)
    np.testing.assert_array_equal(out[task <= month], out_changed[task <= month])


@pytest.mark.parametrize("method", CONFORMAL_METHODS)
def test_removes_a_known_bias(method):
    y, preds, task = _biased_forecasts()
    out = sequential_recalibration(y, preds, task, LEVELS, method)
    later = task > task.min()
    assert pinball_loss(y[later], out[later], LEVELS) < pinball_loss(y[later], preds[later], LEVELS)
    curve = calibration_curve(y[later], out[later], LEVELS)
    assert np.mean(np.abs(np.asarray(curve["empirical"]) - np.asarray(LEVELS))) < 0.02


def test_window_limits_the_calibration_months():
    y, preds, task = _biased_forecasts()
    y_changed = y.copy()
    y_changed[task == 2] += 50  # month 2 is outside a 2-month window for month 5
    a = sequential_recalibration(y, preds, task, LEVELS, "additive", window=2)
    b = sequential_recalibration(y_changed, preds, task, LEVELS, "additive", window=2)
    np.testing.assert_array_equal(a[task == 5], b[task == 5])
    assert not np.array_equal(a[task == 3], b[task == 3])


def test_unknown_method_raises():
    y, preds, task = _biased_forecasts()
    with pytest.raises(ValueError):
        sequential_recalibration(y, preds, task, LEVELS, "isotonic")
