import numpy as np
import pytest

from src.evaluation.metrics import pinball_loss, interval_coverage, calibration_curve


def test_pinball_loss_zero_for_perfect_forecast():
    y_true = np.array([10.0, 20.0, 30.0])
    quantile_levels = np.array([0.1, 0.5, 0.9])
    # perfect forecast: every quantile prediction equals the true value
    y_pred = np.tile(y_true.reshape(-1, 1), (1, 3))
    assert pinball_loss(y_true, y_pred, quantile_levels) == pytest.approx(0.0)


def test_pinball_loss_matches_hand_computation():
    # single point, single quantile: tau=0.9, y_true=10, y_pred=8
    # underprediction (y_true > y_pred): loss = tau * (y_true - y_pred) = 0.9 * 2 = 1.8
    y_true = np.array([10.0])
    y_pred = np.array([[8.0]])
    quantile_levels = np.array([0.9])
    assert pinball_loss(y_true, y_pred, quantile_levels) == pytest.approx(1.8)

    # overprediction (y_true < y_pred): loss = (tau - 1) * (y_true - y_pred)
    # tau=0.1, y_true=10, y_pred=15 -> (0.1 - 1) * (10 - 15) = (-0.9)*(-5) = 4.5
    y_true2 = np.array([10.0])
    y_pred2 = np.array([[15.0]])
    quantile_levels2 = np.array([0.1])
    assert pinball_loss(y_true2, y_pred2, quantile_levels2) == pytest.approx(4.5)


def test_pinball_loss_shape_mismatch_raises():
    y_true = np.array([1.0, 2.0])
    y_pred = np.array([[1.0, 2.0, 3.0]])  # wrong number of rows
    quantile_levels = np.array([0.1, 0.5, 0.9])
    with pytest.raises(ValueError):
        pinball_loss(y_true, y_pred, quantile_levels)


def test_interval_coverage_all_inside():
    y_true = np.array([5.0, 6.0, 7.0])
    lower = np.array([0.0, 0.0, 0.0])
    upper = np.array([10.0, 10.0, 10.0])
    assert interval_coverage(y_true, lower, upper) == pytest.approx(1.0)


def test_interval_coverage_partial():
    y_true = np.array([5.0, 15.0])  # second point outside [0, 10]
    lower = np.array([0.0, 0.0])
    upper = np.array([10.0, 10.0])
    assert interval_coverage(y_true, lower, upper) == pytest.approx(0.5)


def test_calibration_curve_well_calibrated_uniform_data():
    # If y_true are draws from Uniform(0,1) and quantile predictions are
    # exactly the nominal quantiles of Uniform(0,1), empirical should
    # match nominal closely for a large enough sample.
    rng = np.random.default_rng(0)
    y_true = rng.uniform(0, 1, size=100_000)
    quantile_levels = np.array([0.1, 0.5, 0.9])
    y_pred = np.tile(quantile_levels.reshape(1, -1), (len(y_true), 1))
    result = calibration_curve(y_true, y_pred, quantile_levels)
    for nominal, empirical in zip(result["nominal"], result["empirical"]):
        assert abs(nominal - empirical) < 0.01
