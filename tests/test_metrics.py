import numpy as np
import pytest

from src.evaluation.metrics import (
    calibration_curve,
    diebold_mariano_test,
    holm_adjust,
    interval_coverage,
    interval_coverage_at_quantiles,
    circular_block_bootstrap_ci,
    cluster_bootstrap_ci,
    newey_west_variance,
    pinball_loss,
    pinball_loss_per_observation,
)


def _ar1(n, phi, seed=0):
    rng = np.random.default_rng(seed)
    e = rng.normal(size=n)
    x = np.zeros(n)
    for i in range(1, n):
        x[i] = phi * x[i - 1] + e[i]
    return x


def test_pinball_loss_per_observation_mean_is_pinball_loss():
    y = np.array([10.0, 20.0])
    pred = np.array([[8.0, 12.0], [25.0, 15.0]])
    q = np.array([0.1, 0.9])
    per_obs = pinball_loss_per_observation(y, pred, q)
    assert per_obs.shape == (2,)
    assert per_obs.mean() == pytest.approx(pinball_loss(y, pred, q))


def test_newey_west_variance_lag0_is_plain_variance():
    d = np.array([1.0, 3.0, 2.0, 6.0])
    assert newey_west_variance(d, 0) == pytest.approx(np.var(d))


def test_diebold_mariano_larger_max_lag_shrinks_stat_for_autocorrelated_differential():
    # Positively autocorrelated differential: ignoring autocorrelation
    # (max_lag=0) overstates significance, so a larger lag gives smaller |DM|.
    d = _ar1(2000, 0.8) + 0.3
    loss_b = np.full(2000, 10.0)
    dm0, _ = diebold_mariano_test(loss_b + d, loss_b, max_lag=0)
    dm50, _ = diebold_mariano_test(loss_b + d, loss_b, max_lag=50)
    assert abs(dm50) < abs(dm0)


def test_holm_adjust_matches_hand_computation():
    # sorted p: 0.01*3=0.03, 0.03*2=0.06, 0.04*1=0.04 -> running max 0.06.
    np.testing.assert_allclose(holm_adjust([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06])
    # adjusted values are capped at 1
    assert holm_adjust([0.6, 0.9]).max() == pytest.approx(1.0)


def test_bootstrap_ci_contains_true_mean_for_iid_series():
    d = np.random.default_rng(1).normal(5.0, 1.0, 500)
    low, high, _ = circular_block_bootstrap_ci(d, block_length=5, n_bootstrap=500, seed=0)
    assert low < 5.0 < high
    assert low < d.mean() < high


def test_bootstrap_ci_wider_with_longer_blocks_on_autocorrelated_series():
    d = _ar1(2000, 0.9)
    lo1, hi1, _ = circular_block_bootstrap_ci(d, block_length=1, n_bootstrap=500, seed=0)
    lo2, hi2, _ = circular_block_bootstrap_ci(d, block_length=100, n_bootstrap=500, seed=0)
    assert (hi2 - lo2) > 1.5 * (hi1 - lo1)


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


def test_interval_coverage_at_quantiles_picks_correct_columns():
    # 3 quantile levels; only the 0.05/0.95 columns should be used for a
    # nominal-90% coverage check, regardless of what's in between.
    quantile_levels = [0.05, 0.5, 0.95]
    y_true = np.array([5.0, 15.0, 25.0])
    y_pred = np.array([
        [0.0, 999.0, 10.0],   # inside [0, 10]
        [0.0, 999.0, 10.0],   # outside [0, 10] (15 > 10)
        [20.0, 999.0, 30.0],  # inside [20, 30]
    ])
    coverage = interval_coverage_at_quantiles(y_true, y_pred, quantile_levels, lo=0.05, hi=0.95)
    assert coverage == pytest.approx(2 / 3)


def test_interval_coverage_at_quantiles_matches_manual_interval_coverage():
    quantile_levels = [0.1, 0.5, 0.9]
    rng = np.random.default_rng(0)
    y_true = rng.uniform(0, 10, size=50)
    y_pred = np.sort(rng.uniform(0, 10, size=(50, 3)), axis=1)

    expected = interval_coverage(y_true, y_pred[:, 0], y_pred[:, 2])
    actual = interval_coverage_at_quantiles(y_true, y_pred, quantile_levels, lo=0.1, hi=0.9)
    assert actual == pytest.approx(expected)


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


def test_diebold_mariano_detects_clear_difference():
    # loss_a is consistently and substantially worse (higher) than loss_b.
    loss_a = np.array([12, 8, 11, 9, 13, 7, 10, 12, 8, 11], dtype=float)
    loss_b = np.array([2, 3, 1, 4, 2, 3, 1, 2, 4, 1], dtype=float)
    dm_stat, p_value = diebold_mariano_test(loss_a, loss_b)
    assert dm_stat > 0  # positive => loss_a (first arg) is the worse one
    assert p_value < 0.01


def test_diebold_mariano_is_antisymmetric():
    loss_a = np.array([12, 8, 11, 9, 13, 7, 10, 12, 8, 11], dtype=float)
    loss_b = np.array([2, 3, 1, 4, 2, 3, 1, 2, 4, 1], dtype=float)
    dm_ab, p_ab = diebold_mariano_test(loss_a, loss_b)
    dm_ba, p_ba = diebold_mariano_test(loss_b, loss_a)
    assert dm_ab == pytest.approx(-dm_ba)
    assert p_ab == pytest.approx(p_ba)


def test_diebold_mariano_not_significant_for_pure_noise():
    rng = np.random.default_rng(0)
    loss_a = rng.normal(10, 1, 30)
    loss_b = rng.normal(10, 1, 30)
    _, p_value = diebold_mariano_test(loss_a, loss_b)
    assert p_value > 0.05


def test_diebold_mariano_raises_on_too_few_observations():
    with pytest.raises(ValueError):
        diebold_mariano_test([1.0], [2.0])


def test_diebold_mariano_raises_on_zero_variance_differential():
    # A perfectly constant loss differential has no variance to test
    # against -- should raise rather than divide by zero.
    loss_a = np.array([5.0, 6.0, 7.0, 8.0])
    loss_b = loss_a - 2.0
    with pytest.raises(ValueError):
        diebold_mariano_test(loss_a, loss_b)


def test_cluster_bootstrap_detects_clear_win_and_not_noise():
    rng = np.random.default_rng(0)
    clusters = np.repeat(np.arange(10), 50)
    low, high, p = cluster_bootstrap_ci(rng.normal(2.0, 1.0, 500), clusters, n_bootstrap=1000, seed=0)
    assert low > 0 and p < 0.01
    # A month-level effect that changes sign between months: the cluster
    # bootstrap should NOT call it significant even though hours are many.
    month_effect = np.repeat(rng.choice([-1.0, 1.0], 10), 50)
    _, _, p_noise = cluster_bootstrap_ci(month_effect + rng.normal(0, 0.1, 500), clusters, n_bootstrap=1000, seed=0)
    assert p_noise > 0.05


def test_circular_block_bootstrap_returns_p_value_in_unit_interval():
    d = np.random.default_rng(1).normal(0.5, 1.0, 300)
    low, high, p = circular_block_bootstrap_ci(d, block_length=10, n_bootstrap=500, seed=0)
    assert low < d.mean() < high
    assert 0.0 <= p <= 1.0
