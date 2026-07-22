"""
Verifies the quantile-crossing fix: since each quantile level is trained
as an independent LightGBM model (see lightgbm_model.py), nothing
otherwise guarantees pred(tau=0.05) <= pred(tau=0.10) <= ... for a given
row. _enforce_monotonic_quantiles (the rearrangement fix) must always
produce non-decreasing predictions across quantile levels, for both the
batch ("minimal") and recursive ("full") prediction paths.
"""
import numpy as np
import pytest

from src.models.lightgbm_model import _enforce_monotonic_quantiles, lightgbm_quantiles
from tests.test_lightgbm_no_leakage import _FAST_PARAMS, _synthetic_dataset


def test_enforce_monotonic_quantiles_sorts_each_row():
    quantile_levels = [0.1, 0.5, 0.9]
    # Row 0 already sorted; row 1 has a crossing (0.5 > 0.9's value).
    preds = np.array([
        [1.0, 2.0, 3.0],
        [1.0, 5.0, 4.0],
    ])
    fixed = _enforce_monotonic_quantiles(preds, quantile_levels)
    assert (np.diff(fixed, axis=1) >= 0).all()
    # Row 0 unaffected since it was already monotonic.
    np.testing.assert_array_equal(fixed[0], [1.0, 2.0, 3.0])
    # Row 1: values are rearranged (sorted), not clipped -- same
    # multiset of values, just reordered.
    np.testing.assert_array_equal(sorted(fixed[1]), sorted(preds[1]))


def test_enforce_monotonic_quantiles_raises_if_quantile_levels_not_sorted():
    preds = np.array([[1.0, 2.0, 3.0]])
    with pytest.raises(ValueError):
        _enforce_monotonic_quantiles(preds, [0.5, 0.1, 0.9])


def test_minimal_feature_predictions_are_always_monotonic():
    train_df, test_df = _synthetic_dataset()
    quantile_levels = [0.05, 0.25, 0.5, 0.75, 0.95]
    preds = lightgbm_quantiles(train_df, test_df, quantile_levels, model_params=_FAST_PARAMS, feature_set="minimal")
    assert (np.diff(preds, axis=1) >= 0).all()


def test_full_feature_predictions_are_always_monotonic():
    train_df, test_df = _synthetic_dataset()
    quantile_levels = [0.05, 0.25, 0.5, 0.75, 0.95]
    preds = lightgbm_quantiles(train_df, test_df, quantile_levels, model_params=_FAST_PARAMS, feature_set="full")
    assert (np.diff(preds, axis=1) >= 0).all()
