"""
Tests for src/models/postprocess.py on synthetic data: the inner
validation never reads the fold's test month, recalibration widens a
deliberately too-narrow forecast, and the blend reproduces its components
at weight 1 / 0.
"""
import numpy as np
import pandas as pd
import pytest
import yaml
from scipy.stats import norm

from src.evaluation.metrics import interval_coverage_at_quantiles, pinball_loss
from src.evaluation.run_comparison import postprocess_method_name, postprocess_settings
from src.features.build_features import add_calendar_features
from src.models.baselines import seasonal_climatology_quantiles
from src.models.lightgbm_model import lightgbm_quantiles
from src.models.postprocess import (
    apply_additive_correction,
    apply_recalibration_map,
    blend,
    fit_additive_correction,
    fit_blend_weight,
    fit_recalibration_map,
    inner_validation_months,
    month_period,
    postprocessed_forecasts,
)

LEVELS = [0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]


def _hourly_frame(start, end, seed=0):
    """Hour-ending timestamps (start, end], like the GEFCom task files."""
    rng = np.random.default_rng(seed)
    ts = pd.date_range(start, end, freq="h")[1:]
    df = pd.DataFrame({"timestamp": ts})
    df["load"] = 100 + 10 * np.sin(np.arange(len(df)) * 2 * np.pi / 24) + rng.normal(0, 3, len(df))
    df = add_calendar_features(df)
    for variant in ("actual", "last_year", "climatology"):
        df[f"temp_mean_{variant}"] = 50.0
        df[f"heating_degrees_{variant}"] = 0.0
        df[f"cooling_degrees_{variant}"] = 0.0
    return df


@pytest.fixture
def fold():
    full = _hourly_frame("2020-01-01", "2020-08-01")
    test_start = pd.Timestamp("2020-07-01 01:00")
    return full[full["timestamp"] < test_start].copy(), full[full["timestamp"] >= test_start].copy()


def _empirical_forecaster(train_df, test_df):
    """Cheap stand-in model: hour-of-day empirical quantiles of train."""
    q = train_df.groupby(train_df["timestamp"].dt.hour)["load"].quantile(LEVELS).unstack()
    return q.reindex(test_df["timestamp"].dt.hour).to_numpy()


def _baseline(train_df, test_df):
    return seasonal_climatology_quantiles(train_df, test_df, LEVELS)


def test_month_period_is_hour_ending():
    ts = pd.Series(pd.to_datetime(["2020-02-01 00:00", "2020-02-01 01:00"]))
    assert list(month_period(ts).astype(str)) == ["2020-01", "2020-02"]


def test_inner_validation_months_are_last_months_of_train(fold):
    train_df, _ = fold
    assert [str(m) for m in inner_validation_months(train_df, 3)] == ["2020-04", "2020-05", "2020-06"]


def test_same_month_scheme_uses_test_calendar_month_of_earlier_years():
    full = _hourly_frame("2018-01-01", "2020-08-01")
    train_df = full[full["timestamp"] < pd.Timestamp("2020-07-01 01:00")]
    months = inner_validation_months(train_df, 2, scheme="same_month")
    assert [str(m) for m in months] == ["2018-07", "2019-07"]
    with pytest.raises(ValueError):
        inner_validation_months(train_df, 3, scheme="same_month")  # 2017-07 not in history


def test_same_month_scheme_never_sees_test_month():
    full = _hourly_frame("2018-01-01", "2020-08-01")
    test_start = pd.Timestamp("2020-07-01 01:00")
    train_df, test_df = full[full["timestamp"] < test_start].copy(), full[full["timestamp"] >= test_start].copy()
    seen = []

    def spy(tr, te):
        seen.append((tr["timestamp"].max(), te["timestamp"].min(), te["timestamp"].max()))
        return _empirical_forecaster(tr, te)

    original = postprocessed_forecasts(train_df, test_df, LEVELS, spy, _baseline, n_months=2, scheme="same_month")
    for train_max, val_min, val_max in seen:
        assert train_max < val_min
    assert sum(val_min < test_start for _, val_min, _ in seen) == 2
    corrupted_test = test_df.copy()
    corrupted_test["load"] = 1e9
    corrupted = postprocessed_forecasts(train_df, corrupted_test, LEVELS, _empirical_forecaster, _baseline,
                                        n_months=2, scheme="same_month")
    for name, preds in original.test_predictions.items():
        np.testing.assert_array_equal(preds, corrupted.test_predictions[name])


def test_inner_validation_never_sees_test_month(fold):
    train_df, test_df = fold
    seen = []

    def spy(name, fn):
        def wrapped(tr, te):
            seen.append((name, tr["timestamp"].max(), te["timestamp"].min(), te["timestamp"].max(),
                         "load" in te.columns and te["load"].notna().any()))
            return fn(tr, te)
        return wrapped

    postprocessed_forecasts(train_df, test_df, LEVELS, spy("m", _empirical_forecaster), spy("b", _baseline),
                            n_months=3)
    test_start = test_df["timestamp"].min()
    inner_calls = [s for s in seen if s[2] < test_start]
    final_calls = [s for s in seen if s[2] >= test_start]
    assert len(inner_calls) == 6 and len(final_calls) == 2
    for _, train_max, val_min, val_max, _ in inner_calls:
        assert train_max < val_min  # month-ahead: fit strictly before the validation month
        assert val_max < test_start
    for _, train_max, _, _, has_load in final_calls:
        assert train_max < test_start
        assert not has_load  # the test month's load is dropped before forecasting


def test_corrupting_test_load_changes_nothing(fold):
    train_df, test_df = fold
    original = postprocessed_forecasts(train_df, test_df, LEVELS, _empirical_forecaster, _baseline, n_months=2)
    corrupted_test = test_df.copy()
    corrupted_test["load"] = 1e9
    corrupted = postprocessed_forecasts(train_df, corrupted_test, LEVELS, _empirical_forecaster, _baseline,
                                        n_months=2)
    for name, preds in original.test_predictions.items():
        np.testing.assert_array_equal(preds, corrupted.test_predictions[name])


def test_corrupting_test_load_changes_nothing_with_real_lightgbm(fold):
    train_df, test_df = fold
    params = {"n_estimators": 10, "max_depth": 3, "learning_rate": 0.3, "num_leaves": 7, "n_jobs": 1}

    def model_fn(tr, te):
        return lightgbm_quantiles(tr, te, LEVELS, model_params=params)

    original = postprocessed_forecasts(train_df, test_df, LEVELS, model_fn, _baseline, n_months=2)
    corrupted_test = test_df.copy()
    corrupted_test["load"] = -1e9
    corrupted = postprocessed_forecasts(train_df, corrupted_test, LEVELS, model_fn, _baseline, n_months=2)
    for name, preds in original.test_predictions.items():
        np.testing.assert_allclose(preds, corrupted.test_predictions[name])


def test_inner_validation_load_does_influence_postprocessing(fold):
    # Sanity check that the invariance above is meaningful: corrupting the
    # last TRAINING month (inner validation) must change the calibrated forecast.
    train_df, test_df = fold
    original = postprocessed_forecasts(train_df, test_df, LEVELS, _empirical_forecaster, _baseline,
                                       n_months=2, methods=["calibrated"])
    shifted = train_df.copy()
    last_month = month_period(shifted["timestamp"]) == pd.Period("2020-06", "M")
    shifted.loc[last_month, "load"] += 50
    changed = postprocessed_forecasts(shifted, test_df, LEVELS, _empirical_forecaster, _baseline,
                                      n_months=2, methods=["calibrated"])
    assert not np.allclose(original.test_predictions["calibrated"], changed.test_predictions["calibrated"])


def _too_narrow_setup(n=4000, seed=1):
    rng = np.random.default_rng(seed)
    y_val, y_test = rng.normal(0, 2, n), rng.normal(0, 2, n)
    narrow = np.tile(norm.ppf(LEVELS), (n, 1))  # N(0, 1) quantiles for N(0, 2) data
    return y_val, y_test, narrow


@pytest.mark.parametrize("method", ["additive", "map"])
def test_recalibration_widens_too_narrow_forecast(method):
    y_val, y_test, narrow = _too_narrow_setup()
    if method == "additive":
        fixed = apply_additive_correction(narrow, fit_additive_correction(y_val, narrow, LEVELS))
    else:
        fixed = apply_recalibration_map(narrow, LEVELS, fit_recalibration_map(y_val, narrow, LEVELS))

    lo, hi = LEVELS.index(0.05), LEVELS.index(0.95)
    assert (fixed[:, hi] - fixed[:, lo]).mean() > 1.5 * (narrow[:, hi] - narrow[:, lo]).mean()
    assert interval_coverage_at_quantiles(y_test, narrow, LEVELS) < 0.65
    assert abs(interval_coverage_at_quantiles(y_test, fixed, LEVELS) - 0.90) < 0.03
    assert pinball_loss(y_test, fixed, LEVELS) < pinball_loss(y_test, narrow, LEVELS)
    assert np.all(np.diff(fixed, axis=1) >= 0)


def test_recalibration_map_is_near_identity_when_calibrated():
    rng = np.random.default_rng(2)
    y = rng.normal(0, 1, 20000)
    preds = np.tile(norm.ppf(LEVELS), (len(y), 1))
    mapped = fit_recalibration_map(y, preds, LEVELS)
    np.testing.assert_allclose(mapped, LEVELS, atol=0.01)
    np.testing.assert_allclose(apply_recalibration_map(preds, LEVELS, np.array(LEVELS)), preds, atol=1e-9)


def test_additive_correction_sorts_crossings():
    preds = np.array([[0.0, 1.0, 2.0]])
    out = apply_additive_correction(preds, np.array([3.0, 0.0, 0.0]))
    assert np.all(np.diff(out, axis=1) >= 0)


def test_blend_weight_one_and_zero_reproduce_components():
    rng = np.random.default_rng(3)
    a = np.sort(rng.normal(size=(50, len(LEVELS))), axis=1)
    b = np.sort(rng.normal(5, 2, size=(50, len(LEVELS))), axis=1)
    np.testing.assert_allclose(blend(a, b, 1.0), a)
    np.testing.assert_allclose(blend(a, b, 0.0), b)


def test_fit_blend_weight_prefers_the_better_component():
    y_val, _, _ = _too_narrow_setup()
    good = np.tile(norm.ppf(LEVELS, scale=2), (len(y_val), 1))
    bad = good + 10
    grid = np.linspace(0, 1, 11)
    assert fit_blend_weight(y_val, good, bad, LEVELS, grid) == 1.0
    assert fit_blend_weight(y_val, bad, good, LEVELS, grid) == 0.0


def test_wrapper_blend_with_fixed_grid_reproduces_components(fold):
    train_df, test_df = fold
    for w in (0.0, 1.0):
        result = postprocessed_forecasts(train_df, test_df, LEVELS, _empirical_forecaster, _baseline,
                                         n_months=2, methods=["blend"], blend_grid=[w])
        expected = result.test_predictions["model"] if w == 1.0 else result.test_predictions["baseline"]
        np.testing.assert_allclose(result.test_predictions["blend"], np.sort(expected, axis=1))


def test_unknown_method_rejected(fold):
    train_df, test_df = fold
    with pytest.raises(ValueError):
        postprocessed_forecasts(train_df, test_df, LEVELS, _empirical_forecaster, _baseline, methods=["magic"])


def test_postprocess_off_by_default_in_configs():
    assert postprocess_settings({})["enabled"] is False
    for path in ("configs/default.yaml", "configs/dev.yaml"):
        with open(path) as f:
            assert postprocess_settings(yaml.safe_load(f))["enabled"] is False


def test_postprocess_method_names():
    assert postprocess_method_name("full", "calibrated") == "lightgbm_full_calibrated"
    assert postprocess_method_name("full", "recalmap") == "lightgbm_full_recalmap"
    assert postprocess_method_name("full", "blend") == "blend_full"
    assert postprocess_method_name("minimal", "blend_calibrated") == "blend_minimal_calibrated"
