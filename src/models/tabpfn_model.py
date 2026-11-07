"""
TabPFN quantile forecasts through the Prior Labs API (tabpfn-client).

TabPFN is a tabular foundation model: a transformer pre-trained on
millions of synthetic regression/classification datasets. It does not
train on our data -- "fit" uploads the training rows as CONTEXT, and
each prediction is one forward pass that reads the context and the test
rows together (in-context learning). It returns a full predictive
distribution per row, so all quantile levels come from one call and are
monotone by construction (LightGBM needs one model per level).

Same inputs as LightGBM (build_model_inputs in lightgbm_model.py): the
same features, and with temperature scenarios the test month is predicted
under every scenario and the scenario forecasts are mixed. Two things
differ, both forced by the model:
- Context: the training rows go in as context; context_rows caps them
  (None = all; the API accepts up to 1.1M rows) with a random subsample.
- No sample weights: LightGBM's recency weighting is reproduced by
  drawing the subsample with inclusion probability proportional to the
  recency weight (recency_subsample), so recent years are over-represented
  the same way.

Requires `pip install -r requirements-tabpfn.txt` and a Prior Labs access
token in the TABPFN_TOKEN environment variable. The client is imported
lazily, so nothing else in the project needs it.
"""
from __future__ import annotations

import time
from typing import Callable

import numpy as np
import pandas as pd

from src.models.lightgbm_model import build_model_inputs, finish_quantiles, recency_weights

DEFAULT_TABPFN = {"context_rows": None, "predict_chunk_rows": 200_000, "seed": 0, "regressor_params": {}}


def inclusion_probabilities(weights: np.ndarray, n: int) -> np.ndarray:
    """pi_i proportional to weights_i, summing to n, capped at 1 (rows whose
    share would exceed 1 are taken for certain, the rest rescaled)."""
    pi = np.zeros(len(weights))
    certain = np.zeros(len(weights), dtype=bool)
    while True:
        rest = ~certain
        pi[certain] = 1.0
        pi[rest] = (n - certain.sum()) * weights[rest] / weights[rest].sum()
        if (pi[rest] <= 1).all():
            return pi
        certain |= pi > 1


def recency_subsample(timestamps: pd.Series, cutoff: pd.Timestamp, n: int, half_life_years: float | None,
                      seed: int) -> np.ndarray:
    """Positions (into `timestamps`) of a sample of min(n, len) rows whose
    INCLUSION probability is proportional to recency_weights(...,
    half_life_years) -- uniform when it is None -- so the context carries
    the same recency tilt as LightGBM's sample weights. (Drawing rows one
    by one with probability ~ weight, as numpy's choice(replace=False, p)
    does, flattens the tilt once a large share of rows is drawn.)
    Systematic sampling over a random order: exactly n rows, inclusion
    probabilities exact. Returned sorted, i.e. in the rows' original order."""
    size = min(n, len(timestamps))
    rng = np.random.default_rng(seed)
    if half_life_years is None:
        return np.sort(rng.choice(len(timestamps), size=size, replace=False))
    pi = inclusion_probabilities(recency_weights(timestamps, cutoff, half_life_years), size)
    order = rng.permutation(len(pi))
    cumulative = np.concatenate([[0.0], np.cumsum(pi[order])])
    u = rng.uniform()
    # Row j is picked when an integer k + u falls in its cumulative interval.
    picked = np.floor(cumulative[1:] - u) > np.floor(cumulative[:-1] - u)
    return np.sort(order[picked])


def _default_regressor_factory(**params):
    from tabpfn_client import TabPFNRegressor  # lazy: optional dependency
    return TabPFNRegressor(**params)


def tabpfn_quantiles(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    quantile_levels: list[float],
    temperature_source: str = "climatology",
    feature_set: str = "full",
    temperature_scenarios: dict | None = None,
    recency_half_life_years: float | None = None,
    temperature_dynamics_features: list[str] | None = None,
    tabpfn: dict | None = None,
    regressor_factory: Callable | None = None,
    timings: dict | None = None,
) -> np.ndarray:
    """Quantile forecasts of shape (len(test_df), len(quantile_levels)).

    Inputs as in lightgbm_quantiles (test_df's load and, with scenarios,
    its temperatures are never read). tabpfn: {context_rows (None = all
    training rows), predict_chunk_rows, seed, regressor_params} (see
    DEFAULT_TABPFN);
    regressor_params go to TabPFNRegressor. regressor_factory replaces the
    API client (tests). If given, `timings` gets fit_seconds,
    predict_seconds, context_rows and model_file (None if unknown).
    """
    settings = {**DEFAULT_TABPFN, **(tabpfn or {})}
    inputs = build_model_inputs(train_df, test_df, temperature_source, feature_set, temperature_scenarios,
                                temperature_dynamics_features)
    n_context = settings["context_rows"] or len(inputs.X_train)  # None = every training row
    rows = recency_subsample(inputs.train_timestamps, inputs.cutoff, n_context, recency_half_life_years,
                             settings["seed"])
    X_context, y_context = inputs.X_train.iloc[rows], inputs.y_train.iloc[rows]

    model = (regressor_factory or _default_regressor_factory)(**settings["regressor_params"])
    start = time.perf_counter()
    model.fit(X_context, y_context)
    fit_seconds = time.perf_counter() - start

    start = time.perf_counter()
    chunk = settings["predict_chunk_rows"]
    parts = []
    for i in range(0, len(inputs.X_test), chunk):
        out = model.predict(inputs.X_test.iloc[i:i + chunk], output_type="quantiles",
                            quantiles=list(quantile_levels))
        parts.append(np.column_stack(out))  # one array per level -> (rows, levels)
    preds = np.vstack(parts)
    predict_seconds = time.perf_counter() - start

    if timings is not None:
        timings.update({"fit_seconds": fit_seconds, "predict_seconds": predict_seconds, "context_rows": len(rows)})
        # The exact model file the API used (more precise than its version name).
        meta = getattr(model, "last_meta", None) or {}
        timings["model_file"] = (meta.get("tabpfn_config") or {}).get("model_path")
    return finish_quantiles(preds, inputs, quantile_levels)
