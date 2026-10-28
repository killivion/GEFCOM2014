"""
The official GEFCom2014-L benchmark forecasts, as shipped with the data.

Each `Task t/Lt-benchmark.csv` holds the organisers' benchmark for the
month AFTER task t's training data -- the month released in task t+1's
train file. Our backtest labels a fold by the task whose train file
releases the test month (fold 2 = Oct 2010), so fold t is scored against
`L{t-1}-benchmark.csv`: fold t = GEFCom2014 competition task t-1.

Per the data instructions, the benchmark is "the same month last year as
the predicted load across all quantiles": a point forecast repeated over
the 99 quantile columns (a degenerate distribution). It was published
with each task, so it only uses data before the forecast month.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.data.loader import _parse_concatenated_timestamps

OFFICIAL_BENCHMARK = "gefcom_benchmark"


def benchmark_path(raw_load_dir: str | Path, test_task: int) -> Path:
    """The benchmark file for backtest fold `test_task` (see module docstring)."""
    competition_task = test_task - 1
    return Path(raw_load_dir) / f"Task {competition_task}" / f"L{competition_task}-benchmark.csv"


def load_official_benchmark(raw_load_dir: str | Path, test_task: int, previous_hour: pd.Timestamp) -> pd.DataFrame:
    """The benchmark for fold `test_task`: a `timestamp` column plus one
    column per quantile level (as floats, e.g. 0.01 ... 0.99).

    previous_hour: the hour just before the file's first row. The raw
    dates are ambiguous ("1012010" = 1 Jan or 1 Oct), and the loader
    resolves them against the previous row, as for the train files."""
    raw = pd.read_csv(benchmark_path(raw_load_dir, test_task))
    timestamps, _ = _parse_concatenated_timestamps(raw["TIMESTAMP"], prev=previous_hour)
    quantiles = raw.drop(columns=["ZONEID", "TIMESTAMP"])
    quantiles.columns = [float(c) for c in quantiles.columns]
    return pd.concat([timestamps.rename("timestamp"), quantiles], axis=1)


def official_benchmark_quantiles(raw_load_dir: str | Path, test_task: int, test_timestamps: pd.Series,
                                 quantile_levels: list[float]) -> np.ndarray:
    """The benchmark as an (n_hours, n_levels) array aligned to the fold's
    test hours. Raises if the file does not cover exactly those hours or
    lacks a requested level, so a mismatched file can never be scored."""
    expected = pd.Series(pd.to_datetime(test_timestamps)).reset_index(drop=True)
    bench = load_official_benchmark(raw_load_dir, test_task, expected.iloc[0] - pd.Timedelta(hours=1))
    if len(bench) != len(expected) or not (bench["timestamp"].reset_index(drop=True) == expected).all():
        raise ValueError(f"{benchmark_path(raw_load_dir, test_task)} does not cover exactly the test hours "
                         f"of fold {test_task} ({expected.iloc[0]} .. {expected.iloc[-1]})")
    missing = [q for q in quantile_levels if not np.isclose(bench.columns[1:].to_numpy(float), q).any()]
    if missing:
        raise ValueError(f"benchmark file lacks quantile levels {missing}")
    columns = [next(c for c in bench.columns[1:] if np.isclose(c, q)) for q in quantile_levels]
    return bench[columns].to_numpy(float)
