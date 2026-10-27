# AI-Generated Engineering Summary

> Produced with AI assistance (Claude Code) while building this project.
> It holds implementation detail and design rationale beyond
> [`README.md`](README.md), which has the problem statement, results,
> reproduction steps, limitations and the experiment log. Numbers quoted
> here are from the final run at commit `a6e563d` (99 quantiles).

## Status
Done: data loading and features, rolling-origin backtest with leakage
guards, four baselines, the LightGBM quantile model with temperature
scenarios and recency weighting, an oracle upper bound and three ablations,
hour-level and month-level significance tests, reports with provenance,
76 tests.

Not done (see README "Limitations"): a real hyperparameter search, use of
the official GEFCom2014 benchmark files, and closing the remaining
calibration gap (coverage 0.88 vs 0.90 target).

## Data and timestamps
- `src/data/loader.py` parses the Task 1..15 CSVs into one frame, tagging
  each row with the task that released it. The raw TIMESTAMP concatenates
  month+day+year without separators (`1012010 1:00` = 1 Oct 2010, 01:00);
  the ambiguous 3-digit month+day case (`127`: 27 Jan or 7 Dec) is resolved
  by choosing the split closest to the previous row, since rows are hourly
  and increasing.
- Timestamps are **hour-ending**: `00:00` is hour 24 of the previous day.
  All day-level keys (weekday, month, day of year, holiday, the climatology
  year) use `timestamp - 1h` (`build_features.operating_day_timestamps`).
  Example: `2010-11-01 00:00` belongs to Sunday 31 October.
- The processed cache (`src/data/processed/full_load.csv`) is written with a
  `.version` file; `get_data()` rebuilds it when `FEATURE_VERSION` changes,
  so stale features are never silently reused.
- Load starts in 2005, temperatures in 2001; the 25 stations have no
  missing readings (the code raises if any are missing).

## Backtest
`src/evaluation/backtest.py`: fold *t* trains on every row released before
task *t* and tests on the month task *t* releases (tasks 2..15 = Oct 2010 to
Nov 2011). It asserts that no test timestamp is in training and that
training ends before the test month starts.

Fold groups (`backtest.development_last_test_task: 8`): development = tasks
2-8 (Oct-Apr), holdout = tasks 9-15 (May-Nov).

## Features (`src/features/build_features.py`)
Computed once on the full dataset, which is safe because each feature only
reads the row's own timestamp or fixed, strictly earlier ones:
- calendar: hour, weekday, month, day of year, weekend, US federal holiday
  (observed date), cyclic encodings of hour and weekday;
- temperature: the station mean (`temp_mean_actual`), the same hour one
  year earlier (`temp_mean_last_year`), and a prior-years climatology
  (`temp_mean_climatology`: same hour, +/-7 days of the same no-leap day of
  year, averaged over earlier years only), each with heating and cooling
  degrees around 65 F.

## Models
- **LightGBM** (`src/models/lightgbm_model.py`): one `LGBMRegressor` with
  the quantile objective per level (`max_depth=4`, 16 leaves, 300 trees,
  learning rate 0.05, fixed seed); predictions are sorted per row so
  quantiles never cross (rearrangement; about a quarter of rows crossed
  before this was added).
- **Temperature scenarios** (`build_temperature_scenarios`,
  `combine_scenario_quantiles`): with scenarios on, the models train on the
  actual past temperatures. For the test month, scenario (k, s) takes the
  observed temperature at the same timestamp k years earlier, shifted by
  s days (s in -7..+7), read only from training rows before the cutoff; a
  scenario with any missing or post-cutoff source hour is dropped whole.
  About 140 scenarios per fold. The predictions under each scenario are
  combined as a mixture: per hour, all scenario quantiles are pooled, each
  weighted by the probability mass its level stands for, and the weighted
  quantiles are read off. (Plain averaging of the scenario quantiles made
  the intervals far too narrow.) The mixture cannot reach beyond the most
  extreme scenario quantile.
- **Recency weighting**: sample weight 0.5^(age / 2 years), age measured
  from the training cutoff. It makes training about 5x slower.
- **Feature sets**: `full` (the reported model) and `minimal` (hour,
  weekday, month, weekend, temperature), kept as a check.

## Baselines (`src/models/baselines.py`)
All use only the fold's training data:
- `seasonal_climatology`: empirical quantiles per (month, hour,
  weekday/weekend) bucket;
- `seasonal_climatology_recency` (primary): the same buckets with
  recency-weighted quantiles (half-life 2 years); it was the better of the
  two on the development folds (8.79 vs 9.12);
- `seasonal_naive`: the load 52 weeks earlier (same weekday) plus empirical
  quantiles of 52-week differences;
- `climatology`: per (hour, weekday), all months pooled; a naive reference.

## Statistics (`src/evaluation/metrics.py`, `run_comparison.py`)
- Pinball loss per hour (mean over the 99 levels) is the unit of the
  hourly tests; the month (fold) mean is the unit of the month tests.
- Diebold-Mariano with a Newey-West variance, at 168 and 720 lags.
- Circular block bootstrap (one-week blocks) and a cluster bootstrap that
  resamples whole test months; the latter keeps any within-month
  dependence and is the most conservative hourly test.
- Months won and the Wilcoxon signed-rank test on month losses.
- `p_conservative` = the largest of those p-values, Holm-adjusted within
  each family (baselines; model variants); the oracle is reference only.
- `compare_runs.py` pairs runs with different configs hour by hour,
  checking that timestamps and quantile grids match.

## Outlier months
Two folds stand out, and both are real events:
- **Task 4, December 2010:** the coldest December in the history (mean
  34.5 F vs 41.6-49.6 F in 2001-2009) with record load (mean 203.8 vs a
  prior maximum of 162.3). No model had seen such a December; it is the
  worst fold for every method (final model: 19.8).
- **Task 12, August 2011:** normal until 27 August, then load collapses
  to as low as 48 MW over 27-28 August while temperature stays normal,
  matching the Hurricane Irene power outages. Temperature-driven models
  cannot predict an outage; it is among the hardest folds (final model:
  10.3, third after December 2010 and January 2011).

## Calibration
Final model, all 14 folds: 90% interval coverage 0.880; mean absolute gap
between nominal and empirical quantile levels 0.039 (recency-weighted
seasonal climatology: 0.022). The predicted median has 44.8% of observed
load below it, i.e. a slight under-forecast. Conformal or
quantile-recalibration post-processing was tried with in-fold inner
validation and did not improve the final model (README experiment log).

## Repo layout
```
configs/
  default.yaml               final model (single source of truth)
  make_variants.py           regenerates the configs below from default.yaml
  dev.yaml                   20-quantile grid for iteration
  oracle_temperature.yaml    actual forecast-month temperature (upper bound)
  ablation_*.yaml            plain / recency_only / scenarios_only
src/data/        loader.py (parse + cache), download_data (kagglehub)
src/features/    build_features.py (calendar, temperature variants)
src/models/      baselines.py, lightgbm_model.py
src/evaluation/  backtest.py, metrics.py, run_comparison.py,
                 compare_runs.py, report_utils.py
src/eda/         explore_data.py -> reports/eda/
tests/           76 tests on synthetic data (leakage, features, baselines,
                 metrics, scenarios, recency, comparison tooling)
reports/         one folder per run: results, per-hour losses, summary and
                 significance per fold group, calibration, run_info.json
```
