# AI-Generated Engineering Summary

> Produced with AI assistance (Claude Code) while building this project.
> It holds implementation detail and design rationale beyond
> [`README.md`](README.md), which has the problem statement, results,
> reproduction steps, limitations and the experiment log. Numbers quoted
> here are from the final run (99 quantiles; commit `a6e563d`, main
> comparison re-run identically at `0a02e17`).

## Status
Done: data loading and features, rolling-origin backtest with leakage
guards, four baselines, the LightGBM quantile model with temperature
scenarios and recency weighting, an oracle upper bound and three ablations,
hour-level and month-level significance tests, reports with provenance,
the official GEFCom2014 benchmark files, sequential conformal
recalibration (tested, not adopted), the TabPFN benchmark with seasonal
calibration as a secondary, post-hoc analysis (not adopted), 122 tests,
Docker image and GitHub Actions CI.

Not done (see README "Limitations"): a real hyperparameter search, more
test years, and closing the remaining calibration gap (coverage 0.88 vs
0.90 target).

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
- `gefcom_benchmark` (`src/data/official_benchmark.py`): the organisers'
  `Task t/Lt-benchmark.csv`, which forecasts the month after task t's
  training data, so fold t uses `L{t-1}`. The raw dates are resolved
  against the hour before the fold (`1012010` alone could be 1 Jan); the
  loader raises unless the file covers exactly the fold's test hours.
  Verified on all 14 folds: one value across the 99 quantiles, equal to
  the load on the same date one year earlier.

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
- `--baseline none` / `--skip-stats` exist for the external experiment
  agent (hundreds of runs reading only summary/per-fold losses/run_info);
  they skip work, never change a reported number. Single-method runs skip
  significance (nothing to test against).

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
load below it, i.e. a slight under-forecast on average -- but per month
that share runs from 0.05 (Dec 2010) to 0.69 (Oct 2011), so the bias is
weather, not a stable offset. Two post-processing approaches were tried
and dropped: in-fold recalibration with inner validation, and sequential
conformal recalibration (`src/models/conformal.py`,
`src/evaluation/run_conformal.py`: each test month adjusted from the
model's own forecasts of earlier test months; the selection on the
development folds kept "none", and every variant was also worse on the
holdout).

## TabPFN benchmark (`run_tabpfn_benchmark.py`, `tabpfn_model.py`)
- Question: is a tabular foundation model (TabPFN, Prior Labs API) better
  than the final LightGBM model? Same test hours, 99 levels, same metrics.
- Shared inputs: `build_model_inputs` (lightgbm_model.py) builds features
  and the scenario test matrix for both models; refactored out of
  `lightgbm_quantiles` with bit-identical LightGBM output (checked on a fold).
- TabPFN (API default v3.5) accepts up to 1.1M context rows, so the
  primary variant `tabpfn` uses every training row, unweighted. TabPFN
  has no sample weights, so `tabpfn_recency` (time series only) uses a
  20k-row sample with inclusion probability proportional to the recency
  weight (systematic sampling). Both fixed in advance, both reported.
- Client constraints: no test-set chunking -> `predict_chunk_rows`;
  quantiles come from one predictive distribution (monotone, no per-level
  models).
- Fairness: each model's best honest setup (LightGBM keeps all rows),
  identical evaluation. TabPFN untuned; config fixed before results.
- Tabular task: random week blocks of 2005 - Sep 2010 with a 7-day
  embargo, climatology temperature; interpolation, labelled as such.
- Calibration: conformal from other test data only -- sequential
  (earlier months) for the forecast, cross-fold for the tabular task
  (valid only because week blocks are exchangeable).
- Predictions per method are cached (`predictions_<method>.npz`), so the
  API runs once and the evaluation can be rebuilt offline. Transient API
  network errors are retried per fold (deterministic predictions);
  run_info.json is written after every method.

- Results: tabular (interpolation) LightGBM 8.38 vs TabPFN 8.60 (0/5
  folds). Month-ahead holdout LightGBM 6.79 vs TabPFN 7.84 (all rows) /
  7.22 (recency sample); recalibration helps TabPFN (it under-forecasts:
  32% of loads below its median, load level drift with an equally weighted
  context) but not LightGBM. TabPFN ~49 s per fold via the API, LightGBM
  ~436 s locally.
- Secondary, post-hoc: seasonal calibration (`seasonal_variants`,
  `conformal.seasonal_recalibration`). Each scored month is recalibrated on
  earlier forecast months within one calendar month of its own, including
  never-scored history months Jan 2008 - Sep 2010
  (`calibration_<method>.npz`, ~2.2M API credits). Not adopted, because the
  gain does not carry over: development LightGBM 8.27-8.36 vs 8.78 raw, but
  holdout 6.77-6.88 vs 6.79, with coverage 0.83-0.85 instead of 0.87 and
  mean calibration gap ~0.10 instead of 0.011. It narrows TabPFN's deficit
  (recency variant holdout 6.91-7.04 vs 7.22; all 14 months 7.63 vs
  LightGBM's 7.62 with seasonal additive), but LightGBM still leads on
  the holdout (2.4-4.5%, Holm p = 0.22). The two seasonal comparisons
  extend the Holm family to 6, so its holdout floor is 0.094.

### Self-audit: ML pitfalls in the TabPFN benchmark
| pitfall | status |
|---|---|
| leakage of test load / temperature | guarded by tests (corrupting them changes nothing) |
| neighbour leakage in the tabular split | week blocks + 7-day embargo, tested to the hour |
| selection bias: LightGBM tuned on the development folds | dev comparison favours LightGBM; the holdout is the fair one |
| holdout reuse | the LightGBM holdout (and a first TabPFN pass) had been seen before seasonal calibration was proposed, so its holdout result is not a clean confirmation; its window was fixed before any seasonal result |
| common-mode failure (shared inputs, metrics, calibration) | LightGBM reproduces the published losses to 1e-15; identical y / timestamps asserted across methods; TabPFN quantile order verified on the API (the per-row sort would mask a scrambled order) |
| multiple comparisons (2 TabPFN variants x 3 calibration states) | Holm across all comparisons per fold group |
| calibration history from weaker models (2008 months trained on 3 years) | may over-widen seasonal intervals; checked via residual spread by history year |
| model drift behind the API (`auto`) | exact model file recorded per run; warning if history and scored runs differ |
| runtime contamination (sleep, concurrent runs) | median per fold; per-fold table kept; runtimes approximate |
| tabular: training on later weeks; cross-conformal | disclosed; interpolation only |
| 5-fold Wilcoxon floor (p >= 0.0625) | disclosed |

## Repo layout
```
configs/
  default.yaml               final model (single source of truth)
  make_variants.py           regenerates the configs below from default.yaml
  dev.yaml                   20-quantile grid for iteration
  oracle_temperature.yaml    actual forecast-month temperature (upper bound)
  ablation_*.yaml            plain / recency_only / scenarios_only
  tabpfn_benchmark.yaml      TabPFN vs. LightGBM benchmark (not generated)
src/data/        loader.py (parse + cache), official_benchmark.py,
                 download_data (kagglehub)
src/features/    build_features.py (calendar, temperature variants)
src/models/      baselines.py, lightgbm_model.py, tabpfn_model.py, conformal.py
src/evaluation/  backtest.py, metrics.py, run_comparison.py,
                 run_conformal.py, compare_runs.py, report_utils.py,
                 run_tabpfn_benchmark.py
src/eda/         explore_data.py -> reports/eda/
tests/           122 tests on synthetic data (leakage, features, baselines,
                 metrics, scenarios, recency, conformal, comparison tooling)
reports/         one folder per run: results, per-hour losses, summary and
                 significance per fold group, calibration, run_info.json
Dockerfile, docker-compose.yml, .github/workflows/ci.yml
```
