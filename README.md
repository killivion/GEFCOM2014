# GEFCom2014-L Probabilistic Load Forecasting

## Problem
One-month-ahead hourly probabilistic electricity load forecasting on the
GEFCom2014 load track. For each hour in a forecast month, the task is to
predict the full distribution as 99 quantiles (1st-99th percentile),
scored with pinball loss, evaluated across the dataset's 15 built-in
rolling forecast rounds.

## Status
Data loading, the rolling-origin backtest harness (with a leakage guard),
pinball/coverage/calibration metrics, and both baselines (seasonal-naive,
climatology) are implemented and wired together end to end — running
`src/evaluation/run_baseline.py` produces a per-fold and summary results
table across all 14 test folds. Feature engineering (`src/features/build_features.py`)
and the main model (LightGBM quantile regression) are next.

## Setup
```
python -m venv .venv
.venv\Scripts\activate      # Windows; use `source .venv/bin/activate` on macOS/Linux
pip install -r requirements.txt
```

## Getting the data
1. Run: `python .\src\data\download_data` (uses `kagglehub`, requires a Kaggle account/API token configured).
2. Otherwise by hand: download `GEFCom2014-L_V2` from [Kaggle](https://www.kaggle.com/datasets/cthngon/gefcom2014-dataset) and unzip the `Load` folder's `Task 1` .. `Task 15` subfolders into `src/data/raw/Load/`.

Either way, the raw CSVs are expected at `src/data/raw/Load/Task <n>/L<n>-train.csv`.

## Running the pipeline
```
python main.py                              # smoke test: loads/parses all 15 tasks, prints shape/dtypes
python -m src.evaluation.run_baseline        # runs both baselines across all rolling folds
pytest tests/ -v                             # 10 tests: leakage guard + metric correctness
```

The first run of `main.py` or `run_baseline.py` parses the raw per-task CSVs
and caches the combined, tidy DataFrame to `src/data/processed/full_load.csv`
(see "Design notes" below); later runs load straight from that cache. Delete
the cache file (or pass `force_rebuild=True` to `get_data`) after changing
the loader.

## Current results (baselines only)
Rolling-origin backtest, tasks 2-15 (14 folds), mean pinball loss ± std across folds:

| baseline         | mean pinball loss | std  |
|------------------|-------------------|------|
| climatology      | 10.06              | 4.36 |
| seasonal-naive    | 10.69              | 4.48 |

Climatology still edges out seasonal-naive on average, but they're now
close and similarly stable fold-to-fold (see "Design notes" below for a
fix that closed most of the previous gap). Task 4 is a notable outlier
for both baselines (pinball loss spikes, 90%-interval coverage collapses
to ~20-60%) and hasn't been investigated yet. These are baseline numbers
only — no learned model has been trained yet, so there's no statistical
comparison (Diebold-Mariano) to report.

## Models

- **Climatology** (`climatology_quantiles`): for each (hour-of-day,
  day-of-week) bucket, predicts the empirical quantiles of historical load
  in that bucket. Ignores temperature and any recent trend entirely — it's
  a pure "what usually happens at this time" forecast.
- **Seasonal-naive** (`seasonal_naive_quantiles`): anchors on the load
  value from the same hour one week earlier, then adds the empirical
  quantiles of past week-over-week residuals as spread. Tracks recent
  trend better than climatology in principle, but is more sensitive to
  whatever happened in that one specific prior week.
- **Planned: LightGBM quantile regression**: a separate gradient-boosted
  tree trained per quantile level (via LightGBM's `quantile` objective),
  using calendar features, lagged load, and temperature as inputs. Chosen
  because it handles the nonlinear temperature-load relationship well,
  trains fast on a CPU, and supports quantile loss natively — no
  distributional assumption is needed, unlike a parametric regression.

## Models for future development

Not planned, only pursued if time allows after the LightGBM model and its
evaluation are done:

- **CatBoost quantile regression**: an alternative gradient-boosted-tree
  model (trained per quantile level, like the LightGBM one) as a second
  competitor. CatBoost's ordered boosting and native categorical handling
  could behave differently on calendar features and provide a useful
  cross-check on whether LightGBM's results are model-specific.
- **LightGBM + CatBoost ensemble**: simple averaging (or another
  combination) of the two models' predicted quantiles. Ensembling
  different tree-boosting implementations often reduces variance versus
  either model alone, at the cost of doubling training/inference and
  making results harder to attribute to a single, explainable model.


## Repo layout

```
configs/             experiment configuration (quantile levels, model
                      params, leakage assumptions, backtest fold range,
                      raw/processed data paths)
src/data/
  loader.py           parses Task 1..15 raw CSVs into one tidy DataFrame,
                       tagging each row with the task it was released in;
                       also handles the dataset's concatenated
                       month+day+year TIMESTAMP format (see Design notes)
  processed/          cached parsed DataFrame (gitignored, rebuilt on demand)
  raw/Load/            raw per-task CSVs from Kaggle (gitignored except this)
src/evaluation/
  metrics.py          pinball loss, interval coverage, calibration curve
  backtest.py         rolling-origin fold generator + leakage assertion
  run_baseline.py     runs both baselines across all folds, prints results
src/models/
  baselines.py        seasonal-naive and climatology quantile baselines
src/features/
  build_features.py   calendar + temperature feature engineering (WIP)
tests/                pytest suite for metrics correctness and the
                      leakage guard
main.py                smoke test for src/data/loader.py
```

## Design notes

- **Rolling-origin folds**: fold `t`'s training set is exactly the rows
  released in the Task `t-1` file; its test set is the new rows Task `t`
  reveals. A hard assertion in `backtest.py` blocks any fold where a
  test timestamp leaks into training.
- **Timestamp parsing**: the raw TIMESTAMP column concatenates
  month+day+4-digit-year with no separators or leading zeros (e.g.
  `1012010 1:00` = Oct 1, 2010, 1am). When only 3 digits remain for
  month+day, the split is genuinely ambiguous (`"127"` could be Jan 27 or
  Dec 7); `loader.py` resolves this by picking whichever valid split lands
  closest to the previous row's timestamp, since readings are hourly and
  strictly increasing.
- **Processed-data cache**: `get_data()` in `loader.py` parses the raw
  per-task CSVs once and writes the combined DataFrame to
  `src/data/processed/full_load.csv`; subsequent runs (and tests of
  downstream code) load the cache directly instead of re-parsing raw CSVs,
  which also makes it easy to open the cache and manually spot-check rows.
- **Temperature leakage**: real forecast time would not have actual
  future temperature. `configs/default.yaml:leakage.use_actual_future_temperature`
  makes this assumption explicit and switchable, so its impact on the
  score can be measured rather than silently assumed away.
- **Baselines first**: seasonal-naive and climatology are implemented
  and run across every fold before any learned model, so every later
  result is judged against them.
- **Seasonal-naive's multi-week anchor lookback**: each fold forecasts an
  entire month in one batch, so for test hours more than 7 days past the
  training cutoff, "same hour one week ago" falls inside the (not yet
  known) test period itself. `seasonal_naive_quantiles` in
  `baselines.py` handles this by stepping back additional whole weeks
  (14 days, 21 days, ...) until it finds a timestamp that's actually in
  training data — still the same hour-of-day/day-of-week, just further
  back — rather than an earlier version of this baseline, which fell
  back to a single flat last-known-value anchor for the ~75% of each
  month beyond that first week. That fallback silently discarded the
  daily/weekly load pattern for most of every fold and was the main
  reason seasonal-naive originally scored notably worse and less
  consistently than climatology; fixing it brought its mean pinball loss
  from 12.13±7.31 down to 10.69±4.48, much closer to climatology's
  10.06±4.36.

## Limitations / open items

- No learned model yet (LightGBM quantile regression is the planned next
  step), so there's no statistical comparison (Diebold-Mariano / paired
  test across folds) against the baselines yet.
- Naive temperature-forecast substitution (for the
  `use_actual_future_temperature: false` no-leakage variant) is stubbed
  in `build_features.py` but not implemented.
- Task 4's outlier pinball loss / coverage for both baselines hasn't been
  investigated.
- `kagglehub` (used by `src/data/download_data`) is not yet pinned in
  `requirements.txt`.

