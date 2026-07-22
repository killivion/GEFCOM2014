# AI-Generated Engineering Summary

> This file was produced with AI assistance (Claude Code) over the
> course of building this project. It captures implementation detail,
> design rationale, and engineering notes beyond the four things
> [`README.md`](README.md) is scoped to (problem description, repro
> instructions, model/baseline comparison, limitations). Kept for our
> own reference -- e.g. before the oral walkthrough -- not a required
> deliverable.

## Status
Data loading + feature engineering, the rolling-origin backtest harness
(with a leakage guard), pinball/coverage/calibration metrics, both
baselines (seasonal-naive, climatology), and a LightGBM quantile-
regression model are implemented and wired together end to end.

`src/evaluation/run_comparison.py` runs baselines and model side by side
across all 14 test folds in one pass and reports both significance tests
plus a calibration reliability diagram. `src/eda/explore_data.py` is a
brief standalone data overview. Not yet done: a real hyperparameter
search and the CatBoost/ensemble stretch goals (see "Limitations" in
`README.md`).

## Exploratory data overview
`src/eda/explore_data.py` is a brief, standalone look at the parsed
dataset -- independent of the modelling pipeline, meant as a first thing
to run after getting the data. It prints the row count, date range,
missing-load fraction, and a correlation matrix between load and the
temperature-derived features, and saves both the numeric stats/correlation
CSVs and four plots -- load distribution, temperature-vs-load scatter,
average daily (hour-of-day) load profile, and load-by-month boxplot --
to `reports/eda/`.

The temperature-vs-load relationship is the most useful finding: raw
temperature correlates only weakly with load (r=0.11), because the
relationship is U-shaped (see `temperature_vs_load.png`) -- load rises in
both cold and hot weather (heating and cooling demand), so a plain linear
correlation cancels itself out. `cooling_degrees_actual` (temperature
above 65°F) correlates far more strongly (r=0.59) than
`heating_degrees_actual` (r=0.21), suggesting cooling/AC demand drives
more of this zone's temperature-sensitive load than heating does -- this
is exactly why the model uses heating/cooling-degree features rather than
raw temperature (see "Models" below).

## Models

- **Climatology** (`climatology_quantiles`): for each (hour-of-day,
  day-of-week) bucket, predicts the empirical quantiles of historical load
  in that bucket. Ignores temperature and any recent trend entirely — it's
  a pure "what usually happens at this time" forecast.
- **Seasonal-naive** (`seasonal_naive_quantiles`): anchors on the load
  value from the same hour one week earlier (stepping back further whole
  weeks when that falls inside the very month being forecast), then adds
  the empirical quantiles of past week-over-week residuals as spread.
- **LightGBM quantile regression** (`lightgbm_quantiles`): one gradient-
  boosted-tree regressor per quantile level (LightGBM's `quantile`
  objective). Two feature sets:
  - `full`: calendar + temperature + load-lag/rolling features (see
    "Design notes" for how lag features avoid leakage). This is the
    primary model and what the results table in `README.md` reports.
  - `minimal`: calendar + temperature only (no load lags), predicted in
    one batch -- a faster baseline to check whether the lag/rolling
    features in `full` actually help. Run `run_model.py --feature-set both`
    to compare them directly.

## Models for future development

Not planned, only pursued if time allows:

- **CatBoost quantile regression**: an alternative gradient-boosted-tree
  model (trained per quantile level, like the LightGBM one) as a second
  competitor, to cross-check whether results are LightGBM-specific.
- **LightGBM + CatBoost ensemble**: simple averaging of the two models'
  predicted quantiles, at the cost of doubling training/inference and
  making results harder to attribute to a single, explainable model.
- **Conformalized Quantile Regression (CQR)** for the calibration gap:
  the monotonicity fix and hyperparameter tweak already applied (see
  "Calibration: full diagnosis and fixes" below) help but don't close the
  90%-coverage gap to the 0.90 target. CQR would hold out a calibration
  slice (e.g. the last month or two of each fold's *training* data --
  never touching the test month) and use it to measure exactly how far
  off the raw quantile predictions are from nominal coverage, then apply
  a single additive correction per quantile before scoring. Unlike more
  hyperparameter tuning, this gives a distribution-free, finite-sample
  coverage guarantee and directly targets the systematically-too-narrow
  tails observed in the reliability diagram, rather than hoping some
  config happens to fix it.

## Repo layout

```
configs/             experiment configuration (quantile levels, model
                      params, leakage assumptions, backtest fold range,
                      raw/processed data paths); no_leakage.yaml swaps in
                      the honest (last-year) temperature substitute
src/data/
  loader.py           parses Task 1..15 raw CSVs into one tidy DataFrame,
                       tagging each row with the task it was released in;
                       handles the dataset's concatenated month+day+year
                       TIMESTAMP format; adds calendar/temperature
                       features once via build_features.py (see Design notes)
  processed/          cached, feature-augmented DataFrame (gitignored, rebuilt on demand)
  raw/Load/            raw per-task CSVs from Kaggle (gitignored except this)
src/features/
  build_features.py   calendar features, both temperature variants
                       (actual / last-year substitute + heating/cooling
                       degrees), and the vectorized train-only load-lag
                       helper (see Design notes for the leakage boundary)
src/evaluation/
  metrics.py          pinball loss, interval coverage, calibration curve,
                      Diebold-Mariano test
  backtest.py         rolling-origin fold generator + leakage assertion
  run_baseline.py     runs both baselines across all folds, prints results
  run_model.py        runs LightGBM (full/minimal/both) across all folds
  run_comparison.py   baselines + model together, in one pass: per-fold
                      table, summary, coverage, paired t-test, Diebold-
                      Mariano test, and the calibration reliability
                      diagram -> reports/run_comparison/
src/models/
  baselines.py        seasonal-naive and climatology quantile baselines
  lightgbm_model.py   LightGBM quantile regression, full + minimal feature
                      sets, recursive no-leakage day-by-day prediction
src/eda/
  explore_data.py     brief, standalone data overview -> reports/eda/
tests/                34 tests: leakage guard, metric correctness, baseline
                      correctness, timestamp-parsing correctness, and
                      LightGBM-specific no-leakage and monotonicity checks
reports/                generated output, regenerate via the commands in
                      README.md: one subfolder per script (e.g.
                      reports/run_comparison/, reports/eda/), each holding
                      both its CSVs and any plots together. Every script
                      always overwrites its own files on the next run --
                      reports/ holds only the latest run, not a history.
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
  per-task CSVs once, adds calendar/temperature features, and writes the
  combined DataFrame to `src/data/processed/full_load.csv`; subsequent
  runs load the cache directly.
- **Feature engineering, and where the leakage boundary actually is**:
  calendar features and both temperature variants (`build_features.py`)
  are safe to compute ONCE on the full dataset, because each is a fixed
  lookup relative to a row's own timestamp (itself, or exactly one year
  earlier) that never depends on which fold that row lands in. Load-lag
  features (`load_lag_24h`, `load_lag_168h`, rolling mean/std) are NOT
  precomputed globally -- what's "known" at forecast time differs between
  training rows (their own real history) and a test fold's rows (only
  real history up to the fold's cutoff). `lightgbm_model.py` handles this
  by forecasting the test month one calendar day at a time, feeding each
  day's own median (0.5-quantile) prediction back in as "known" load for
  the next day's lag features -- so no lag/rolling feature, at any point,
  is built from real data belonging to the period being forecast. This is
  covered directly by `tests/test_lightgbm_no_leakage.py`, including a
  test that swaps in deliberately wrong "real" test-period load and
  confirms every prediction is unchanged.
- **Temperature leakage**: real forecast time would not have actual
  future temperature. `configs/default.yaml:leakage.use_actual_future_temperature`
  switches between the real value and a same-hour-last-year substitute
  (sourced only from training data), so the assumption is explicit and
  its impact measurable rather than silently assumed away. With the
  `full` feature set: using actual future temperature gives a
  meaningfully lower mean pinball loss than the honest last-year
  substitute, which roughly doubled it in an earlier comparison (see git
  history for the exact run) -- confirming temperature is doing real work
  and that the assumption matters.
- **Baselines first**: seasonal-naive and climatology are implemented
  and run across every fold before any learned model, so every later
  result is judged against them.
- **Two statistical tests, not one**: `run_comparison.py` reports both a
  paired t-test and a Diebold-Mariano test (`metrics.py:diebold_mariano_test`)
  for the model against each baseline. With h=1 (appropriate here, since
  folds are distinct non-overlapping months rather than an overlapping
  multi-step horizon) and the Harvey-Leybourne-Newbold small-sample
  correction, the DM test is mathematically identical to the paired
  t-test in this non-autocorrelated case -- included because the
  assignment specifically names Diebold-Mariano as an example comparison,
  and because reporting two independent tests that agree is stronger
  evidence than reporting either alone.
- **Seasonal-naive's multi-week anchor lookback**: each fold forecasts an
  entire month in one batch, so for test hours more than 7 days past the
  training cutoff, "same hour one week ago" falls inside the (not yet
  known) test period itself. `seasonal_naive_quantiles` steps back
  additional whole weeks until it finds a timestamp actually in training
  data, rather than an earlier version of this baseline, which fell back
  to a single flat last-known-value anchor for ~75% of each month and was
  the main reason it originally scored notably worse than climatology.
- **Task 4 and Task 12 outliers, explained**: both are the worst fold for
  every method (baselines and LightGBM alike), and both turned out to be
  real events rather than bugs:
  - **Task 4 (test month: December 2010)**: this was the coldest
    December in the entire training history by a wide margin -- mean
    temperature 34.5°F, versus 41.6-49.6°F for every other December from
    2001-2009 in this dataset. Load hit a record high accordingly (mean
    203.8 vs. a prior max of 162.3). This is an out-of-distribution
    weather extreme: no model had ever seen a December this cold, so
    every one of them underestimated the resulting demand.
  - **Task 12 (test month: August 2011)**: temperature and load were
    completely unremarkable for most of the month (in line with prior
    Augusts) until Aug 27, when load craters from its normal ~175-190 MW
    range down to as low as 48 MW over Aug 27-28, with temperature
    staying normal throughout. That's a demand collapse with no weather
    cause -- consistent with the widespread power outages caused by
    Hurricane Irene, which struck the US East Coast on exactly those
    dates. No temperature-driven model could reasonably predict an
    infrastructure outage.
  - Since climatology (which doesn't use temperature at all) also
    struggled on both folds, this rules out "just use better temperature
    features" as a fix -- these are genuinely unpredictable tail events
    from the information available at forecast time.

## Calibration: full diagnosis and fixes

LightGBM's much lower pinball loss does not mean its intervals are
better calibrated -- the opposite, in fact. Its mean 90%-coverage (0.803)
is further from the 0.90 target than climatology's (0.898). The full
reliability diagram (`reports/run_comparison/calibration.png`) shows
*why*: LightGBM's curve sits slightly above the diagonal for low
quantiles and increasingly below it for high quantiles (a downward
S-shape) -- its lower quantiles are a touch too high, but more
importantly its upper quantiles (0.55 and up) are systematically too
low, so real load exceeds the predicted upper bound more often than it
should. Climatology's curve hugs the diagonal far more closely across
the whole range; seasonal-naive is the worst calibrated of the three,
running well above the diagonal from 0.05 through 0.65. Pinball loss
aggregates error across all quantiles, so LightGBM's large
central-quantile accuracy win can outweigh this real tail-calibration
weakness in the aggregate score -- worth stating plainly rather than
only reporting the pinball-loss win.

Investigating the S-shape turned up a real, checkable cause -- since
each quantile level is trained as an *independent* LightGBM model,
nothing guarantees `pred(tau=0.05) <= pred(tau=0.10) <= ...` for a given
row. Checked directly: a meaningful fraction of rows had at least one
such crossing violation. Fixed via the standard rearrangement correction
(`_enforce_monotonic_quantiles` in `lightgbm_model.py`, applied inside
both prediction paths, including before the recursive path feeds a day's
median forward) -- crossings are now 0% on the same check.

Separately, a quick manual hyperparameter sweep on 2 folds (4 configs)
found shallower-but-wider trees (`max_depth=4, num_leaves=63`, now the
default in `configs/`) gave meaningfully better 90%-coverage there (0.859
vs. 0.842) without hurting pinball loss -- but re-running the FULL
14-fold comparison with both fixes applied told a more modest, more
honest story: coverage@90% moved from 0.795 to only 0.803, and mean
pinball loss actually got slightly worse (2.81 to 2.97, ~5.5%). **The
2-fold quick check overestimated the hyperparameter change's benefit** --
a reminder that a 2-fold sweep is not a substitute for evaluating across
all folds, even for a "quick check." The monotonicity fix itself is
unambiguously correct to keep (it fixes a real bug, not a tunable
tradeoff); the hyperparameter change is a smaller, more debatable win
than first reported, and is kept mainly for the modest calibration
improvement, with the pinball-loss cost noted plainly here rather than
left undisclosed. This tuning was intentionally sparse -- one 2-fold,
4-config manual sweep, not a real search -- because the assignment's
time budget didn't justify a longer one once it became clear
hyperparameters alone weren't closing the gap to the 0.90 coverage
target. The deeper, more principled fix -- conformal calibration -- is
proposed but not implemented; see "Models for future development" above.
