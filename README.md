# GEFCom2014 Probabilistic Load Forecasting

> Engineering detail, design rationale, and everything beyond the four
> points below live in [`AI_SUMMARY.md`](AI_SUMMARY.md).

## Problem
Based on the dataset GEFCom2014 the task is to forecast the full distribution of hourly electricity load
as 99 quantiles (1st-99th quantile) for each hour in a forecast month. The dataset consists of timestamps,
load, and temperature measures in 25 locations - this is split into 15 built-in rolling forecasting rounds.
The scored metric is pinball loss and is measured against two baselines (here climatology and seasonal-naive).

Current results can be found in the `reports/` folder as .csv files and pictures (this only keeps the latest run's results, not a history).

## Reproducing results end to end

### Setup
```
python -m venv .venv
.venv\Scripts\activate      # for Windows; on macOS/Linux use `source .venv/bin/activate`
pip install -r requirements.txt
```

### Getting the data
1. Run: `python .\src\data\download_data` 
(uses `kagglehub`).
2. Or by hand: download `GEFCom2014-L_V2` from [Kaggle](https://www.kaggle.com/datasets/cthngon/gefcom2014-dataset) and unzip the `Load` folder's `Task 1` .. `Task 15` subfolders into `src/data/raw/Load/`.

Either way, the raw CSVs are expected at `src/data/raw/Load/`.

### Running the pipeline
After getting the data (previous step), run in this order:
```
python -m src.data.loader                         # parses raw CSVs -> src/data/processed/full_load.csv,
                                                   # prints a summary so you can sanity-check the parse
python -m src.eda.explore_data                    # brief, independent data overview -> reports/eda/
pytest tests/ -v                                  # 36 tests on all kinds of matters

# The next 3 commands are individual pieces; to get everything (both
# baselines + the model + both significance tests + calibration) in one go,
# just run the 3rd one (run_comparison) on its own. Runtime roughly 20-30 min -
# for faster development runs, add --config configs/dev.yaml calculating less quantiles:
python -m src.evaluation.run_baseline             # both baselines across all rolling folds
python -m src.evaluation.run_model                # LightGBM (full feature set) across all rolling folds
python -m src.evaluation.run_comparison           # baselines + model together: pinball table, both
                                                   # significance tests, and the calibration reliability
                                                   # diagram (all in one run) -> reports/run_comparison/
```

## Model vs. baseline comparison
Rolling-origin backtest, tasks 2-15 (14 folds), mean pinball loss ± std,
and mean 90%-interval coverage (target ~0.90):

| method            | mean pinball loss | std  | mean coverage@90% |
|-------------------|--------------------|------|--------------------|
| **lightgbm (full)** | **3.65**         | 1.41 | 0.819              |
| climatology       | 12.81              | 5.61 | 0.898              |
| seasonal-naive    | 13.33              | 5.39 | 0.826              |

**The `std` column above *is* the uncertainty across folds** -- the
spread of pinball loss over the 14 rolling-origin folds. Two independent
statistical tests turn that spread into a formal significance check
(`run_comparison.py`, full output in `reports/run_comparison/`):

Paired t-test (positive `mean_diff` = LightGBM wins on average):

| model         | baseline       | mean_diff (baseline − model) | t_stat | p_value  |
|---------------|----------------|-------------------------------|--------|----------|
| lightgbm_full | climatology    | 9.16                          | 6.82   | 1.22e-05 |
| lightgbm_full | seasonal_naive | 9.68                          | 7.14   | 7.59e-06 |

Diebold-Mariano test (positive `dm_stat` = LightGBM wins on average):

| model         | baseline       | dm_stat | p_value  |
|---------------|----------------|---------|----------|
| lightgbm_full | climatology    | 6.82    | 1.22e-05 |
| lightgbm_full | seasonal_naive | 7.14    | 7.59e-06 |

Both tests agree with each other almost exactly (expected: with `h=1` and
the small-sample correction, the DM statistic is mathematically identical
to the paired t-test's here) and both put LightGBM ahead of both
baselines with p ≈ 0.00001, far below any conventional significance
threshold. LightGBM also has the lower pinball loss on every single one
of the 14 folds individually, not just on average.

**Calibration**: LightGBM's much lower pinball loss does not mean its
intervals are better calibrated. Its coverage@90% (0.819) is further
from the 0.90 target than climatology's (0.898) -- the reliability
diagram (`reports/run_comparison/calibration.png`) shows its upper
quantiles are systematically a bit too low, so real load exceeds the
predicted upper bound more often than it should. 

## Limitations and unsuccessful approaches

- **Runtime is not optimized beyond the basics.** The main lever is
  `configs/dev.yaml`'s reduced 20-quantile grid for faster iteration (see
  above); no other performance work (e.g. parallelizing the per-quantile
  LightGBM training, caching intermediate features) was attempted.
- **LightGBM's calibration gap isn't fully closed.** A quantile-crossing
  fix and a small hyperparameter adjustment improved it slightly but
  didn't close the gap to 0.90 coverage -- the deeper issue is that
  independently-trained per-quantile models have no mechanism to jointly
  target a coverage level. Conformalized Quantile Regression (CQR) is
  the principled next step; not implemented though due to time limitations.
- **Hyperparameter tuning was a single small manual sweep** (2 folds, 4
  configs), not a real search -- the assignment's time budget didn't
  justify more once it was clear a crossing/calibration fix mattered more
  than further tuning.
- **CatBoost / an ensemble were not attempted**, time-permitting stretch
  goals only.
- The two statistical comparison tests both assume the loss
  differential's variance is stable across folds; two of the 14 folds
  (see `AI_SUMMARY.md`) are genuine outlier months, so that assumption is
  only approximate. Not corrected for.
- **Unsuccessful approaches along the way**: 
  1. an early version of the
  seasonal-naive baseline had a bug where its anchor silently fell back
  to a single flat value for most of each forecast month, discarding the
  daily/weekly pattern for ~75% of it -- found and fixed. 
  2. A quick 2-fold hyperparameter sweep initially suggested a bigger calibration
  improvement than the full 14-fold run actually confirmed -- caught by
  re-checking on all folds rather than trusting the small sample.
  3. On an initial implementation, LightGBM performed exceptionally well due
  to a leakage of future temperatures used to predict the future load of that
  same time. This is fixed now, and multiple tests prevent similar leakage from appearing.
