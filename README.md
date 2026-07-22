# GEFCom2014-L Probabilistic Load Forecasting

> Engineering detail, design rationale, and everything beyond the four
> points below live in [`AI_SUMMARY.md`](AI_SUMMARY.md) (AI-generated,
> kept for our own reference -- not part of this required README).

## Problem
One-month-ahead hourly probabilistic electricity load forecasting on the
GEFCom2014 load track. For each hour in a forecast month, the task is to
predict the full distribution as 99 quantiles (1st-99th percentile),
scored with pinball loss, evaluated across the dataset's 15 built-in
rolling forecast rounds (each round trains on all history released so
far and forecasts the next month).

## Reproducing results end to end

### Setup
```
python -m venv .venv
.venv\Scripts\activate      # Windows; use `source .venv/bin/activate` on macOS/Linux
pip install -r requirements.txt
```

### Getting the data
1. Run: `python .\src\data\download_data` (uses `kagglehub`, requires a Kaggle account/API token configured).
2. Or by hand: download `GEFCom2014-L_V2` from [Kaggle](https://www.kaggle.com/datasets/cthngon/gefcom2014-dataset) and unzip the `Load` folder's `Task 1` .. `Task 15` subfolders into `src/data/raw/Load/`.

Either way, the raw CSVs are expected at `src/data/raw/Load/Task <n>/L<n>-train.csv`.

### Running the pipeline
```
python -m src.eda.explore_data                    # brief, independent data overview -> reports/eda/
python -m src.evaluation.run_baseline             # both baselines across all rolling folds
python -m src.evaluation.run_model                # LightGBM (full feature set) across all rolling folds
python -m src.evaluation.run_comparison           # baselines + model together: pinball table, both
                                                   # significance tests, and the calibration reliability
                                                   # diagram (all in one run) -> reports/run_comparison/
pytest tests/ -v                                  # 34 tests
```

The first run of any `run_*.py` script (or `src.eda.explore_data`) parses
the raw per-task CSVs, builds features, and caches the combined DataFrame
to `src/data/processed/full_load.csv`; later runs load straight from that
cache. Delete the cache file (or pass `force_rebuild=True` to `get_data`)
after changing the loader or feature code.

## Model vs. baseline comparison
Rolling-origin backtest, tasks 2-15 (14 folds), mean pinball loss ± std,
and mean 90%-interval coverage (target ~0.90):

| method            | mean pinball loss | std  | mean coverage@90% |
|-------------------|--------------------|------|--------------------|
| **lightgbm (full)** | **2.97**         | 1.31 | 0.803              |
| climatology       | 10.06              | 4.36 | 0.898              |
| seasonal-naive    | 10.69              | 4.48 | 0.826              |

*(Numbers above are from the 27-quantile grid used during development;
being refreshed for the full 99-quantile grid the assignment asks for --
see `reports/run_comparison/` for the latest run.)*

Two independent statistical tests across the 14 folds -- a paired t-test
and a Diebold-Mariano test (`run_comparison.py`) -- both put LightGBM
ahead of both baselines with p ≈ 0.0, and it has the lower pinball loss
on every single fold, not just on average.

**Calibration**: LightGBM's much lower pinball loss does not mean its
intervals are better calibrated. Its coverage@90% (0.803) is further
from the 0.90 target than climatology's (0.898) -- the reliability
diagram (`reports/run_comparison/calibration.png`) shows its upper
quantiles are systematically a bit too low, so real load exceeds the
predicted upper bound more often than it should. See `AI_SUMMARY.md` for
the full diagnosis and the fixes already applied (quantile-crossing
correction, a small hyperparameter adjustment) versus what's still open.

## Limitations and unsuccessful approaches

- **LightGBM's calibration gap isn't fully closed.** A quantile-crossing
  fix and a small hyperparameter adjustment improved it slightly but
  didn't close the gap to 0.90 coverage -- the deeper issue is that
  independently-trained per-quantile models have no mechanism to jointly
  target a coverage level. Conformalized Quantile Regression (CQR) is
  the principled next step; not implemented (see `AI_SUMMARY.md`).
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
- **Unsuccessful approaches along the way**: an early version of the
  seasonal-naive baseline had a bug where its anchor silently fell back
  to a single flat value for most of each forecast month, discarding the
  daily/weekly pattern for ~75% of it -- found and fixed. A quick 2-fold
  hyperparameter sweep initially suggested a bigger calibration
  improvement than the full 14-fold run actually confirmed -- caught by
  re-checking on all folds rather than trusting the small sample.
