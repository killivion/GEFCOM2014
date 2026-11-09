# GEFCom2014 Probabilistic Load Forecasting

![CI](https://github.com/killivion/GEFCOM2014/actions/workflows/ci.yml/badge.svg)

> Engineering detail and design rationale beyond this page live in
> [`AI_SUMMARY.md`](AI_SUMMARY.md).

## Problem
Forecast the full distribution of hourly electricity load for a whole month,
one month ahead, as 99 quantiles (1st-99th percentile), on the GEFCom2014-L
dataset (hourly load of one US utility plus temperatures from 25 weather
stations, released in 15 rolling rounds). The score is the pinball (quantile)
loss averaged over all hours and quantiles; lower is better.

**What is known at forecast time.** A forecast for month *m* is issued at the
end of month *m-1*. Everything up to that point (load and temperature) may be
used; nothing from month *m* may. In particular the actual temperature of the
forecast month is **not** known. The model therefore forecasts *with weather
uncertainty*: it is trained on past observed temperatures and predicts the
test month under temperature scenarios taken from earlier years (see "Model").

## Results

Rolling-origin backtest over GEFCom2014 tasks 2-15 = 14 monthly folds
(Oct 2010 - Nov 2011), each trained on everything released before it. A
fold is named after the task whose file releases its test month, so fold
*t* is GEFCom2014 competition task *t-1* (fold 2 = Oct 2010 = competition
task 1). All numbers: 99 quantiles. The ablation and oracle runs are from
commit `a6e563d`; the main comparison was re-run at `0a02e17` to add the
official benchmark and reproduced every earlier loss exactly (the model
code is unchanged between the two; see `reports/*/run_info.json`).

**Evaluation protocol.** After an external review, folds 2-8 (Oct 2010 -
Apr 2011) became the *development* folds: every later modelling decision
was made on them only. Folds 9-15 (May - Nov 2011) are the *holdout*,
evaluated once, in the final run. The holdout row is the headline.

### Headline: holdout (7 months, May - Nov 2011)

| method | mean pinball loss | 90% coverage (target 0.90) |
|---|---|---|
| **final model** (LightGBM, temperature scenarios, recency weighting) | **6.79** | 0.874 |
| seasonal climatology, recency-weighted (primary baseline) | 7.32 | 0.862 |
| seasonal climatology | 7.48 | 0.861 |
| seasonal naive (same hour, 52 weeks earlier) | 9.28 | 0.931 |
| climatology (hour x weekday, all months pooled) | 12.81 | 0.916 |
| official GEFCom2014 benchmark (same date last year, one value for all 99 quantiles) | 12.83 | 0.003 |
| *oracle: final model given the actual test-month temperature (not available in practice)* | *3.86* | *0.872* |

Improvement of the final model on the holdout (loss reduction = how much lower the model's loss is):

| vs. | loss reduction | months won | hourly tests (DM, block bootstrap) | month-level tests (month bootstrap, Wilcoxon) | conservative p | Holm |
|---|---|---|---|---|---|---|
| seasonal climatology, recency-weighted | **7.2%** (0.53) | 4/7 | DM p = 0.028, CI [0.09, 1.06] | CI [-0.09, 1.29], p = 0.12; Wilcoxon p = 0.38 | 0.38 | 0.38 |
| seasonal climatology | **9.2%** (0.69) | 5/7 | DM p = 0.020, CI [0.17, 1.32] | CI [0.02, 1.47], p = 0.04; Wilcoxon p = 0.16 | 0.16 | 0.31 |
| seasonal naive | **26.6%** (2.47) | 7/7 | DM p = 0.0001 | p < 0.001; Wilcoxon p = 0.016 | 0.016 | 0.078 |
| climatology | **47.0%** (6.04) | 7/7 | DM p < 0.0001 | p < 0.001; Wilcoxon p = 0.016 | 0.016 | 0.078 |
| official GEFCom2014 benchmark | **46.9%** (6.02) | 7/7 | DM p < 0.0001 | p < 0.001; Wilcoxon p = 0.016 | 0.016 | 0.078 |

How to read this:
- The final model has the lowest holdout loss of every honest method
  (bold above is the size of the gap, not a claim of significance). It
  beats the seasonal-naive and naive-climatology baselines and the official
  benchmark in every single month. Against the two seasonal climatologies it is better on average by
  7-9%, and the hour-level tests call that significant (p ~ 0.02-0.03), but
  it wins only 4-5 of 7 months, so the month-level tests do not: **the gain
  over the strongest baseline is not established** with 7 holdout months.
- "Conservative p" is the largest p-value of all tests for that comparison.
  With 7 months, the exact Wilcoxon p cannot go below 0.0156, so after the
  Holm correction for 5 baselines nothing can go below 0.078: no month-level
  claim on 7 months can be significant at 5% after correction. Over all 14
  months the gains over the official benchmark (43.3%, 14/14 months) and
  over seasonal naive (24.2%, 13/14) are significant after correction
  (Holm p = 0.0006 and 0.012).
- The official benchmark is the organisers' file shipped with each task
  (`Task t/Lt-benchmark.csv`, scored by `src/data/official_benchmark.py`,
  which checks that each file covers exactly its fold's hours). It is a
  point forecast copied to all 99 quantiles, so its "90% interval" has zero
  width. The competition scored tasks 4-15; our folds 5-15 are competition
  tasks 4-14 (task 15, Dec 2011, is not in the backtest).
- The oracle row shows that knowing the month's actual weather would cut
  the loss by 43% on the holdout (by more than half on the development
  folds and over all 14 folds). Weather uncertainty, not the model, dominates
  a month-ahead forecast; this is also why the earlier headline (which used
  the actual temperature) looked so good.

### Development folds and all 14 folds

| method | development (Oct-Apr) | all 14 folds |
|---|---|---|
| final model | **8.78** | **7.79** |
| seasonal climatology, recency-weighted | 8.79 | 8.06 |
| seasonal climatology | 9.12 | 8.30 |
| seasonal naive | 11.35 | 10.31 |
| climatology | 12.80 | 12.81 |
| official GEFCom2014 benchmark | 14.75 | 13.79 |
| oracle | 3.28 | 3.57 |

On the development folds the final model only ties the recency-weighted
seasonal climatology (8.78 vs 8.79). Over all 14 folds it is 3.3% better
than it (not significant, conservative p = 0.39) and 6.2% better than the
plain seasonal climatology (DM p = 0.005, month bootstrap p = 0.017,
9/14 months, Wilcoxon p = 0.058).

### Ablation: what each part contributes

Same backtest, same hours; each row changes one thing (`configs/ablation_*.yaml`).

| model | development | holdout | all 14 | 90% cov. (all) |
|---|---|---|---|---|
| plain: climatology temperature input, no scenarios, no recency | 9.56 | 6.93 | 8.25 | 0.852 |
| + recency weighting only | 9.16 | **6.74** | 7.95 | 0.850 |
| + temperature scenarios only | 8.96 | 6.98 | 7.97 | 0.879 |
| **final: scenarios + recency** | **8.78** | 6.79 | **7.79** | 0.880 |

- Final vs plain: 8.2% lower loss on the development folds (7/7 months,
  conservative p = 0.034) and 5.6% over all 14 folds (11/14, conservative
  p = 0.032), but only 2.0% on the holdout (4/7, not significant).
- On the holdout, recency weighting alone was slightly better than the final
  model (6.74 vs 6.79, not significant). The temperature scenarios improve
  the 90% coverage (0.85 -> 0.88) everywhere, but their pinball-loss gain
  seen on the development (winter) folds did not carry over to the summer
  holdout. Both components were chosen on the development folds before the
  holdout was run; they are reported as they came out.

### Calibration
`reports/run_comparison/calibration.png`. The final model's 90% intervals
cover 88% of hours (target 90%). Its quantile curve is shifted: the predicted
median has only 44.8% of observed load below it, i.e. the model tends to
under-forecast slightly (mean absolute calibration gap 0.039 across the 99
levels, vs 0.022 for the recency-weighted seasonal climatology).

The shift is not a stable bias. The share of hours below the predicted
median ranges from 5% (Dec 2010, the record cold month) to 69% (Oct 2011)
and changes sign from month to month with the weather. That is why
sequential conformal recalibration (experiment log) did not help: earlier
months' errors do not predict the next month's. Calibrating on the same
season of earlier years does lower the loss on the development folds, but
not reliably on the holdout, and the final model stays unadjusted (see
"Secondary analysis: seasonal calibration" below).

## TabPFN vs. LightGBM benchmark

Is a tabular foundation model better than gradient boosting here, and by
how much? [TabPFN](https://priorlabs.ai) is a transformer pre-trained on
millions of synthetic datasets; it does not train on our data but reads
the training rows as context and predicts in one forward pass, returning
a full predictive distribution per row. It runs through the Prior Labs API
(`tabpfn-client`; the API's default model, v3.5 during these runs, is
recorded in each `run_info.json`). Both models are scored on identical test hours,
quantile levels (99) and metrics: accuracy (pinball loss), calibration
(90% coverage, mean |empirical - nominal| over the levels), runtime per
fold, and Diebold-Mariano / bootstrap / Wilcoxon tests (`configs/tabpfn_benchmark.yaml`,
fixed before any TabPFN result was seen).

**Two tasks:**
- *Time series* -- the month-ahead forecast on all 14 folds, reported as
  development (folds 2-8) and holdout (folds 9-15). LightGBM is the final
  model exactly as reported. TabPFN gets the same features and temperature
  scenarios and default settings, in two variants fixed in advance:
  `tabpfn` (primary) with every training row as context, equally weighted,
  and `tabpfn_recency` with a 20,000-row sample whose inclusion
  probability is proportional to LightGBM's recency weight (TabPFN takes
  no sample weights).
- *Tabular* -- interpolation, not forecasting: the history before the
  first forecast month (2005 - Sep 2010) cut into calendar weeks, assigned
  at random to 5 folds; training rows within 7 days of a test week are
  dropped (embargo), so a week's neighbours cannot leak its weather
  anomaly. Climatology temperature for every row. This is the i.i.d.
  setting TabPFN was pre-trained for; its numbers are not forecast
  accuracy.

Neither model ever sees the observed temperature or load of an hour it
predicts (tests in `tests/test_tabpfn_benchmark.py`). Every forecast is
also recalibrated (conformal, three methods) from other test data only --
earlier test months, or the other folds in the tabular task; "additive"
was fixed in advance as the one tested for significance. Seasonal
calibration was added later and is reported separately, as a secondary
analysis.

Reading the numbers: TabPFN's runtime is the API's (their GPU plus
network), LightGBM's is local (6 CPU cores); both include building the
features, and the summary uses the median per fold (one LightGBM fold's
wall-clock time includes the machine sleeping; see `runtime_by_fold.csv`).
The whole benchmark used about 1.5M of the free plan's 20M monthly API
credits (~50k per time-series fold), plus about 2.2M for the seasonal
calibration history. With 5 tabular folds the fold-level Wilcoxon test cannot go
below p = 0.0625, so the conservative p of the tabular task is never
below that; the hourly tests (DM, week bootstrap) carry the evidence there.

**Caveats, checked in a self-audit** (details in `AI_SUMMARY.md`):
- *The development folds favour LightGBM*: its settings were chosen on them,
  TabPFN's were not tuned at all. The holdout (folds 9-15) is the fair
  comparison.
- *Shared code, shared failure*: both models use the same input, scenario,
  metric and calibration code, so a bug there would hit both and hide in
  the comparison. Guards: LightGBM reproduces the published losses on all
  14 folds to 1e-15, both models are checked to be scored on identical
  hours and loads, and TabPFN's quantile order was verified on the API
  before the per-row sort that would otherwise mask it.
- *Multiple comparisons*: p-values are Holm-adjusted across every
  TabPFN-vs-LightGBM comparison of a fold group (variants x calibrations).
- Whether the API uses a 50k-row context whole or subsamples it internally
  is not reported; the measured gain from 10k to 50k rows shows it uses
  more than 10k.

**Tabular task** (5 week-block folds, 2005 - Sep 2010; pinball loss, lower is better):

| method | raw | 90% cov. | + additive conformal | 90% cov. |
|---|---|---|---|---|
| LightGBM | **8.38** | 0.780 | **8.29** | 0.900 |
| TabPFN (all training rows) | 8.60 | 0.768 | 8.46 | 0.899 |

TabPFN is 2.7% worse raw and 2.0% worse after recalibration, in 0 of 5
folds better (raw: DM p = 0.020, week bootstrap p = 0.034; conservative
p = 0.074, bounded below by the 5-fold Wilcoxon test). Both models' raw
intervals are much too narrow (coverage 0.77-0.78 for a 90% interval);
recalibration from the other folds fixes that for both. TabPFN took about
11 s per fold through the API, LightGBM (99 quantile models) about 59 s
locally.

**Time series** (month-ahead, 14 folds; pinball loss, lower is better; 90% coverage in brackets):

| method | development (Oct-Apr) | holdout (May-Nov) | all 14 |
|---|---|---|---|
| **LightGBM** (final model) | **8.78** (0.89) | **6.79** (0.87) | **7.79** (0.88) |
| TabPFN, all training rows | 9.12 (0.85) | 7.84 (0.82) | 8.48 (0.84) |
| TabPFN, recency-weighted 20k rows | 8.92 (0.87) | 7.22 (0.83) | 8.07 (0.85) |
| LightGBM + additive conformal | 9.08 (0.84) | 6.95 (0.88) | 8.02 (0.86) |
| TabPFN + additive conformal | 8.88 (0.84) | 7.43 (0.88) | 8.15 (0.86) |
| TabPFN recency + additive conformal | 8.86 (0.84) | 7.24 (0.88) | 8.05 (0.86) |

- On the holdout -- the fair comparison -- LightGBM is better than every
  TabPFN variant: by 15.4% (TabPFN, 0/7 months), 6.2% (recency variant,
  1/7) and still 4.2-6.9% after sequentially recalibrating both. The
  hourly tests are clear (DM p <= 0.003); the conservative, Holm-adjusted
  p is 0.094 (0.14 for the raw recency variant), against a floor of 0.094
  set by 7 months and 6 comparisons (these 4 plus the 2 seasonal ones
  below).
- On the development folds the gap is smaller (TabPFN 4.0% worse, 2/7,
  not significant), and recalibrated TabPFN beats recalibrated LightGBM
  by 2.2-2.5% -- but recalibration makes LightGBM worse, and raw
  LightGBM (8.78) stays ahead of every pre-declared TabPFN variant.
- Raw TabPFN under-forecasts: only 32% of loads fall below its median
  (LightGBM 45%), on average 13.6 MW too low (LightGBM 8.2). Load rose from
  ~135-139 MW (2005-06) to 147-161 MW (2008-10); an equally weighted
  context pulls TabPFN towards the older, lower level. The recency-weighted
  context reduces that (11.0 MW; 6.2% instead of 15.4% behind on the
  holdout), and recalibration from earlier months corrects more of it --
  calibration helps TabPFN on both fold groups, unlike LightGBM.
- Runtime per fold (median): TabPFN 49 s through the API (2 s upload, 45 s
  for ~55k scenario rows), the recency variant 34 s; LightGBM 436 s for 99
  quantile models on 6 local cores. About 1.4M API credits for both
  TabPFN variants over the 14 folds.

### Secondary analysis: seasonal calibration (post-hoc)

This does not change the headline or the final model, which stays
unadjusted LightGBM. The idea came after the LightGBM holdout and a first
TabPFN pass had been seen, so it is not a clean confirmation; only its
window was fixed before any seasonal result.

Each scored month is recalibrated on all earlier forecast months within
one calendar month of its own (Jan 2011 uses Dec-Feb of 2008-2010 and
Dec 2010). The history comes from extra forecasts for Jan 2008 - Sep 2010
(`--calibration-history`), which are never scored themselves.

| method (pinball loss, 90% coverage) | development | holdout | all 14 |
|---|---|---|---|
| LightGBM, raw | 8.78 (0.89) | 6.79 (0.87) | 7.79 (0.88) |
| LightGBM + seasonal additive | 8.36 (0.88) | 6.88 (0.83) | 7.62 (0.86) |
| LightGBM + seasonal scaled | 8.32 (0.88) | 6.79 (0.85) | 7.56 (0.87) |
| LightGBM + seasonal PIT | 8.27 (0.89) | 6.77 (0.85) | 7.52 (0.87) |
| TabPFN recency, raw | 8.92 (0.87) | 7.22 (0.83) | 8.07 (0.85) |
| TabPFN recency + seasonal additive | 8.23 (0.88) | 7.04 (0.85) | 7.63 (0.86) |
| TabPFN recency + seasonal PIT | 8.14 (0.89) | 6.91 (0.86) | 7.53 (0.87) |
| TabPFN, all rows + seasonal additive | 8.29 (0.88) | 7.18 (0.85) | 7.74 (0.87) |

All variants, including seasonal scaled for TabPFN, are in
`reports/tabpfn_benchmark/timeseries/summary.csv`.

- **Development folds:** seasonal beats sequential calibration for every
  model, and beats the raw forecasts too (LightGBM 8.27-8.36 vs 8.78).
- **Holdout, LightGBM:** no useful gain (6.77-6.88 vs 6.79), and calibration
  gets worse: 90% coverage falls from 0.87 to 0.83-0.85, and the mean
  absolute calibration gap rises from 0.011 to about 0.10. That is why it
  is not adopted.
- **Holdout, TabPFN:** it helps more because raw TabPFN under-forecasts
  (recency variant 7.22 to 6.91-7.04), but LightGBM stays ahead.
- **TabPFN vs. LightGBM, both seasonal additive** (the pre-declared
  calibration; 2 of the 6 comparisons in the Holm family): LightGBM is
  4.5% (TabPFN) and 2.4% (recency) better on the holdout, 2/7 months each,
  Holm p = 0.22. Over all 14 months the recency variant is level with
  LightGBM (7.63 vs 7.62, Holm p = 1.0).

## What changed after the external review
The review of the first submission raised three points; all three held.

1. **The headline used the forecast month's actual temperature.** That
   value is not known a month ahead. It is now used only for the labelled
   oracle upper bound (`configs/oracle_temperature.yaml`, which refuses to
   write into the honest report folder). The model instead uses temperature
   scenarios from earlier years.
2. **The baseline ignored seasonality.** Climatology now also exists per
   month (and recency-weighted, with the same 2-year half-life the model
   uses, so "recent years count more" is not something only the model gets).
   Seasonal naive now looks back 52 weeks instead of 1 week. The primary
   baseline is the stronger of the two seasonal climatologies on the
   development folds.
3. **The statistical comparison was too basic** (a paired t-test on 14 fold
   means, and a Diebold-Mariano test identical to it). Significance is now
   tested on per-hour losses with Diebold-Mariano at 1-week and 1-month
   Newey-West lags, a circular block bootstrap, a bootstrap over whole
   months, and Wilcoxon on month losses; the most conservative p is
   Holm-corrected per family (baselines, model variants), with the oracle
   kept out.

The result of the earlier version, for reference: with honest inputs its
loss is 8.25 over all 14 folds (the `plain` ablation row), against the
3.65 originally reported with the actual temperature (earlier code
version; the oracle row is today's equivalent, 3.57).

## Model
- **LightGBM quantile regression**, one model per quantile level (99
  models), with a monotone rearrangement so quantiles never cross.
- **Features:** calendar (hour, weekday, month, day of year, weekend,
  holiday, cyclic encodings; hour-ending timestamps, so 00:00 belongs to
  the previous day) and temperature (station mean, heating and cooling
  degrees). No load-based features (see below).
- **Temperature scenarios** ("weather ensemble"): trained on past observed
  temperatures; the test month is predicted under the temperatures of the
  same dates in every earlier year, shifted by -7..+7 days (9-10 years x 15
  shifts ~ 140 scenarios), and the scenario forecasts are combined as a
  probability mixture.
- **Recency weighting:** training hours weighted 0.5^(age / 2 years).

Leakage protection is tested: the backtest asserts training ends before the
test month; the model drops the test month's load column before building
any feature; scenario temperatures are read only from training rows before
the cutoff (tests corrupt the test month's temperature and add post-cutoff
rows, and the predictions must not change).

## Reproducing results

### Setup
```
python -m venv .venv
.venv\Scripts\activate      # Windows; on macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

### Data
Run `python src/data/download_data` (uses `kagglehub`), or download
`GEFCom2014-L_V2` from [Kaggle](https://www.kaggle.com/datasets/cthngon/gefcom2014-dataset)
and unzip the `Load` folder's `Task 1` .. `Task 15` subfolders into
`src/data/raw/Load/`.

### Pipeline
```
python -m src.data.loader                  # parse raw CSVs -> src/data/processed/ (cached, versioned)
python -m src.eda.explore_data             # short data overview -> reports/eda/
pytest tests/                              # 122 tests, synthetic data only

# Final model + all baselines + official benchmark, 14 folds -> reports/run_comparison/  (~70 min on 6 cores)
python -m src.evaluation.run_comparison --official-benchmark --save-predictions
# Sequential conformal recalibration on the saved predictions -> reports/run_conformal/  (~1 min)
python -m src.evaluation.run_conformal
# Oracle and ablations (each writes its own reports/ folder)
python -m src.evaluation.run_comparison --config configs/oracle_temperature.yaml --output-dir run_comparison_oracle
python -m src.evaluation.run_comparison --config configs/ablation_plain.yaml --output-dir ablation_plain
python -m src.evaluation.run_comparison --config configs/ablation_recency_only.yaml --output-dir ablation_recency_only
python -m src.evaluation.run_comparison --config configs/ablation_scenarios_only.yaml --output-dir ablation_scenarios_only
# Paired hour-by-hour comparison of the runs -> reports/run_ablation/
python -m src.evaluation.compare_runs --primary final=run_comparison --runs plain=ablation_plain recency_only=ablation_recency_only scenarios_only=ablation_scenarios_only oracle=run_comparison_oracle --reference oracle
# TabPFN vs. LightGBM -> reports/tabpfn_benchmark/<task>/  (pip install -r requirements-tabpfn.txt; TABPFN_TOKEN)
python -m src.evaluation.run_tabpfn_benchmark --task tabular
python -m src.evaluation.run_tabpfn_benchmark --task timeseries   # --methods lightgbm tabpfn tabpfn_recency, --max-folds N
# Secondary analysis: forecasts for Jan 2008 - Sep 2010, then re-evaluates with the seasonal calibration
python -m src.evaluation.run_tabpfn_benchmark --task timeseries --calibration-history
# tabpfn-client 0.3.1 is the last release for Python 3.9 and stops working after 7 Jan 2027;
# re-running TabPFN later needs a newer Python and client.
```
`configs/default.yaml` is the single source of truth; the other configs are
generated from it by `python configs/make_variants.py`. For quick iteration:
`--config configs/dev.yaml --folds development` (20 quantiles; not
comparable with the 99-quantile numbers). Runs with recency weighting take
about 5x longer than without (LightGBM's weighted quantile objective).
`--baseline none --skip-stats` (model only; no significance tests,
calibration outputs or hourly losses) exists for the external experiment
agent that calls `run_comparison` hundreds of times; it doesn't change any
reported number.

Each `reports/<run>/` folder holds per-fold and per-hour losses, the summary
and significance tables per fold group, the calibration curve, and
`run_info.json` (commit, config). `predictions.npz` (all quantile
predictions, ~17 MB) is written locally but not committed. Note: the
`dirty: true` flag in the ablation and oracle `run_info.json` files comes from the check used at run time, which
looked at every tracked file -- including the report files the run itself
overwrites. `src/` and `configs/` were unchanged from `a6e563d` during the
runs; the check was changed afterwards to look at `src/` and `configs/`
only. Inside Docker there is no `.git`, so container runs record
`commit: null`.

## Docker and CI
Every push and pull request runs GitHub Actions
([`.github/workflows/ci.yml`](.github/workflows/ci.yml)): `pytest` on
Python 3.9, plus a Docker build with the test suite run inside the built
image.

- Tests in a container (no dataset needed): `docker build -t gefcom2014 .`
  then `docker run --rm gefcom2014` (or `docker compose run --rm test`).
- Full pipeline: put the Kaggle data in `src/data/raw/Load/`, create
  `src/data/processed/` and `reports/`, then `docker compose run --rm pipeline`.
  Raw data is mounted read-only; the processed cache and `reports/` are
  written back to the host.
- Data and reports are never baked into the image (see `.dockerignore`);
  the container runs as a non-root user.

## Limitations and disclosures
- **The holdout is not perfectly clean.** Folds 9-15 were visible in
  earlier full-backtest runs, before the development/holdout split existed.
  Choices made before the split, and therefore with the holdout in view:
  the baseline designs, the seasonal-climatology buckets, the 7-day window
  of the temperature climatology, the tree depth (`max_depth=4`, from an
  early 2-fold check whose folds were not recorded), the feature set, and the quantile grid. Only the
  choices made after the split are holdout-clean: temperature scenarios
  (shift range, mixture), recency weighting and its half-life, dropping
  post-processing, and the choice of primary baseline.
- **The split is confounded with season.** Development is mostly the heating
  season (Oct-Apr), holdout the cooling season (May-Nov). The holdout tests
  a different regime, which is honest but not an identically distributed
  comparison.
- **One test year.** All 14 test months come from Oct 2010 - Nov 2011, so
  the evidence rests on one year of weather. Dec 2010 (the coldest December in
  the history) is the worst month for every method, and Aug 2011 contains
  the Hurricane Irene outages; see `AI_SUMMARY.md`.
- **Selection on the development folds.** 4 scenario settings and 2
  recency half-lives were tried there (see the experiment log); the chosen
  settings are therefore slightly optimistic on those folds.
- **Calibration.** 90% coverage is 0.87-0.89, not 0.90, and the median is
  slightly too low.
- **The official benchmark is weak.** Beating it by 43-47% shows the model
  is far better than a naive point forecast, not that it would have ranked
  high in the competition; the seasonal climatologies are the meaningful
  comparison.

## Experiment log (development folds 2-8, 20-quantile grid)
These are development-grid numbers, used only for decisions; they are not
comparable with the 99-quantile results above. Reference: plain model with
climatology temperature 9.13-9.15, seasonal climatology 8.76.

| idea | variants | best result | decision |
|---|---|---|---|
| temperature scenarios | shifts 0 / ±4 / ±7, mixture vs quantile averaging | ±7 mixture 8.61 (quantile averaging: coverage 0.52) | **kept** |
| recency weighting | half-life 2 / 4 years | 2 years: 8.75; on top of scenarios 8.43 | **kept** |
| load-level normalization (trailing year, same month last year) | 2 | 9.13 / 9.28 | dropped: load has no steady trend, the level carries last year's weather |
| post-processing: recalibration and blend with seasonal climatology (inner validation inside each fold) | 5 + 2 | 8.47 on the plain model; no gain on the final model (8.43 -> 8.43 / 8.48) | dropped |
| sequential conformal recalibration from the model's own earlier test months (additive / interval-scaled / PIT; all earlier months or last 3)** | 6 | every variant worse: best 9.08 vs 8.78 unadjusted (holdout, all variants: 6.83-6.98 vs 6.79) | dropped: the selection keeps "none" |
| temperature dynamics on top of the final model: 24h moving average + same hour the day before; + daily max/min; + 72h moving average (each scenario uses its own temperature path) | 3 | 8.44 / 8.46 / 8.45 vs 8.43 without (coverage 0.890 -> 0.898) | dropped: the scenario mixture already spans the weather; path features add nothing a month ahead |
| recursive 24h/168h load lags fed by the model's own median | 1 | 9.81 vs 8.17 without lags, same run* | dropped: errors compound over the month |
| load-level features at the forecast origin | 2 | 8.65-8.69 vs 7.92 without, same run* | dropped: one value per month, trees overfit |

\* These two were run before the development/holdout split, on all 14
folds (20-quantile grid, last-year resp. climatology temperature input).
\*\* Final model, 99 quantiles, `reports/run_conformal/`. The selection used
the development folds only; the holdout column is shown for transparency.

Earlier work, before the review:
1. An early seasonal-naive baseline silently fell back to a flat value for
   ~75% of each month; found and fixed.
2. A 2-fold hyperparameter check overstated a calibration gain that the
   14-fold run did not confirm.
3. An early model leaked future temperatures into the features. That was
   fixed, but the headline still used the actual forecast-month temperature
   as an input, which is the issue the review caught (point 1 above).
