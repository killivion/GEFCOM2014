# GEFCom2014-L Probabilistic Load Forecasting

## Problem
One-month-ahead hourly probabilistic electricity load forecasting on the
GEFCom2014 load track. For each hour in a forecast month, the task is to
predict the full distribution as 99 quantiles (1st-99th percentile),
scored with pinball loss, evaluated across the dataset's 15 built-in
rolling forecast rounds.

## Status
Skeleton stage: project structure, data loader, rolling-origin backtest
harness (with a leakage guard), pinball/coverage/calibration metrics, and
two baselines (seasonal-naive, climatology) are implemented.
Feature engineering and the main models (LightGBM quantile regression) are
next.

## Setup
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

## Getting the data
1 Run `src/data/download_data`. 
2 Otherwise: download `GEFCom2014-L_V2` from Kaggle (https://www.kaggle.com/datasets/cthngon/gefcom2014-dataset) and unzip the `load` folder into `data/raw/load/`.

## Running tests - TO-DO
pytest tests/ -v

## Reproducing results - TO-DO
(To be filled in as the pipeline is completed — will be a single command
like `python -m src.run_experiment --config configs/default.yaml`.)

## Repo layout

```
configs/            experiment configuration (quantile levels, model
                     params, leakage assumptions, backtest fold range)
src/data/loader.py   parses Task 1..15 folders into one tidy DataFrame,
                     tagging each row with the task it was released in
src/evaluation/
  metrics.py         pinball loss, interval coverage, calibration curve
  backtest.py        rolling-origin fold generator + leakage assertion
src/models/
  baselines.py       seasonal-naive and climatology quantile baselines
src/features/
  build_features.py  calendar + temperature feature engineering (WIP)
tests/               pytest suite for metrics correctness and the
                     leakage guard
```
