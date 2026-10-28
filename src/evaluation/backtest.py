"""
Rolling-origin backtest harness.

GEFCom2014-L already comes pre-split into 15 rolling rounds ("tasks").
Task t's train file contains all history released up to and including
round t. The convention we use here:

    - For test round t (t >= first_test_task), the TRAINING set is every
      row released in Task (t-1)'s file, i.e. everything known BEFORE
      round t's data was revealed.
    - The TEST set is the rows that are newly present in Task t but not
      in Task (t-1) — i.e. exactly the one month that round t reveals.

This guarantees no row used for training in fold t was generated at or
after the forecast period of fold t, which is the leakage protection
requirement in the assignment brief.

Extended folds (make_monthly_folds): the months BEFORE the first task
fold (e.g. Jan 2008 - Sep 2010) are backtested the same way -- train on
every row strictly before the month, test on the month -- for more test
months. They are numbered on from the task folds (Sep 2010 = fold 1,
Aug 2010 = fold 0, ...), so folds stay ordered in time, and a fold < the
first task fold is an "extended" fold.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.features.build_features import operating_day_timestamps


@dataclass
class Fold:
    test_task: int
    train_df: pd.DataFrame
    test_df: pd.DataFrame


def _check_no_leakage(label, train_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    # Hard safety check: no test timestamp may appear in train.
    overlap = set(train_df["timestamp"]) & set(test_df["timestamp"])
    if overlap:
        raise AssertionError(
            f"Leakage detected for test_task={label}: {len(overlap)} timestamps "
            f"appear in both train and test."
        )
    # ...and all of train must lie strictly before the test month, so a
    # mislabelled task can't put future rows into training.
    if train_df["timestamp"].max() >= test_df["timestamp"].min():
        raise AssertionError(
            f"Leakage detected for test_task={label}: training data reaches "
            f"{train_df['timestamp'].max()}, test starts {test_df['timestamp'].min()}."
        )


def make_monthly_folds(full_df: pd.DataFrame, first_month: str, first_test_task: int) -> list[Fold]:
    """One fold per calendar month from `first_month` up to the month
    before task `first_test_task`'s test month. A month is its hour-ending
    operating days (1st 01:00 to the next 1st 00:00), as in the task
    files; training is every row strictly before the month."""
    first_task_start = full_df.loc[full_df["task"] == first_test_task, "timestamp"].min()
    first_task_month = operating_day_timestamps(pd.Series([first_task_start])).dt.to_period("M").iloc[0]
    months = pd.period_range(pd.Period(first_month, "M"), first_task_month - 1, freq="M")
    operating_month = operating_day_timestamps(full_df["timestamp"]).dt.to_period("M")
    folds = []
    for month in months:
        label = first_test_task - (first_task_month - month).n
        test_df = full_df[operating_month == month].copy()
        if test_df.empty or test_df["load"].isna().all():
            raise ValueError(f"No load to test on in {month} (fold {label})")
        train_df = full_df[full_df["timestamp"] < test_df["timestamp"].min()].copy()
        _check_no_leakage(label, train_df, test_df)
        folds.append(Fold(test_task=label, train_df=train_df, test_df=test_df))
    return folds


def make_rolling_folds(full_df: pd.DataFrame, first_test_task: int, last_test_task: int) -> list[Fold]:
    """
    full_df must have columns: timestamp, load, task, <temperature cols...>
    where `task` marks the task release a row came from (see
    src/data/loader.py -- each task's file is a pure continuation of the
    previous one, so no timestamp appears under more than one task).
    """
    folds = []
    for t in range(first_test_task, last_test_task + 1):
        train_df = full_df[full_df["task"] < t].copy()
        test_df = full_df[full_df["task"] == t].copy()

        if train_df.empty or test_df.empty:
            raise ValueError(
                f"Empty train or test set for test_task={t}. "
                f"Check that `task` column reflects release order correctly."
            )

        _check_no_leakage(t, train_df, test_df)

        folds.append(Fold(test_task=t, train_df=train_df, test_df=test_df))

    return folds
