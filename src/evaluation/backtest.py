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
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class Fold:
    test_task: int
    train_df: pd.DataFrame
    test_df: pd.DataFrame


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

        # Hard safety check: no test timestamp may appear in train.
        overlap = set(train_df["timestamp"]) & set(test_df["timestamp"])
        if overlap:
            raise AssertionError(
                f"Leakage detected for test_task={t}: {len(overlap)} timestamps "
                f"appear in both train and test."
            )

        folds.append(Fold(test_task=t, train_df=train_df, test_df=test_df))

    return folds
