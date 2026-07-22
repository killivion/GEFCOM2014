import pandas as pd
import pytest

from src.evaluation.backtest import make_rolling_folds


def _make_fake_data():
    # 3 "tasks": each task adds one more day of hourly data.
    # task 1: day 1 only (this is training-only history)
    # task 2: day 1 + day 2 (day 2 is the new test period for fold t=2)
    # task 3: day 1 + day 2 + day 3 (day 3 is the new test period for fold t=3)
    rows = []
    for day, task_introduced in [(1, 1), (2, 2), (3, 3)]:
        for hour in range(24):
            ts = pd.Timestamp("2020-01-01") + pd.Timedelta(days=day - 1, hours=hour)
            rows.append({"timestamp": ts, "load": 100.0 + hour, "task": task_introduced})
    return pd.DataFrame(rows)


def test_folds_have_no_timestamp_overlap():
    df = _make_fake_data()
    folds = make_rolling_folds(df, first_test_task=2, last_test_task=3)

    assert len(folds) == 2
    for fold in folds:
        train_ts = set(fold.train_df["timestamp"])
        test_ts = set(fold.test_df["timestamp"])
        assert train_ts.isdisjoint(test_ts), f"Leakage in fold test_task={fold.test_task}"


def test_train_set_only_contains_earlier_tasks():
    df = _make_fake_data()
    folds = make_rolling_folds(df, first_test_task=2, last_test_task=3)

    for fold in folds:
        assert (fold.train_df["task"] < fold.test_task).all()
        assert (fold.test_df["task"] == fold.test_task).all()


def test_raises_on_manually_injected_leakage():
    df = _make_fake_data()
    # Corrupt the data: give one test-period row the wrong (earlier) task
    # label so it would end up in BOTH train and test if the guard didn't
    # catch it. This simulates a bug in the loader mislabeling a row.
    leaked = df.copy()
    test_task_2_mask = leaked["task"] == 2
    leaked_idx = leaked[test_task_2_mask].index[0]
    duplicate_row = leaked.loc[[leaked_idx]].copy()
    duplicate_row["task"] = 1  # pretend this same timestamp was also in task 1's training data
    corrupted = pd.concat([leaked, duplicate_row], ignore_index=True)

    with pytest.raises(AssertionError):
        make_rolling_folds(corrupted, first_test_task=2, last_test_task=3)


def test_empty_fold_raises_clear_error():
    df = _make_fake_data()
    with pytest.raises(ValueError):
        make_rolling_folds(df, first_test_task=10, last_test_task=12)
