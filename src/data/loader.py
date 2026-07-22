"""
Parses the GEFCom2014-L_V2/load folder (Task 1 ... Task 15) into a single
tidy, time-indexed DataFrame, tagging each row with the task round. 
Tagging is used for correct rolling-origin backtesting: e.g. training data
has to be <t for a given time point.

Columns:
zone_id, date/timestamp columns, load, and w1..w25
(hourly temperature from 25 weather stations).

Column names have varied slightly across re-uploads of this dataset, so
this loader inspects the actual header of the first file it reads and
adapts rather than hard-coding exact names. It will raise a clear error
if it can't find a load column or a temperature column at all.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass
class LoadedData:
    df: pd.DataFrame          # tidy frame: timestamp, task, load, temp_* columns
    temp_cols: list[str]      # names of the temperature station columns


def _find_task_dirs(raw_load_dir: Path, n_tasks: int) -> list[tuple[int, Path]]:
    found = []
    for t in range(1, n_tasks + 1):
        # folder naming has been seen as both "Task 1" and "Task1"
        candidates = [raw_load_dir / f"Task {t}", raw_load_dir / f"Task{t}"]
        match = next((c for c in candidates if c.exists()), None)
        if match is None:
            raise FileNotFoundError(
                f"Could not find folder for Task {t} under {raw_load_dir}. "
                f"Tried: {candidates}. Check configs/default.yaml:data.raw_load_dir."
            )
        found.append((t, match))
    return found


def _find_train_csv(task_dir: Path, task_num: int) -> Path:
    candidates = [
        task_dir / f"L{task_num}-train.csv",
        task_dir / f"L{task_num}-train.CSV",
    ]
    match = next((c for c in candidates if c.exists()), None)
    if match is None:
        # fall back: any csv in the folder with "train" in the name
        hits = list(task_dir.glob("*train*.csv")) + list(task_dir.glob("*train*.CSV"))
        if not hits:
            raise FileNotFoundError(f"No train csv found in {task_dir}")
        match = hits[0]
    return match


def _identify_columns(columns: list[str]) -> tuple[str, str, list[str]]:
    """Returns (date_col, load_col, temp_cols) from a raw header."""
    lower = {c: c.lower() for c in columns}

    date_col = next((c for c in columns if lower[c] in ("date", "timestamp")), None)
    load_col = next((c for c in columns if lower[c] in ("load", "y", "actual")), None)
    temp_cols = [c for c in columns if re.fullmatch(r"w\d+", lower[c])]

    if date_col is None:
        raise ValueError(f"Could not find a date/timestamp column in {columns}")
    if load_col is None:
        raise ValueError(f"Could not find a load column in {columns}")
    if not temp_cols:
        raise ValueError(f"Could not find temperature columns (expected w1..w25) in {columns}")

    return date_col, load_col, temp_cols


def load_all_tasks(raw_load_dir: str | Path, n_tasks: int = 15) -> LoadedData:
    raw_load_dir = Path(raw_load_dir)
    task_dirs = _find_task_dirs(raw_load_dir, n_tasks)

    frames = []
    temp_cols_ref: list[str] | None = None

    for task_num, task_dir in task_dirs:
        csv_path = _find_train_csv(task_dir, task_num)
        raw = pd.read_csv(csv_path)

        date_col, load_col, temp_cols = _identify_columns(list(raw.columns))
        if temp_cols_ref is None:
            temp_cols_ref = temp_cols

        # Some releases give date + separate "hour" column (1-24) instead of
        # a full timestamp. Handle both.
        if "hour" in {c.lower() for c in raw.columns} and not pd.api.types.is_datetime64_any_dtype(raw[date_col]):
            hour_col = next(c for c in raw.columns if c.lower() == "hour")
            hour_offset = pd.to_timedelta(raw[hour_col].astype(int) - 1, unit="h")
            timestamp = pd.to_datetime(raw[date_col]) + hour_offset
        else:
            timestamp = pd.to_datetime(raw[date_col])

        tidy = pd.DataFrame({"timestamp": timestamp, "load": raw[load_col]})
        for c in temp_cols_ref:
            tidy[c] = raw[c] if c in raw.columns else pd.NA

        tidy["task"] = task_num
        frames.append(tidy)

    full = pd.concat(frames, ignore_index=True)

    # Rows for the same timestamp can appear in multiple task files (each
    # task re-releases prior history). Keep the row from the LATEST task
    # release per timestamp for training features, but retain the
    # per-task tagging separately for the backtest harness (see
    # src/evaluation/backtest.py) which needs to know, for a given test
    # task t, exactly which rows were visible in task t-1's release.
    full = full.sort_values(["timestamp", "task"]).reset_index(drop=True)

    return LoadedData(df=full, temp_cols=temp_cols_ref)
