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

Usage (builds/refreshes the processed cache all the other scripts read
from, and prints a summary so you can sanity-check the parse):
    python -m src.data.loader [--config configs/default.yaml] [--force-rebuild]
"""
from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import yaml

from src.features.build_features import add_calendar_features, add_temperature_variants


@dataclass
class LoadedData:
    df: pd.DataFrame          # tidy frame: timestamp, task, load, temp_* columns
    temp_cols: list[str]      # names of the temperature station columns


DEFAULT_PROCESSED_PATH = Path("src/data/processed/full_load.csv")


def save_processed(data: LoadedData, path: str | Path = DEFAULT_PROCESSED_PATH) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data.df.to_csv(path, index=False)


def load_processed(path: str | Path = DEFAULT_PROCESSED_PATH) -> LoadedData:
    df = pd.read_csv(Path(path), parse_dates=["timestamp"])
    temp_cols = [c for c in df.columns if re.fullmatch(r"w\d+", c.lower())]
    return LoadedData(df=df, temp_cols=temp_cols)


def get_data(
    raw_load_dir: str | Path,
    n_tasks: int = 15,
    processed_path: str | Path = DEFAULT_PROCESSED_PATH,
    force_rebuild: bool = False,
) -> LoadedData:
    """Loads the cached processed CSV if present, otherwise parses the raw
    Task 1..n_tasks CSVs (see load_all_tasks), adds calendar and
    temperature-variant features (see build_features.py -- both are safe
    to compute once here since they only ever look up a fixed point
    relative to a row's own timestamp), and writes the cache so
    subsequent runs and error-checking don't need to redo any of that
    each time."""
    processed_path = Path(processed_path)
    if processed_path.exists() and not force_rebuild:
        return load_processed(processed_path)

    data = load_all_tasks(raw_load_dir, n_tasks=n_tasks)
    data.df = add_calendar_features(data.df)
    data.df = add_temperature_variants(data.df, data.temp_cols)
    save_processed(data, processed_path)
    return data


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


_CONCATENATED_TS_RE = re.compile(r"^(?P<mdy>\d+)\s+(?P<hour>\d{1,2}):(?P<minute>\d{2})$")


def _split_month_day_candidates(mdy_digits: str) -> list[tuple[int, int]]:
    """Some GEFCom2014 releases write TIMESTAMP as month+day+4-digit-year
    concatenated with no separators and no leading zeros (e.g. "1012010"
    for Oct 1, 2010). The year is unambiguous (last 4 digits), but a
    3-digit month+day remainder is ambiguous: "127" could be month=1/day=27
    or month=12/day=7. Both candidates are returned; the caller picks the
    one consistent with the surrounding sequence of hourly readings.
    """
    n = len(mdy_digits)
    if n == 2:
        return [(int(mdy_digits[0]), int(mdy_digits[1]))]
    if n == 4:
        return [(int(mdy_digits[:2]), int(mdy_digits[2:]))]
    if n == 3:
        return [
            (int(mdy_digits[0]), int(mdy_digits[1:])),
            (int(mdy_digits[:2]), int(mdy_digits[2:3])),
        ]
    raise ValueError(f"Unexpected month/day digit count in TIMESTAMP: {mdy_digits!r}")


def _parse_concatenated_timestamps(
    raw_dates: pd.Series, prev: pd.Timestamp | None
) -> tuple[pd.Series, pd.Timestamp | None]:
    parsed = []
    for raw in raw_dates:
        m = _CONCATENATED_TS_RE.match(str(raw).strip())
        if m is None:
            raise ValueError(f"Could not parse TIMESTAMP value: {raw!r}")
        mdy, hour, minute = m.group("mdy"), int(m.group("hour")), int(m.group("minute"))
        year = int(mdy[-4:])

        candidates = []
        for month, day in _split_month_day_candidates(mdy[:-4]):
            try:
                candidates.append(pd.Timestamp(year=year, month=month, day=day, hour=hour, minute=minute))
            except ValueError:
                continue
        if not candidates:
            raise ValueError(f"No valid calendar date for TIMESTAMP value: {raw!r}")

        if len(candidates) == 1:
            ts = candidates[0]
        elif prev is None:
            ts = min(candidates)
        else:
            # Readings are hourly and strictly increasing, so the correct
            # split is always the candidate closest to the previous
            # timestamp; the wrong split lands months away.
            ts = min(candidates, key=lambda c: abs(c - prev))

        parsed.append(ts)
        prev = ts

    return pd.Series(parsed, index=raw_dates.index), prev


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
    # Carries the last parsed timestamp across task files so ambiguous
    # concatenated dates (see _parse_concatenated_timestamps) at the start
    # of one task's file resolve using the previous task's ending time.
    prev_ts: pd.Timestamp | None = None

    for task_num, task_dir in task_dirs:
        csv_path = _find_train_csv(task_dir, task_num)
        raw = pd.read_csv(csv_path)

        date_col, load_col, temp_cols = _identify_columns(list(raw.columns))
        if temp_cols_ref is None:
            temp_cols_ref = temp_cols

        # Some releases give date + separate "hour" column (1-24) instead of
        # a full timestamp. Others concatenate month+day+year with no
        # separators (e.g. "1012010 1:00"). Handle all three.
        if "hour" in {c.lower() for c in raw.columns} and not pd.api.types.is_datetime64_any_dtype(raw[date_col]):
            hour_col = next(c for c in raw.columns if c.lower() == "hour")
            hour_offset = pd.to_timedelta(raw[hour_col].astype(int) - 1, unit="h")
            timestamp = pd.to_datetime(raw[date_col]) + hour_offset
        elif raw[date_col].astype(str).str.contains("[/-]", regex=True).any():
            timestamp = pd.to_datetime(raw[date_col])
        else:
            timestamp, prev_ts = _parse_concatenated_timestamps(raw[date_col], prev_ts)

        tidy = pd.DataFrame({"timestamp": timestamp, "load": raw[load_col]})
        for c in temp_cols_ref:
            tidy[c] = raw[c] if c in raw.columns else pd.NA

        tidy["task"] = task_num
        frames.append(tidy)

    full = pd.concat(frames, ignore_index=True)

    # Each task's train file is a pure continuation of the previous one
    # (verified: zero duplicate timestamps across all 15 tasks), so no
    # deduplication is needed here -- every row keeps the `task` number of
    # the file it actually came from, which is exactly what the backtest
    # harness (src/evaluation/backtest.py) needs to know, for a given test
    # task t, which rows were visible in task t-1's release.
    full = full.sort_values(["timestamp", "task"]).reset_index(drop=True)

    return LoadedData(df=full, temp_cols=temp_cols_ref)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--force-rebuild", action="store_true", help="Re-parse the raw CSVs even if a cache already exists.")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    result = get_data(
        config["data"]["raw_load_dir"],
        n_tasks=config["data"]["n_tasks"],
        processed_path=config["data"]["processed_path"],
        force_rebuild=args.force_rebuild,
    )
    print(result.df.head())
    print(result.df.dtypes)
    print("temp columns found:", result.temp_cols)
    print("rows per task:", result.df["task"].value_counts().sort_index())
