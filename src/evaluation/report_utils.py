"""
Tiny helper for writing a run's results to reports/. Each path is always
overwritten on the next run -- reports/ holds only the latest run's
output for a given script, not a history of past runs.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

REPORTS_DIR = Path("reports")


def save_report(data: pd.DataFrame | pd.Series, relative_path: str) -> Path:
    """Writes `data` as CSV to reports/<relative_path>, overwriting
    whatever was there before. A named (e.g. groupby) index is turned
    into an explicit column first, so the CSV is self-describing without
    relying on a written-out index column."""
    path = REPORTS_DIR / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(data, pd.Series):
        data = data.rename_axis(data.index.name or "index").reset_index()
    elif data.index.name is not None or isinstance(data.index, pd.MultiIndex):
        data = data.reset_index()
    data.to_csv(path, index=False)
    return path
