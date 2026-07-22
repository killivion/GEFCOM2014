"""
Brief, standalone exploratory look at the parsed GEFCom2014-L dataset --
independent of the modelling pipeline. Prints basic stats and the
temperature/load correlation, and saves everything -- plots (distribution,
temperature-load relationship, daily/seasonal patterns) and the numeric
stats/correlation matrix CSVs -- to reports/eda/ (always overwriting --
reports/ holds only the latest run).

Usage:
    python -m src.eda.explore_data [--config configs/default.yaml]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import yaml

from src.data.loader import get_data
from src.evaluation.report_utils import save_report

DEFAULT_OUTPUT_DIR = Path("reports/eda")


def run(config_path: str = "configs/default.yaml", output_dir: Path = DEFAULT_OUTPUT_DIR) -> None:
    with open(config_path) as f:
        config = yaml.safe_load(f)

    data = get_data(
        config["data"]["raw_load_dir"],
        n_tasks=config["data"]["n_tasks"],
        processed_path=config["data"]["processed_path"],
    )
    df = data.df
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    n_missing = df["load"].isna().sum()
    print(f"Rows: {len(df):,} | Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    print(f"Missing load: {n_missing:,} ({n_missing / len(df):.1%}) -- Task 1's earliest history, load never released")
    print()

    summary_stats = pd.DataFrame([{
        "n_rows": len(df),
        "date_min": df["timestamp"].min(),
        "date_max": df["timestamp"].max(),
        "n_missing_load": int(n_missing),
        "pct_missing_load": n_missing / len(df),
    }])

    corr_cols = ["load", "temp_mean_actual", "heating_degrees_actual", "cooling_degrees_actual"]
    corr = df[corr_cols].corr()
    corr.index.name = "feature"
    print("Correlation matrix (load vs. temperature-derived features):")
    print(corr.round(3).to_string())
    print()

    plot_df = df.dropna(subset=["load"])

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(plot_df["load"], bins=60)
    ax.set_xlabel("load (MW)")
    ax.set_ylabel("count")
    ax.set_title("Load distribution")
    fig.tight_layout()
    fig.savefig(output_dir / "load_distribution.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    sample = plot_df.sample(min(20_000, len(plot_df)), random_state=0)
    ax.scatter(sample["temp_mean_actual"], sample["load"], s=2, alpha=0.3)
    ax.set_xlabel("temperature (deg F)")
    ax.set_ylabel("load (MW)")
    r = corr.loc["load", "temp_mean_actual"]
    ax.set_title(f"Temperature vs. load (r={r:.2f}, 20k-point sample)")
    fig.tight_layout()
    fig.savefig(output_dir / "temperature_vs_load.png", dpi=150)
    plt.close(fig)

    hourly = plot_df.groupby("hour")["load"].mean()
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(hourly.index, hourly.values, marker="o")
    ax.set_xlabel("hour of day")
    ax.set_ylabel("mean load (MW)")
    ax.set_title("Average daily load profile")
    fig.tight_layout()
    fig.savefig(output_dir / "daily_profile.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    plot_df.boxplot(column="load", by="month", ax=ax)
    ax.set_xlabel("month")
    ax.set_ylabel("load (MW)")
    ax.set_title("Load by month")
    fig.suptitle("")
    fig.tight_layout()
    fig.savefig(output_dir / "seasonal_pattern.png", dpi=150)
    plt.close(fig)

    print(f"Saved 4 plots to {output_dir}/")

    save_report(summary_stats, "eda/summary_stats.csv")
    save_report(corr, "eda/correlation_matrix.csv")
    print(f"Saved summary stats and correlation matrix to {output_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    args = parser.parse_args()
    run(args.config, Path(args.output_dir))
