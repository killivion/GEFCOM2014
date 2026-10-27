"""
Regenerates the variant configs from configs/default.yaml, so they can
only differ from the final model where their header says:

    python configs/make_variants.py

- dev.yaml: 20-quantile grid for fast iteration (not for reporting).
- oracle_temperature.yaml: the forecast month's ACTUAL temperature instead
  of scenarios -- an upper bound, never the reported result.
- ablation_plain.yaml: climatology temperature input, no scenarios, no recency.
- ablation_recency_only.yaml: climatology temperature input + recency.
- ablation_scenarios_only.yaml: scenarios, no recency.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SCENARIOS_ON = "  temperature_scenarios:\n    enabled: true\n"
SCENARIOS_OFF = "  temperature_scenarios:\n    enabled: false\n"
RECENCY_ON = "  recency_half_life_years: 2\n"
RECENCY_OFF = "  recency_half_life_years: null\n"
DEV_LEVELS = """  levels: [0.01, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45,
           0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.95, 0.99]
"""

VARIANTS = {
    "dev.yaml": ("""# DEVELOPMENT config: default.yaml with a 20-quantile grid (roughly 5x
# faster). For iterating only -- the grid is uneven (no 0.90), so its
# numbers are not comparable with the 99-quantile results.
# Run with: python -m src.evaluation.run_comparison --config configs/dev.yaml --folds development --output-dir <scratch>
""", [("levels", None)]),
    "oracle_temperature.yaml": ("""# ORACLE upper bound: default.yaml, but the model is given the ACTUAL
# observed temperature of the forecast month (not known a month ahead)
# instead of temperature scenarios. Measures what a perfect temperature
# forecast would be worth; never the reported result.
# Run with: python -m src.evaluation.run_comparison --config configs/oracle_temperature.yaml --output-dir run_comparison_oracle
""", [("  temperature_source: climatology\n", "  temperature_source: actual\n"), (SCENARIOS_ON, SCENARIOS_OFF)]),
    "ablation_plain.yaml": ("""# ABLATION: default.yaml without both improvements -- climatology
# temperature input, no temperature scenarios, no recency weighting.
# Run with: python -m src.evaluation.run_comparison --config configs/ablation_plain.yaml --output-dir ablation_plain
""", [(SCENARIOS_ON, SCENARIOS_OFF), (RECENCY_ON, RECENCY_OFF)]),
    "ablation_recency_only.yaml": ("""# ABLATION: default.yaml without temperature scenarios -- climatology
# temperature input, recency weighting on.
# Run with: python -m src.evaluation.run_comparison --config configs/ablation_recency_only.yaml --output-dir ablation_recency_only
""", [(SCENARIOS_ON, SCENARIOS_OFF)]),
    "ablation_scenarios_only.yaml": ("""# ABLATION: default.yaml without recency weighting.
# Run with: python -m src.evaluation.run_comparison --config configs/ablation_scenarios_only.yaml --output-dir ablation_scenarios_only
""", [(RECENCY_ON, RECENCY_OFF)]),
}


def main():
    default = (ROOT / "default.yaml").read_text(encoding="utf-8")
    body = default[default.index("data:"):]
    for name, (header, replacements) in VARIANTS.items():
        text = body
        for old, new in replacements:
            if old == "levels":
                start = text.index("  levels: [")
                end = text.index("]\n", start) + 2
                text = text[:start] + DEV_LEVELS + text[end:]
                continue
            assert text.count(old) == 1, (name, old)
            text = text.replace(old, new)
        (ROOT / name).write_text(header + "\n" + text, encoding="utf-8")
        print(f"wrote configs/{name}")


if __name__ == "__main__":
    main()
