#!/usr/bin/env python3
"""Run budget sensitivity sweep and write output to data/deds_budget_sweep.csv."""

from pathlib import Path
import sys
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from multipatch.deds_budget_sweep import run_budget_sweep


def main() -> None:
    budgets = np.linspace(0.20, 0.80, 13)
    sweep = run_budget_sweep(budgets)

    csv_path = REPO_ROOT / "data" / "deds_budget_sweep.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["budget,burden_A,burden_C,relative_gain_pct"]
    for b, a, c, g in zip(
        sweep["budget"],
        sweep["burden_A"],
        sweep["burden_C"],
        sweep["relative_gain_pct"],
    ):
        lines.append(f"{b:.2f},{a:.6f},{c:.6f},{g:.6f}")

    csv_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Success! Written to: {csv_path}")


if __name__ == "__main__":
    main()
