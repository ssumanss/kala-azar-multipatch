#!/usr/bin/env python3
"""Run mobility sweep R0(theta) and write output to data/deds_mobility_sweep.csv."""

from pathlib import Path
import sys
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from multipatch.multipatch_config import get_baseline_config, get_mobility_matrix
from multipatch.multipatch_spectral_r0 import compute_isolated_r0, compute_spectral_r0


def main() -> None:
    theta_grid = np.linspace(0.0, 1.5, 61)
    cfg = get_baseline_config()
    isolated = compute_isolated_r0(cfg)
    r0_vals = []

    for theta in theta_grid:
        P_theta = get_mobility_matrix(cfg, float(theta))
        r0 = compute_spectral_r0(cfg, P=P_theta)
        r0_vals.append(r0)

    csv_path = REPO_ROOT / "data" / "deds_mobility_sweep.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["theta,R0,R0_Muzaffarpur,R0_Munger"]
    for t, r in zip(theta_grid, r0_vals):
        lines.append(f"{t:.4f},{r:.6f},{isolated[0]:.6f},{isolated[4]:.6f}")

    csv_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Success! Written to: {csv_path}")


if __name__ == "__main__":
    main()
