"""Mobility-coupling sweep R0(theta) with P_theta = (1-theta)I + theta*P (DEDS journal Fig. 6)."""

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
PYTHON_DIR = REPO_ROOT / "python"
if str(PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_DIR))

from multipatch.multipatch_config import (  # noqa: E402
    Bihar5PatchModelConfig,
    get_baseline_config,
    get_mobility_matrix,
)
from multipatch.multipatch_spectral_r0 import (  # noqa: E402
    compute_isolated_r0,
    compute_spectral_r0,
)

DISTRICTS = ["Muzaffarpur", "Saran", "Patna", "Bhagalpur", "Munger"]


def run_mobility_sweep(theta_grid: np.ndarray) -> dict:
    """Compute R0(theta) over P_theta = (1-theta)I + theta*P_baseline."""
    cfg = get_baseline_config()
    isolated = compute_isolated_r0(cfg)
    r0_values = [
        compute_spectral_r0(cfg, get_mobility_matrix(cfg, float(theta)))
        for theta in theta_grid
    ]
    return {
        "theta": [float(t) for t in theta_grid],
        "R0": r0_values,
        "isolated_r0": [float(r) for r in isolated],
    }


def main() -> None:
    theta_grid = np.linspace(0.0, 1.5, 61)
    sweep = run_mobility_sweep(theta_grid)

    # CSV: theta, R0, and per-district isolated R0_k reference columns
    csv_path = REPO_ROOT / "data" / "processed" / "deds_mobility_sweep.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    header = "theta,R0," + ",".join(f"R0_{d}" for d in DISTRICTS)
    lines = [header]
    for theta, r0 in zip(sweep["theta"], sweep["R0"]):
        iso = ",".join(f"{r:.10f}" for r in sweep["isolated_r0"])
        lines.append(f"{theta:.4f},{r0:.10f},{iso}")
    csv_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    # Figure: serif, no in-artwork title, threshold line at R0 = 1
    plt.rcParams.update({
        "font.family": "serif",
        "mathtext.fontset": "cm",
        "axes.spines.top": False,
        "axes.spines.right": False,
    })
    fig, ax = plt.subplots(figsize=(6.0, 3.8), dpi=300)
    ax.plot(sweep["theta"], sweep["R0"], color="#1f4e79", lw=2.0, label=r"$\mathcal{R}_0(\theta)$")
    ax.axhline(1.0, color="gray", ls="--", lw=1.0)
    ax.text(1.505, 1.0, r"$\mathcal{R}_0=1$", va="center", fontsize=9, color="gray")
    iso = sweep["isolated_r0"]
    ax.plot([0.0], [max(iso)], marker="*", ms=12, color="#c0504d", ls="none",
            label=r"decoupled $\max_k \mathcal{R}_0^k=%.3f$" % max(iso))
    ax.plot([1.0], [sweep["R0"][int(np.argmin(np.abs(np.array(sweep["theta"]) - 1.0)))]],
            marker="o", ms=7, color="#4f6228", ls="none",
            label=r"calibrated $\theta=1$")
    ax.set_xlabel(r"mobility coupling $\theta$")
    ax.set_ylabel(r"network reproduction number $\mathcal{R}_0$")
    ax.set_xlim(-0.02, 1.55)
    ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    for out_dir in (REPO_ROOT / "outputs" / "figures",
                    REPO_ROOT / "latex" / "conference_icmsc2026" / "figures"):
        out_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_dir / "deds_mobility_sweep.pdf")
        fig.savefig(out_dir / "deds_mobility_sweep.png")
    plt.close(fig)
    print(f"Success! Written to: {csv_path}")


if __name__ == "__main__":
    main()
