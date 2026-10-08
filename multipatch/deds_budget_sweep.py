"""Budget-sensitivity sweep for Policies A and C (DEDS journal Fig. 7)."""

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
    compute_effective_populations,
    get_baseline_config,
    get_mobility_matrix,
)
from multipatch.multipatch_control_eval import (  # noqa: E402
    apply_control_allocation,
    compute_weighted_policy_allocations,
)
from multipatch.multipatch_ode import simulate_multipatch  # noqa: E402

T_SPAN = (0, 5 * 365)
T_EVAL = np.linspace(0, 5 * 365, 600)


def _initial_state(config: Bihar5PatchModelConfig, P: np.ndarray) -> np.ndarray:
    """Endemic-approximation initial state, identical to evaluate_control_strategies."""
    n = len(config.districts)
    n_eff = compute_effective_populations(config, P)
    y0 = np.zeros(25, dtype=np.float64)
    for k in range(n):
        y0[5 * k + 0] = config.districts[k].N_H * 0.98
        y0[5 * k + 1] = config.districts[k].N_H * 0.02
        y0[5 * k + 3] = config.districts[k].n_F * n_eff[k] * 0.95
        y0[5 * k + 4] = config.districts[k].n_F * n_eff[k] * 0.05
    return y0


def _burden(config: Bihar5PatchModelConfig, P: np.ndarray, y0: np.ndarray) -> float:
    """Cumulative 5-year human infection-days."""
    _, sol = simulate_multipatch(config, P, T_SPAN, y0, T_EVAL)
    total_i = np.sum(sol[1::5, :], axis=0)
    return float(np.trapezoid(total_i, T_EVAL))


def run_budget_sweep(budgets: np.ndarray) -> dict:
    """Cumulative 5-year infection-days for Policies A (uniform) and C (KKT spectral) per budget level."""
    cfg = get_baseline_config()
    P = get_mobility_matrix(cfg, theta=1.0)
    y0 = _initial_state(cfg, P)
    burden_a, burden_c = [], []
    for budget in budgets:
        alloc_A, _, alloc_C = compute_weighted_policy_allocations(
            cfg, P, target_budget=float(budget)
        )
        burden_a.append(_burden(apply_control_allocation(cfg, alloc_A), P, y0))
        burden_c.append(_burden(apply_control_allocation(cfg, alloc_C), P, y0))
    gain = [
        (a - c) / a * 100.0 if a > 0 else float("nan")
        for a, c in zip(burden_a, burden_c)
    ]
    return {
        "budget": [float(b) for b in budgets],
        "burden_A": burden_a,
        "burden_C": burden_c,
        "relative_gain_pct": gain,
    }


def main() -> None:
    """Execute budget sweep, export CSV, and render publication figure."""
    budgets = np.linspace(0.20, 0.80, 13)
    sweep = run_budget_sweep(budgets)

    csv_path = REPO_ROOT / "data" / "processed" / "deds_budget_sweep.csv"
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

    plt.rcParams.update(
        {
            "font.family": "serif",
            "mathtext.fontset": "cm",
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    fig, ax = plt.subplots(figsize=(6.0, 3.8), dpi=300)
    ax.plot(
        sweep["budget"],
        np.array(sweep["burden_A"]) / 1e6,
        color="#1f4e79",
        lw=2.0,
        marker="o",
        ms=4,
        label="Policy A (uniform)",
    )
    ax.plot(
        sweep["budget"],
        np.array(sweep["burden_C"]) / 1e6,
        color="#4f6228",
        lw=2.0,
        marker="s",
        ms=4,
        label="Policy C (spectral KKT)",
    )
    mid = len(sweep["budget"]) // 2
    ax.annotate(
        f"C burden {sweep['relative_gain_pct'][mid]:.1f}% lower than A at $\\bar B=0.5$",
        xy=(0.5, sweep["burden_C"][mid] / 1e6),
        xytext=(0.28, sweep["burden_A"][0] / 1e6 * 0.92),
        fontsize=8,
        arrowprops=dict(arrowstyle="->", lw=0.8),
    )
    ax.set_xlabel(r"state-wide suppression budget $\bar B$")
    ax.set_ylabel("5-year infection-days (millions)")
    ax.legend(fontsize=9)
    fig.tight_layout()
    for out_dir in (
        REPO_ROOT / "outputs" / "figures",
        REPO_ROOT / "latex" / "conference_icmsc2026" / "figures",
    ):
        out_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_dir / "deds_budget_sensitivity.pdf")
        fig.savefig(out_dir / "deds_budget_sensitivity.png")
    plt.close(fig)
    print(f"Success! Written to: {csv_path}")


if __name__ == "__main__":
    main()
