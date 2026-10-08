"""Policy evaluation comparing Uniform vs Mobility-Targeted Spectral vector control."""

from pathlib import Path
import sys
from typing import Any, Dict, Tuple
import numpy as np

# Ensure python directory is on sys.path for direct script execution
REPO_ROOT = Path(__file__).resolve().parents[2]
PYTHON_DIR = REPO_ROOT / "python"
if str(PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_DIR))

from multipatch.multipatch_config import (  # noqa: E402
    Bihar5PatchModelConfig,
    DistrictConfig,
    compute_effective_populations,
    KKT_BUDGET_TOL,
)
from multipatch.multipatch_ode import simulate_multipatch  # noqa: E402
from multipatch.multipatch_spectral_r0 import (  # noqa: E402
    compute_perron_left_eigenvector,
    compute_spectral_r0,
)


def project_bounded_simplex_allocation(
    target_weights: np.ndarray,
    capacities: np.ndarray,
    budget: float,
    eta_min: float = 0.05,
    eta_max: float = 0.85,
) -> np.ndarray:
    """Exact L2 bounded simplex projection via KKT water-filling bisection.
    
    Solves: min 0.5 * sum (eta_k - w_k)^2 
    subject to: sum eta_k * capacities_k = budget, eta_min <= eta_k <= eta_max.
    Lagrangian: L = 0.5*||eta-w||^2 - lambda*(sum c_k*eta_k - B) + sum alpha_k*(eta_min - eta_k) + sum beta_k*(eta_k - eta_max).
    Stationarity: eta_k(lambda) = clip(w_k + lambda * capacities_k, eta_min, eta_max).
    """
    assert np.all(capacities > 0), "All district capacities must be strictly positive"
    assert eta_min < eta_max, f"eta_min ({eta_min}) must be strictly less than eta_max ({eta_max})"
    weight_capacity_sum = np.sum(target_weights * capacities)
    assert weight_capacity_sum > 0, "Weighted capacity sum must be strictly positive"

    total_cap = np.sum(capacities)
    min_possible_budget = eta_min * total_cap
    max_possible_budget = eta_max * total_cap
    if budget < min_possible_budget - KKT_BUDGET_TOL or budget > max_possible_budget + KKT_BUDGET_TOL:
        raise ValueError(
            f"Infeasible budget: {budget} not in [{min_possible_budget}, {max_possible_budget}]"
        )

    w = target_weights * (budget / weight_capacity_sum)

    # Guaranteed bracketing:
    lambda_min = float(np.min((eta_min - w) / capacities))
    lambda_max = float(np.max((eta_max - w) / capacities))

    def budget_at(lam: float) -> float:
        eta = np.clip(w + lam * capacities, eta_min, eta_max)
        return float(np.dot(eta, capacities))

    # Defensive runtime verification of the bracket
    assert budget_at(lambda_min) <= budget + KKT_BUDGET_TOL, f"Invalid lower bracket: {budget_at(lambda_min)} > {budget}"
    assert budget_at(lambda_max) >= budget - KKT_BUDGET_TOL, f"Invalid upper bracket: {budget_at(lambda_max)} < {budget}"

    low = lambda_min
    high = lambda_max
    for _ in range(120):
        mid = 0.5 * (low + high)
        eta_mid = np.clip(w + mid * capacities, eta_min, eta_max)
        diff = np.sum(eta_mid * capacities) - budget
        if abs(diff) < KKT_BUDGET_TOL:
            return eta_mid
        if diff < 0:
            low = mid
        else:
            high = mid
    return np.clip(w + mid * capacities, eta_min, eta_max)


def compute_weighted_policy_allocations(
    config: Bihar5PatchModelConfig, P: np.ndarray, target_budget: float = 0.50
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute vector reduction fractions eta_k conserving total sandfly population-weighted budget."""
    n = len(config.districts)
    N_eff = compute_effective_populations(config, P)
    N_V_eq = np.array([d.n_F * N_eff[k] for k, d in enumerate(config.districts)])
    total_sandflies = np.sum(N_V_eq)

    # Policy A: Uniform 50% reduction in all districts
    alloc_A = np.full(n, target_budget, dtype=np.float64)

    # Policy B: Incidence proportional (approximate baseline endemicity ranking)
    raw_B = np.array([d.n_F for d in config.districts])
    alloc_B = raw_B * (target_budget * total_sandflies / np.sum(raw_B * N_V_eq))
    alloc_B = np.clip(alloc_B, 0.05, 0.85)
    alloc_B = alloc_B * (target_budget * total_sandflies / np.sum(alloc_B * N_V_eq))

    # Policy C: Spectral Perron left-eigenvector allocation
    # Policy C is a spectral Perron left-eigenvector allocation heuristic targeting high transmission reservoir nodes; 
    # it is an operational heuristic rather than a mathematically proven optimal control.
    # u[:n] corresponds to human infected weighting in left Perron vector of V^-1 * F
    u = compute_perron_left_eigenvector(config, P)
    u_H = u[:n]
    alloc_C = project_bounded_simplex_allocation(
        u_H, N_V_eq, target_budget * total_sandflies, eta_min=0.05, eta_max=0.85
    )

    return alloc_A, alloc_B, alloc_C


def apply_control_allocation(config: Bihar5PatchModelConfig, alloc: np.ndarray) -> Bihar5PatchModelConfig:
    """Create a new model configuration with reduced vector densities n_F_k = (1 - eta_k) * n_F_k."""
    new_districts = [
        DistrictConfig(name=d.name, N_H=d.N_H, n_F=d.n_F * (1.0 - alloc[i]), r_H=d.r_H)
        for i, d in enumerate(config.districts)
    ]
    return Bihar5PatchModelConfig(
        districts=new_districts,
        a=config.a,
        b_H=config.b_H,
        b_V=config.b_V,
        d_V=config.d_V,
        d_H=config.d_H,
        mu_H=config.mu_H,
        delta_H=config.delta_H,
        baseline_P=config.baseline_P,
    )


def evaluate_control_strategies(
    config: Bihar5PatchModelConfig, P: np.ndarray
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Run 5-year simulation of baseline vs Policies A, B, and C, and compute cumulative infection days."""
    n = 5
    N_eff = compute_effective_populations(config, P)
    alloc_A, alloc_B, alloc_C = compute_weighted_policy_allocations(config, P, target_budget=0.50)

    cfg_A = apply_control_allocation(config, alloc_A)
    cfg_B = apply_control_allocation(config, alloc_B)
    cfg_C = apply_control_allocation(config, alloc_C)

    r0_base = compute_spectral_r0(config, P)
    r0_A = compute_spectral_r0(cfg_A, P)
    r0_B = compute_spectral_r0(cfg_B, P)
    r0_C = compute_spectral_r0(cfg_C, P)

    # Initial condition: endemic steady-state approximation
    y0 = np.zeros(25, dtype=np.float64)
    for k in range(n):
        y0[5 * k + 0] = config.districts[k].N_H * 0.98
        y0[5 * k + 1] = config.districts[k].N_H * 0.02
        y0[5 * k + 2] = 0.0
        y0[5 * k + 3] = config.districts[k].n_F * N_eff[k] * 0.95
        y0[5 * k + 4] = config.districts[k].n_F * N_eff[k] * 0.05

    t_span = (0, 5 * 365)
    t_eval = np.linspace(0, 5 * 365, 600)

    _, sol_base = simulate_multipatch(config, P, t_span, y0, t_eval)
    _, sol_A = simulate_multipatch(cfg_A, P, t_span, y0, t_eval)
    _, sol_B = simulate_multipatch(cfg_B, P, t_span, y0, t_eval)
    _, sol_C = simulate_multipatch(cfg_C, P, t_span, y0, t_eval)

    # Cumulative human infection days: sum_k integral I_H_k dt
    total_I_base = np.sum(sol_base[1::5, :], axis=0)
    total_I_A = np.sum(sol_A[1::5, :], axis=0)
    total_I_B = np.sum(sol_B[1::5, :], axis=0)
    total_I_C = np.sum(sol_C[1::5, :], axis=0)

    burden_base = float(np.trapezoid(total_I_base, t_eval))
    burden_A = float(np.trapezoid(total_I_A, t_eval))
    burden_B = float(np.trapezoid(total_I_B, t_eval))
    burden_C = float(np.trapezoid(total_I_C, t_eval))

    reduction_A = (burden_base - burden_A) / burden_base * 100.0
    reduction_C = (burden_base - burden_C) / burden_base * 100.0
    gain_C_over_A = (burden_A - burden_C) / burden_A * 100.0

    metrics: Dict[str, Any] = {
        "r0": {
            "baseline": float(r0_base),
            "policy_A": float(r0_A),
            "policy_B": float(r0_B),
            "policy_C": float(r0_C),
        },
        "burden_infection_days": {
            "baseline": burden_base,
            "policy_A": burden_A,
            "policy_B": burden_B,
            "policy_C": burden_C,
        },
        "reduction_pct": {"policy_A": reduction_A, "policy_C": reduction_C},
        "spectral_gain_over_uniform_pct": gain_C_over_A,
        "allocations": {
            "policy_A": [float(v) for v in alloc_A],
            "policy_B": [float(v) for v in alloc_B],
            "policy_C": [float(v) for v in alloc_C],
        },
    }

    trajectories: Dict[str, Any] = {
        "t_years": t_eval / 365.0,
        "total_base": total_I_base,
        "total_A": total_I_A,
        "total_B": total_I_B,
        "total_C": total_I_C,
    }

    return metrics, trajectories
