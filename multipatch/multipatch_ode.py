"""Numerical integration of 25-state coupled multi-patch Kala-azar transmission model."""

from pathlib import Path
import sys
from typing import Optional, Tuple
import numpy as np
from scipy.integrate import solve_ivp

# Ensure python directory is on sys.path for direct script execution
REPO_ROOT = Path(__file__).resolve().parents[2]
PYTHON_DIR = REPO_ROOT / "python"
if str(PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_DIR))

from multipatch.multipatch_config import (  # noqa: E402
    Bihar5PatchModelConfig,
    compute_effective_populations,
)


def rhs_multipatch(t: float, y: np.ndarray, config: Bihar5PatchModelConfig, P: np.ndarray) -> np.ndarray:
    """Compute orbital derivatives dy/dt for the 5-patch host-vector ODE network.

    The physical vector field is naturally positively invariant on R_+^25:
    inflow rates are strictly non-negative whenever any state variable approaches zero.
    """
    n = len(config.districts)
    dydt = np.zeros_like(y)

    S_H = y[0::5]
    I_H = y[1::5]
    R_H = y[2::5]
    S_V = y[3::5]
    I_V = y[4::5]

    N_eff = compute_effective_populations(config, P)
    N_eff_safe = np.maximum(N_eff, 1e-12)
    I_eff = I_H @ P  # Effective infectious humans present in patch j

    # Forces of infection
    lambda_H = config.a * config.b_H * (I_V / N_eff_safe)
    lambda_V = config.a * config.b_V * (I_eff / N_eff_safe)

    for k in range(n):
        d = config.districts[k]
        force_H_k = np.dot(P[k, :], lambda_H)

        dydt[5 * k + 0] = config.mu_H * d.N_H - force_H_k * S_H[k] - config.d_H * S_H[k]
        dydt[5 * k + 1] = force_H_k * S_H[k] - (config.d_H + config.delta_H + d.r_H) * I_H[k]
        dydt[5 * k + 2] = d.r_H * I_H[k] - config.d_H * R_H[k]

        N_V_eq = d.n_F * N_eff[k]
        Lambda_V = config.d_V * N_V_eq

        dydt[5 * k + 3] = Lambda_V - lambda_V[k] * S_V[k] - config.d_V * S_V[k]
        dydt[5 * k + 4] = lambda_V[k] * S_V[k] - config.d_V * I_V[k]

    return dydt


def simulate_multipatch(
    config: Bihar5PatchModelConfig,
    P: np.ndarray,
    t_span: Tuple[float, float],
    y0: np.ndarray,
    t_eval: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Integrate the system using solve_ivp with RK45."""
    sol = solve_ivp(
        fun=lambda t, y: rhs_multipatch(t, y, config, P),
        t_span=t_span,
        y0=y0,
        t_eval=t_eval,
        method="RK45",
        rtol=1e-5,
        atol=1e-7,
    )
    return sol.t, sol.y
