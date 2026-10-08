"""Spectral analysis of next-generation operator for 5-patch Kala-azar network."""

import json
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
    compute_effective_populations,
    get_baseline_config,
    get_mobility_matrix,
)


def compute_isolated_r0(config: Bihar5PatchModelConfig) -> np.ndarray:
    """Compute decoupled isolated reproduction numbers: R0_k = sqrt(a^2 * b_H * b_V * n_F_k / (k_H_k * d_V))."""
    r0_list = []
    for d in config.districts:
        k_H = config.d_H + config.delta_H + d.r_H
        r0_sq = (config.a**2 * config.b_H * config.b_V * d.n_F) / (k_H * config.d_V)
        r0_list.append(np.sqrt(r0_sq))
    return np.array(r0_list, dtype=np.float64)


def compute_block_matrices(
    config: Bihar5PatchModelConfig, P: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Construct transmission blocks F_HV, F_VH and transition blocks V_H, V_V."""
    n = len(config.districts)
    N_H = np.array([d.N_H for d in config.districts])
    N_eff = compute_effective_populations(config, P)

    # F_HV[k, j] = p_kj * a * b_H * (N_H_k / N_eff_j)
    F_HV = np.zeros((n, n), dtype=np.float64)
    for k in range(n):
        for j in range(n):
            F_HV[k, j] = P[k, j] * config.a * config.b_H * (N_H[k] / N_eff[j])

    # F_VH[k, j] = a * b_V * n_F_k * p_jk
    F_VH = np.zeros((n, n), dtype=np.float64)
    for k in range(n):
        for j in range(n):
            F_VH[k, j] = config.a * config.b_V * config.districts[k].n_F * P[j, k]

    # V_H = diag(d_H + delta_H + r_H_k)
    k_H = np.array([config.d_H + config.delta_H + d.r_H for d in config.districts])
    V_H = np.diag(k_H)

    # V_V = diag(d_V)
    V_V = np.diag(np.full(n, config.d_V, dtype=np.float64))

    return F_HV, F_VH, V_H, V_V


def compute_spectral_r0(config: Bihar5PatchModelConfig, P: np.ndarray) -> float:
    """Compute the multi-patch basic reproduction number R0 = rho(F * V^-1).

    Note on matrix operators:
    While the Next-Generation Matrix is defined as K = F @ V_inv with spectral radius R0 = rho(F @ V_inv),
    the Lyapunov functional V(x) = u^T @ V_inv @ x requires the left eigenvector of M = V_inv @ F:
    u^T @ (V_inv @ F) = R0 * u^T.
    Since rho(F @ V_inv) == rho(V_inv @ F), both operators have the exact same spectral radius R0.
    """
    F_HV, F_VH, V_H, V_V = compute_block_matrices(config, P)
    n = len(config.districts)

    F = np.block([[np.zeros((n, n)), F_HV], [F_VH, np.zeros((n, n))]])
    V = np.block([[V_H, np.zeros((n, n))], [np.zeros((n, n)), V_V]])

    V_inv = np.linalg.inv(V)
    K = F @ V_inv

    eigenvalues = np.linalg.eigvals(K)
    return float(np.max(np.real(eigenvalues)))


def compute_perron_left_eigenvector(config: Bihar5PatchModelConfig, P: np.ndarray) -> np.ndarray:
    """Compute strictly positive left Perron eigenvector u of V^-1 * F: u^T * (V^-1 * F) = R0 * u^T."""
    F_HV, F_VH, V_H, V_V = compute_block_matrices(config, P)
    n = len(config.districts)

    F = np.block([[np.zeros((n, n)), F_HV], [F_VH, np.zeros((n, n))]])
    V = np.block([[V_H, np.zeros((n, n))], [np.zeros((n, n)), V_V]])

    V_inv = np.linalg.inv(V)
    M = V_inv @ F  # u^T * M = R0 * u^T <=> M^T * u = R0 * u

    eigenvalues, eigenvectors = np.linalg.eig(M.T)
    leading_idx = np.argmax(np.real(eigenvalues))
    u = np.real(eigenvectors[:, leading_idx])

    # Ensure positive orientation and normalize
    if np.sum(u) < 0:
        u = -u
    u = u / np.sum(u)
    return u


def main() -> None:
    """Run standalone spectral decomposition and save summary metrics."""
    cfg = get_baseline_config()
    P = get_mobility_matrix(cfg, theta=1.0)

    r0_network = compute_spectral_r0(cfg, P)
    r0_isolated = compute_isolated_r0(cfg)
    u_perron = compute_perron_left_eigenvector(cfg, P)

    summary: Dict[str, Any] = {
        "network_r0": float(r0_network),
        "isolated_r0": {d.name: float(r0_isolated[i]) for i, d in enumerate(cfg.districts)},
        "perron_eigenvector": [float(val) for val in u_perron],
    }

    out_path = REPO_ROOT / "outputs" / "multipatch_spectral_summary.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Success! Written to: {out_path}")


if __name__ == "__main__":
    main()
