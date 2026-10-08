"""Configuration and mobility matrix models for 5-patch Kala-azar transmission in Bihar."""

from dataclasses import dataclass
from typing import List
import numpy as np

MATRIX_IDENTITY_TOL: float = 1e-12
SPECTRAL_RADIUS_TOL: float = 1e-9
KKT_BUDGET_TOL: float = 1e-10


@dataclass(frozen=True)
class DistrictConfig:
    """Demographic and epidemiological parameters for an individual district patch."""

    name: str
    N_H: float
    n_F: float  # Baseline vector-to-effective-human ratio
    r_H: float  # District-specific recovery/treatment rate (day^-1)


@dataclass(frozen=True)
class Bihar5PatchModelConfig:
    """Full 5-patch configuration for Bihar transmission model."""

    districts: List[DistrictConfig]
    a: float  # Sandfly biting rate (day^-1)
    b_H: float  # Vector-to-human transmission probability
    b_V: float  # Human-to-vector transmission probability
    d_V: float  # Adult sandfly mortality rate (day^-1)
    d_H: float  # Natural human mortality rate (day^-1)
    mu_H: float  # Human birth rate (day^-1)
    delta_H: float  # Kala-azar induced human mortality rate (day^-1)
    baseline_P: np.ndarray  # 5x5 baseline gravity residence matrix


def get_baseline_config() -> Bihar5PatchModelConfig:
    """Create calibrated baseline configuration for Muzaffarpur, Saran, Patna, Bhagalpur, and Munger."""
    districts = [
        DistrictConfig(name="Muzaffarpur", N_H=4_800_000.0, n_F=4.5, r_H=1.0 / 180.0),
        DistrictConfig(name="Saran", N_H=3_900_000.0, n_F=3.2, r_H=1.0 / 180.0),
        DistrictConfig(name="Patna", N_H=5_800_000.0, n_F=1.2, r_H=1.0 / 90.0),  # Better tertiary care
        DistrictConfig(name="Bhagalpur", N_H=3_000_000.0, n_F=2.0, r_H=1.0 / 150.0),
        DistrictConfig(name="Munger", N_H=1_400_000.0, n_F=0.8, r_H=1.0 / 120.0),
    ]

    # Gravity-calibrated residence matrix (rows: resident patch, cols: visited patch)
    # Off-diagonal flows reflect central commuter convergence onto Patna and adjacent corridors
    P_base = np.array(
        [
            [0.86, 0.03, 0.08, 0.01, 0.02],  # Muzaffarpur
            [0.03, 0.88, 0.07, 0.01, 0.01],  # Saran
            [0.04, 0.03, 0.87, 0.03, 0.03],  # Patna
            [0.01, 0.01, 0.06, 0.87, 0.05],  # Bhagalpur
            [0.02, 0.01, 0.07, 0.06, 0.84],  # Munger
        ],
        dtype=np.float64,
    )

    # Normalize rows to guarantee exact row-stochasticity
    P_base = P_base / P_base.sum(axis=1, keepdims=True)

    return Bihar5PatchModelConfig(
        districts=districts,
        a=0.25,
        b_H=0.09,
        b_V=0.10,
        d_V=1.0 / 14.0,
        d_H=1.0 / (65.0 * 365.0),
        mu_H=1.0 / (65.0 * 365.0),
        delta_H=0.1 / 365.0,
        baseline_P=P_base,
    )


def get_mobility_matrix(config: Bihar5PatchModelConfig, theta: float = 1.0) -> np.ndarray:
    """Return parameterized mobility matrix P(theta), scaling off-diagonals by theta.

    Guards against negative diagonal probabilities by enforcing theta in [0, theta_max].
    """
    P_base = config.baseline_P.copy()
    n = P_base.shape[0]

    # Calculate maximum permissible theta before any diagonal becomes non-positive
    off_diag_sums = np.array([np.sum([P_base[k, j] for j in range(n) if j != k]) for k in range(n)])
    theta_max = float(1.0 / np.max(off_diag_sums))

    if theta < 0.0 or theta > theta_max:
        raise ValueError(f"theta must be in [0, {theta_max:.2f}], got {theta}")

    P = np.zeros_like(P_base)
    for k in range(n):
        for j in range(n):
            if k != j:
                P[k, j] = theta * P_base[k, j]
        P[k, k] = 1.0 - np.sum([P[k, j] for j in range(n) if j != k])

    return P


def compute_effective_populations(config: Bihar5PatchModelConfig, P: np.ndarray) -> np.ndarray:
    """Compute effective human populations present in each patch: N_tilde_j = sum_k p_kj * N_H_k."""
    N_H = np.array([d.N_H for d in config.districts])
    return N_H @ P
