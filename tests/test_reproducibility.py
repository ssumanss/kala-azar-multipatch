"""Reproducibility test suite for kala-azar-multipatch."""

from pathlib import Path
import sys
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from multipatch.multipatch_config import get_baseline_config, get_mobility_matrix
from multipatch.multipatch_spectral_r0 import compute_isolated_r0, compute_spectral_r0
from multipatch.deds_budget_sweep import run_budget_sweep


def test_baseline_config_and_mobility_properties() -> None:
    cfg = get_baseline_config()
    assert len(cfg.districts) == 5
    assert cfg.districts[0].name == "Muzaffarpur"
    assert cfg.districts[4].name == "Munger"

    # Verify baseline P is 5x5, non-negative, and strictly row-stochastic
    assert cfg.baseline_P.shape == (5, 5)
    assert np.all(cfg.baseline_P >= 0.0)
    assert np.allclose(cfg.baseline_P.sum(axis=1), 1.0)


def test_spectral_r0_and_isolated() -> None:
    cfg = get_baseline_config()
    P_base = get_mobility_matrix(cfg, 1.0)
    r0_network = compute_spectral_r0(cfg, P_base)
    isolated_r0 = compute_isolated_r0(cfg)

    # Network baseline
    assert abs(r0_network - 2.195901) < 1e-4

    # Isolated district reproduction numbers
    assert abs(isolated_r0[0] - 2.456690) < 1e-4  # Muzaffarpur (supercritical)
    assert abs(isolated_r0[4] - 0.853446) < 1e-4  # Munger (subcritical)

    # Decoupled limit identity: R0(0) = max_k R0^k
    P_0 = get_mobility_matrix(cfg, 0.0)
    r0_decoupled = compute_spectral_r0(cfg, P=P_0)
    assert abs(r0_decoupled - float(np.max(isolated_r0))) < 1e-9


def test_mobility_sweep_properties() -> None:
    cfg = get_baseline_config()
    thetas = np.linspace(0.0, 1.5, 7)
    for theta in thetas:
        P_theta = get_mobility_matrix(cfg, float(theta))
        assert np.all(P_theta >= 0.0)
        assert np.allclose(P_theta.sum(axis=1), 1.0)


def test_budget_sweep_regime_split() -> None:
    # Anchor check at endpoints
    sweep = run_budget_sweep(np.array([0.20, 0.50, 0.80]))

    gain_low = sweep["relative_gain_pct"][0]
    gain_mid = sweep["relative_gain_pct"][1]
    gain_high = sweep["relative_gain_pct"][2]

    # At low budget B=0.20: Policy C underperforms Policy A (negative relative gain)
    assert gain_low < 0.0, f"Expected negative gain at B=0.20, got {gain_low}"
    assert abs(gain_low - (-2.8163)) < 0.1

    # At medium budget B=0.50: Policy C outperforms Policy A
    assert gain_mid > 0.0

    # At high budget B=0.80: Policy C outperforms Policy A by ~19.6%
    assert gain_high > 15.0
    assert abs(gain_high - 19.5779) < 0.1


def test_csv_data_integrity() -> None:
    mobility_csv = REPO_ROOT / "data" / "deds_mobility_sweep.csv"
    assert mobility_csv.exists()
    lines = mobility_csv.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 62  # header + 61 points

    budget_csv = REPO_ROOT / "data" / "deds_budget_sweep.csv"
    assert budget_csv.exists()
    lines_b = budget_csv.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines_b) == 14  # header + 13 points
