# kala-azar-multipatch

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python: >=3.10](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

Reproducibility and computational modeling repository for the journal manuscript:

> **Global Stability and Spectral Thresholds in a Multi-Patch Host–Vector Model of Kala-azar**  
> Raj Kumar Raj and Sandeep Suman*  
> *University Department of Mathematics, Tilka Manjhi Bhagalpur University, Bhagalpur, Bihar 812007, India*  
> Submitted to *Differential Equations and Dynamical Systems* (Springer Nature)  
> Corresponding author: `ssumantmbu@gmail.com` | ORCID: [0000-0001-8616-4767](https://orcid.org/0000-0001-8616-4767)

---

## Overview

This repository provides full numerical simulation scripts, parameter calibrations, spectral analysis routines, and test suites for a nonlinear 25-dimensional host-vector epidemic model with residence-time mobility coupling. The model is parameterised for visceral leishmaniasis (VL / Kala-azar) across five representative districts in Bihar, India (Muzaffarpur, Saran, Patna, Bhagalpur, Munger).

### Key Features
1. **Host-Only Spectral Reduction**: Evaluates the network basic reproduction number $\mathcal{R}_0 = \sqrt{\rho(\mathcal{K}_H)}$ through an aggregated 5-dimensional host matrix $\mathcal{K}_H = \mathcal{F}_{HV}\mathcal{V}_V^{-1}\mathcal{F}_{VH}\mathcal{V}_H^{-1}$ and computes analytical Collatz--Wielandt bounds.
2. **Global Asymptotic Stability (GAS)**: Verifies composite Lyapunov functional $V(x) = u^T \mathcal{V}^{-1} x$ stability bounds for $\mathcal{R}_0 \le 1$, resolving nonhyperbolic dynamics at the threshold $\mathcal{R}_0 = 1$ on the disease-free demographic manifold $\mathcal{D}$.
3. **Mobility-Coupling Sweep $\mathcal{R}_0(\theta)$**: Analyzes the continuous convex mobility blend $P_\theta = (1-\theta)I + \theta P$ across $\theta \in [0, 1.5]$, showing how spatial coupling redistributes transmission risk while sustaining spillover infection in subcritical districts.
4. **Spectral Vector-Control Allocation**: Implements a Karush--Kuhn--Tucker (KKT) water-filling heuristic (Policy C) that projects the Perron eigenvector onto operational bounds, capturing the budget efficiency regime split across low ($\bar{B} \le 0.35$) and high ($\bar{B} \ge 0.50$) resource constraints.

---

## Repository Structure

```
kala-azar-multipatch/
├── README.md                          # Documentation and schema guide
├── LICENSE                            # MIT Open Source License
├── pyproject.toml                     # Python packaging and dependencies
├── data/
│   ├── baseline_parameters.json       # Calibrated district parameters & mobility matrix
│   ├── deds_mobility_sweep.csv        # Numerical output of R0(theta) sweep
│   └── deds_budget_sweep.csv          # Numerical output of budget sensitivity sweep
├── multipatch/
│   ├── __init__.py                    # Multipatch package initialization
│   ├── multipatch_config.py           # District configurations & baseline P matrix
│   ├── multipatch_spectral_r0.py      # Next-generation matrix & spectral radius routines
│   ├── multipatch_ode.py              # 25D ODE system and RK45 simulation drivers
│   ├── multipatch_control_eval.py     # Vector control policies (A, B, C) & KKT projection
│   ├── deds_mobility_sweep.py         # Mobility sweep experiment runner
│   └── deds_budget_sweep.py           # Budget sensitivity sweep runner
├── scripts/
│   ├── run_mobility_sweep.py          # Standalone runner for mobility sweep
│   └── run_budget_sweep.py            # Standalone runner for budget sensitivity
└── tests/
    ├── __init__.py
    └── test_reproducibility.py        # Verification test suite
```

---

## Dataset Schema

### 1. `data/baseline_parameters.json`
Contains demographic, clinical, and vector parameter calibrations for the five districts:
- `districts`: District population $N_H^k$, vector-to-human ratio $n_F^k$, and recovery/treatment rate $r_H^k$.
- `biological_parameters`: Biting rate $a=0.25$, transmission probabilities $b_H=b_V=0.15$, sandfly mortality $d_V=1/14$, human mortality $d_H=1/(70\times 365)$, disease-induced mortality $\delta_H=0.1/365$.
- `baseline_residence_mobility_matrix_P`: Calibrated $5\times 5$ row-stochastic residence-time matrix.

### 2. `data/deds_mobility_sweep.csv`
- `theta`: Mobility coupling intensity parameter $\theta \in [0, 1.5]$ (61 points).
- `R0`: Network basic reproduction number $\mathcal{R}_0(\theta) = \sqrt{\rho(\mathcal{K}_H(\theta))}$.
- `R0_Muzaffarpur`: Isolated reproduction number for Muzaffarpur ($2.457$).
- `R0_Munger`: Isolated reproduction number for Munger ($0.853$).

### 3. `data/deds_budget_sweep.csv`
- `budget`: Mean spray coverage budget $\bar{B} \in [0.20, 0.80]$.
- `burden_A`: 5-year cumulative infection burden under Policy A (uniform spreading, million infection-days).
- `burden_C`: 5-year cumulative infection burden under Policy C (spectral KKT allocation, million infection-days).
- `relative_gain_pct`: $(\text{Burden}_A - \text{Burden}_C) / \text{Burden}_A \times 100\%$.

---

## Installation & Replication

### Prerequisites
- Python $\ge 3.10$
- [uv](https://github.com/astral-sh/uv) (recommended) or standard `pip`

### Step 1: Install Dependencies
```bash
git clone https://github.com/ssumanss/kala-azar-multipatch.git
cd kala-azar-multipatch
uv sync
```

### Step 2: Run Verification Tests
```bash
uv run pytest tests/ -v
```

### Step 3: Reproduce Simulation Sweeps
```bash
# Mobility-coupling sweep
uv run python scripts/run_mobility_sweep.py

# Budget sensitivity sweep
uv run python scripts/run_budget_sweep.py
```

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
