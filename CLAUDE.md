# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Purpose

LSBI-SMC implements **Latent space-based Bayesian Inference with Sequential Monte Carlo** for probabilistic structural parameter estimation from vibration data. The target application is finite element model updating of buildings under seismic excitation.

Reference paper: "Finite element model updating of building structures under seismic excitation: A parallelized latent space–based Bayesian framework" (`papers/2603_ned260329.pdf`)

Detailed explanations of the math, ML, and code are in `docs/` (Japanese).

## Development Commands

```bash
# Lint & format
uv run ruff check src/
uv run ruff format src/

# Run a script (always use uv run; scripts assume CWD = project root)
uv run python src/lsbi_smc/example_shear4dof/create_dataset.py   # Step 1: generate train_data.npz
uv run python src/lsbi_smc/example_shear4dof/train.py             # Step 2: train MVAE → mvae_best.pth
uv run python src/lsbi_smc/example_shear4dof/inference.py         # Step 3: run SMC → posterior.mat

# Docker (GPU environment)
make build && make up
```

No test suite exists. Ruff is the only linter/formatter (line length 100, rules: E, F, UP, B, SIM, I, N, A, Q).

## Three-Step Pipeline

```
create_dataset.py  →  train_data.npz
train.py           →  mvae_best.pth
inference.py       →  posterior.mat
```

All three scripts must be run from the project root because they read/write files by relative path (`train_data.npz`, `mvae_best.pth`, `posterior.mat`).

## Architecture

### Core problem
Estimate stiffness parameters θ = (k₁…k₄) of a 4-DOF shear building from a single observed Frequency Response Function (FRF). FRFs are 1024-point vectors — too high-dimensional for direct likelihood evaluation, and the forward simulator is expensive.

### Solution: MVAE + SMC

**Step 1 — Simulator** (`simulator/simulator.py`, `example_shear4dof/frfshearm.py`)  
`Simulator` wraps `frfshearm2`, a physics-based FRF calculator (N-DOF shear building with Rayleigh damping). It parallelizes over a thread pool and normalizes θ from `[0,1]` to `[LLIM, ULIM]` = `[0.33, 3.00]`. Output shape: `(n, n_floors, 1, n_freq)`.

**Step 2 — MVAE** (`example_shear4dof/mvae.py`)  
`MVAE` has three networks sharing a single decoder:
- `Encoder` (enc_x): Conv ResNet, maps FRF `(batch, 1, 1, 1024)` → latent `(batch, z_dim)`
- `EncoderW` (enc_w): FC ResNet, maps parameters `(batch, ndof)` → latent `(batch, z_dim)`
- `Decoder`: maps latent → reconstructed FRF `(mean, var)`

Loss aligns the two latent spaces via bidirectional KL terms (`KL_x1x2 + KL_x2x1`) plus reconstruction terms. The key design: after training, enc_x and enc_w produce compatible latent representations, enabling cross-modal likelihood evaluation.

`model.loss(xo, xi, w)` signature: `xo` = noisy FRF target, `xi` = clean FRF input to enc_x, `w` = parameter input to enc_w.

**Step 3 — Latent likelihood** (`likelihood/latentlik.py`)  
`MVAEBasedLogLikelihood` pre-encodes the observation once (`mu_obs`, `vr_obs`) and at each SMC step encodes candidate θ via enc_w. `latent_space_loglik` computes the analytically tractable log-likelihood by treating both encodings as Gaussians and integrating over the latent prior `N(0,1)`.

**Step 4 — SMC** (`smc/`)  
`SMC` anneals from prior (q=0) to posterior (q=1) by adaptively finding the next inverse temperature q using binary search on ESS (effective sample size). At each step: resample particles by weight → run MCMC moves (RW-Metropolis kernel with Ching & Chen 2007 proposal).

Parameterization: SMC explores in the unconstrained space θ_latent ∈ ℝ (prior N(0,1)), and the likelihood maps to physical space via `Φ(θ_latent) × (ULIM - LLIM) + LLIM`.

### Module dependency
```
inference.py
  ├── mvae.py          (MVAE, Encoder, EncoderW, Decoder)
  ├── latentlik.py     (MVAEBasedLogLikelihood → latent_space_loglik)
  ├── smc.py           (SMC, Particles, _find_next_q)
  ├── kernel.py        (RWMetropolisKernel, HMCKernel)
  ├── proposal.py      (ChingAndChenProposal)
  ├── prior.py         (HierarchicalPrior)
  ├── variables.py     (Normal, Uniform, HalfNormal, Laplace, Exponential, Constant)
  └── simulator.py     (Simulator)
```

### Key data shapes
| Variable | Shape | Description |
|----------|-------|-------------|
| `x_sim` | `(N, 4)` | Normalized parameters [0,1] |
| `y_sim` / `y_sim_n` | `(N, 4, 1, 1024)` | FRF all floors / with noise |
| Training `y` / `yn` | `(batch, 1, 1, 1024)` | Roof FRF only (channel `[-1]`) |
| `mu_obs`, `vr_obs` | `(1, z_dim)` | Latent encoding of observation |
| SMC `pop` | `(pop_size, ndof)` | Particles in unconstrained space |

Only the roof channel (`ch = [-1]`) is used during training and inference, matching the paper's benchmark assumption.

### Prior DAG (`smc/variables.py`)
Variable classes (`Uniform`, `Normal`, `HalfNormal`, `Laplace`, `Exponential`) form a DAG via `depth` ordering. Each implements `sample(n)`, `lp(values)`, `check_support(values)`. `HierarchicalPrior` wraps a list of these into a joint prior with named dimensions.

## Numerical Stability Patterns
- Log weights clipped: `torch.nan_to_num(z, neginf=-1e30, posinf=1e30)`
- Variances clamped: `torch.clamp(vr, min=eps)` before log/division
- Proposal covariance regularized: `cov * b² + ε·I`
- Weight collapse fallback: uniform weights when sum ≤ 0 or non-finite
