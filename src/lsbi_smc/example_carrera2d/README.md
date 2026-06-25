# example_carrera2d — Example 1 of Carrera & Papaioannou (2024)

Verification that the lab's **LSBI-SMC** method (MVAE latent-space likelihood +
SMC) reproduces the **two-dimensional Gaussian example** (Section 5.1) of

> B. Carrera, I. Papaioannou, *Covariance-based MCMC for high-dimensional
> Bayesian updating with Sequential Monte Carlo*, Prob. Eng. Mech. 77 (2024).

This is **step 1** of the research plan: confirm the LSBI-SMC pipeline behaves on
the paper's setup before scaling to the high-dimensional examples (beam n=10,
n=102, tomography n=236).

## Problem

| | |
|---|---|
| parameters | θ ∈ R² |
| prior | N(0, I) |
| likelihood | N(θ; μ_L, Σ_L), μ_L = [2, 2], σ = [√0.1, √0.05], ρ = 0.98 |
| posterior | conjugate Gaussian, **analytic** (μ_d ≈ [1.705, 1.792]) |
| evidence | ln Z ≈ −5.405 (analytic) |

A Gaussian likelihood with mean μ_L is the likelihood of an **identity forward
model** `x = θ + ε`, `ε ~ N(0, Σ_L)`, observed at `d = μ_L`. We therefore train
the MVAE on `(θ, x = θ + ε)` and feed `d = μ_L` at inference. See
[problem.py](problem.py) for the single source of truth (spec + analytic
solution).

## What is reused vs new

- **Reused unchanged** from the package: `smc/` (SMC, RW-Metropolis kernel,
  Ching–Chen proposal, hierarchical prior) and `likelihood/latentlik.py`
  (`MVAEBasedLogLikelihood`).
- **New, shared** across all Carrera examples: [`mvae_fc.py`](../mvae_fc.py) — a
  fully-connected MVAE parameterised by `obs_dim` / `param_dim` / `z_dim`
  (the shear example's conv encoder is specific to the 1024-point FRF).

## Run (from the project root)

```bash
# Step 1: dataset  ->  train_data.npz
uv run python -m lsbi_smc.example_carrera2d.create_dataset

# Step 2: train MVAE  ->  mvae_best.pth
uv run python -m lsbi_smc.example_carrera2d.train

# Step 3: SMC inference  ->  posterior.npz
uv run python -m lsbi_smc.example_carrera2d.inference

# Step 4: compare to analytic + plot  ->  posterior_compare.png
uv run python -m lsbi_smc.example_carrera2d.validate
```

All scripts accept `--help`. For a quick smoke test use small settings, e.g.
`create_dataset --n-sim 5000`, `train --epochs 30`, `inference --pop-size 500`.

## Success criterion

`validate` prints `err_mu` (Eq. 33) and `err_var` (Eq. 34). The SMC posterior
mean/std/correlation should match the analytic values; the sampler's own error
floor (pCN/RW with finite J) is the reference, so small residual errors are
expected — the goal here is to confirm the **LSBI latent likelihood** does not
add significant bias on the paper's setup.
