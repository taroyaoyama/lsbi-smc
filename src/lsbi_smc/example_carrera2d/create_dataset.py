"""Step 1: generate the training set for the two-dimensional example.

Draws a Latin-hypercube design of parameters over a box that comfortably covers
both the standard-normal prior and the (concentrated) posterior, evaluates the
identity forward model, and adds correlated Gaussian noise with covariance
``Sigma_L`` (Section 5.1 of Carrera & Papaioannou 2024).

Outputs ``train_data.npz`` (written to the current directory; run from the
project root, like the other example scripts):
    x_sim    (N, 2)  parameters theta
    y_sim    (N, 2)  clean observation  G(theta) = theta
    y_sim_n  (N, 2)  noisy observation  theta + eps,  eps ~ N(0, Sigma_L)
"""

from __future__ import annotations

import argparse

import numpy as np
from scipy.stats import qmc

from lsbi_smc.example_carrera2d.problem import DIM, forward, sigma_l


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-sim", type=int, default=50000, help="number of training samples")
    parser.add_argument(
        "--box", type=float, default=5.0, help="half-width of the LHS sampling box per dimension"
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=str, default="train_data.npz")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)

    # Latin-hypercube design over [-box, box]^DIM (covers prior 5-sigma + posterior).
    sampler = qmc.LatinHypercube(d=DIM, seed=args.seed)
    unit = sampler.random(args.n_sim)
    x_sim = (unit * 2 - 1) * args.box  # -> [-box, box]

    # forward model + correlated observation noise.
    y_sim = forward(x_sim)
    chol = np.linalg.cholesky(sigma_l())
    eps = rng.standard_normal((args.n_sim, DIM)) @ chol.T
    y_sim_n = y_sim + eps

    np.savez(
        args.out,
        x_sim=x_sim.astype(np.float32),
        y_sim=y_sim.astype(np.float32),
        y_sim_n=y_sim_n.astype(np.float32),
        box=args.box,
    )
    print(f"saved {args.out}: x_sim {x_sim.shape}, y_sim {y_sim.shape}, y_sim_n {y_sim_n.shape}")


if __name__ == "__main__":
    main()
