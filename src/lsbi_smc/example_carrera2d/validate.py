"""Step 4: compare the LSBI-SMC posterior against the analytic solution.

Reports the paper's error measures (Eqs. 33-34: average relative error of the
mean and of the variance) and overlays the SMC samples on the analytic posterior
density (saved to ``posterior_compare.png``).
"""

from __future__ import annotations

import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import multivariate_normal

from lsbi_smc.example_carrera2d.problem import analytic_posterior, log_evidence


def rel_err(est: np.ndarray, ref: np.ndarray) -> float:
    """Average relative error 1/n sum |est - ref| / |ref| (Eqs. 33-34)."""
    return float(np.mean(np.abs(est - ref) / np.abs(ref)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--posterior", type=str, default="posterior.npz")
    parser.add_argument("--out", type=str, default="posterior_compare.png")
    args = parser.parse_args()

    pop = np.load(args.posterior)["pop"]
    mu_hat = pop.mean(axis=0)
    cov_hat = np.cov(pop.T)
    var_hat = np.diag(cov_hat)

    mu_d, sigma_d = analytic_posterior()
    var_d = np.diag(sigma_d)
    ln_z = log_evidence()

    np.set_printoptions(precision=4, suppress=True)
    print("=== LSBI-SMC vs analytic (Example 1, Carrera & Papaioannou 2024) ===")
    print(f"mean   analytic = {mu_d}   SMC = {mu_hat}")
    print(f"std    analytic = {np.sqrt(var_d)}   SMC = {np.sqrt(var_hat)}")
    print(f"corr   analytic = {sigma_d[0, 1] / np.sqrt(var_d[0] * var_d[1]):.4f}"
          f"   SMC = {cov_hat[0, 1] / np.sqrt(var_hat[0] * var_hat[1]):.4f}")
    print(f"ln Z   analytic = {ln_z:.4f}   (SMC evidence not extracted here)")
    print(f"err_mu  (Eq.33) = {rel_err(mu_hat, mu_d):.4f}")
    print(f"err_var (Eq.34) = {rel_err(var_hat, var_d):.4f}")

    # plot analytic posterior contours + SMC samples
    lo = np.minimum(mu_d - 4 * np.sqrt(var_d), pop.min(axis=0))
    hi = np.maximum(mu_d + 4 * np.sqrt(var_d), pop.max(axis=0))
    gx, gy = np.mgrid[lo[0]:hi[0]:200j, lo[1]:hi[1]:200j]
    grid = np.dstack((gx, gy))
    dens = multivariate_normal(mu_d, sigma_d).pdf(grid)

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.contour(gx, gy, dens, levels=6, cmap="viridis")
    ax.scatter(pop[:, 0], pop[:, 1], s=4, alpha=0.15, color="tomato", label="SMC samples")
    ax.scatter(*mu_d, marker="x", color="black", s=80, label="analytic mean")
    ax.scatter(*mu_hat, marker="+", color="blue", s=80, label="SMC mean")
    ax.set_xlabel(r"$\theta_1$")
    ax.set_ylabel(r"$\theta_2$")
    ax.set_title("Example 1: LSBI-SMC posterior vs analytic")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
