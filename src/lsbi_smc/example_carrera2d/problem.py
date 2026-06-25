"""Example 1 of Carrera & Papaioannou (2024): two-dimensional Gaussian example.

Paper spec (Section 5.1):
    - parameters    theta in R^2
    - prior         p0(theta) = N(mu0 = [0, 0], Sigma0 = I)
    - likelihood    L(theta) = N(theta; mu_L, Sigma_L)
                    mu_L  = [2, 2]
                    sigma = [sqrt(0.1), sqrt(0.05)]
                    rho   = 0.98   (correlation of the likelihood covariance)
    - posterior     conjugate Gaussian, available in closed form (Eqs. 29-31)
    - evidence      ln Z available in closed form (Eq. 32)

Casting into the LSBI framework
-------------------------------
A Gaussian likelihood with mean ``mu_L`` is exactly the likelihood of an
*identity* forward model observed under correlated Gaussian noise:

    x = G(theta) + eps,   G(theta) = theta,   eps ~ N(0, Sigma_L),
    observation d = mu_L.

Then  L(theta) = N(d; theta, Sigma_L) = N(theta; mu_L, Sigma_L), matching the
paper. So we train the MVAE on pairs ``(theta, x = theta + eps)`` and feed the
single observation ``d = mu_L`` at inference time. This module is the single
source of truth for the spec and the analytic reference solution.
"""

from __future__ import annotations

import numpy as np

# ----------------------------------------------------------------------
# problem specification
# ----------------------------------------------------------------------

DIM = 2

PRIOR_MEAN = np.zeros(DIM)
PRIOR_COV = np.eye(DIM)

MU_L = np.array([2.0, 2.0])
SIGMA_L_STD = np.array([np.sqrt(0.1), np.sqrt(0.05)])
RHO = 0.98


def sigma_l() -> np.ndarray:
    """Likelihood covariance Sigma_L (also the observation-noise covariance)."""
    cov = RHO * SIGMA_L_STD[0] * SIGMA_L_STD[1]
    return np.array(
        [
            [SIGMA_L_STD[0] ** 2, cov],
            [cov, SIGMA_L_STD[1] ** 2],
        ]
    )


def forward(theta: np.ndarray) -> np.ndarray:
    """Forward model G(theta) = theta (identity)."""
    return np.asarray(theta, dtype=float)


def observation() -> np.ndarray:
    """The single noiseless observation d = mu_L used at inference."""
    return MU_L.copy()


# ----------------------------------------------------------------------
# analytic reference solution
# ----------------------------------------------------------------------


def analytic_posterior() -> tuple[np.ndarray, np.ndarray]:
    """Conjugate Gaussian posterior (mean mu_d, covariance Sigma_d), Eqs. 29-31.

    With standard-normal prior (Sigma0 = I):
        Sigma_d = (I + Sigma_L^-1)^-1
        mu_d    = Sigma_d (Sigma_L^-1 mu_L)
    """
    sl_inv = np.linalg.inv(sigma_l())
    sigma_d = np.linalg.inv(np.eye(DIM) + sl_inv)
    mu_d = sigma_d @ (sl_inv @ MU_L)
    return mu_d, sigma_d


def log_evidence() -> float:
    """Analytic log model evidence ln Z (Eq. 32)."""
    sl = sigma_l()
    sl_inv = np.linalg.inv(sl)
    mu_d, sigma_d = analytic_posterior()
    sd_inv = np.linalg.inv(sigma_d)
    diff = MU_L - PRIOR_MEAN
    return float(
        -0.5
        * (
            DIM * np.log(2 * np.pi)
            + np.log(np.linalg.det(sl) / np.linalg.det(sigma_d))
            + diff @ sl_inv @ diff
            - mu_d @ sd_inv @ mu_d
        )
    )


if __name__ == "__main__":
    mu_d, sigma_d = analytic_posterior()
    np.set_printoptions(precision=4, suppress=True)
    print("Sigma_L      =\n", sigma_l())
    print("posterior mu =", mu_d)
    print("posterior cov=\n", sigma_d)
    print("posterior sd =", np.sqrt(np.diag(sigma_d)))
    print("ln Z         =", log_evidence())
