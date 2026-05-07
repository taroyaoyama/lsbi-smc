from typing import Annotated, Literal

import numpy as np
import numpy.typing as npt
from scipy.linalg import eigh


def frfshearm2(
    ks: npt.ArrayLike,
    ms: Annotated[np.ndarray, "(ndof,)"] | float | int,
    zeta: Annotated[np.ndarray, "(2,)"] | float | int,
    damping: Literal["rayleigh", "stiffness"] = "rayleigh",
    omega_target: Annotated[np.ndarray, "(2,)"] | None = None,
    dlf: float = 0.005,
    fmax: float = 5.12,
) -> Annotated[np.ndarray, "(ndof, n_freq)"]:
    """
    Frequency response function for an N-DOF shear building
    with Rayleigh damping: C = a0 M + a1 K.
    """
    ks = np.asarray(ks, dtype=float)
    n_dof = len(ks)

    # mass matrix -----
    ms = np.full(n_dof, float(ms)) if isinstance(ms, (int, float)) else np.asarray(ms, dtype=float)
    m_mat = np.diag(ms)

    # stiffness matrix -----
    k_mat = np.zeros((n_dof, n_dof), dtype=float)
    k_mat[0, 0] += ks[0]
    for i in range(1, n_dof):
        k_mat[(i - 1) : (i + 1), (i - 1) : (i + 1)] += np.array(
            [[ks[i], -ks[i]], [-ks[i], ks[i]]], dtype=float
        )

    # eigenvalue analysis -----
    lam, phi = eigh(k_mat, m_mat)
    omega_nat = np.sqrt(lam)

    # determine coefficents a0 & a1 -----
    if damping == "stiffness":
        if isinstance(zeta, (int, float)):
            zeta_val: float = float(zeta)
        else:
            zeta_val = float(np.asarray(zeta, dtype=float).flat[0])
        a1 = 2.0 * zeta_val / omega_nat[0]
        a0 = 0.0

    elif damping == "rayleigh":
        if omega_target is None:
            omega1, omega2 = omega_nat[0], omega_nat[1]
        else:
            omega_target = np.asarray(omega_target, dtype=float)
            omega1, omega2 = omega_target[0], omega_target[1]
        if isinstance(zeta, (int, float)):
            zeta1 = zeta2 = float(zeta)
        else:
            zeta_arr = np.asarray(zeta, dtype=float)
            zeta1, zeta2 = float(zeta_arr[0]), float(zeta_arr[1])
        a_coeff = np.array(
            [[1 / (2 * omega1), omega1 / 2], [1 / (2 * omega2), omega2 / 2]], dtype=float
        )
        b = np.array([zeta1, zeta2], dtype=float)
        a0, a1 = np.linalg.solve(a_coeff, b)

    else:
        raise ValueError(f"Unknown damping type: {damping!r}")

    # damping ratios -----
    zeta_r = a0 / (2 * omega_nat) + a1 * omega_nat / 2

    # FRF -----
    r = np.ones(n_dof)
    m_r = m_mat @ r
    g = phi.T @ m_r
    omeg = (np.arange(dlf, fmax + dlf, dlf) * 2 * np.pi).astype(float)
    omega_nat_col = omega_nat[:, None]
    zeta_r_col = zeta_r[:, None]
    denom = (
        -(omeg[None, :] ** 2) + 2j * zeta_r_col * omega_nat_col * omeg[None, :] + omega_nat_col**2
    )
    v = (1.0 / denom) * g[:, None]
    u_mat = phi @ v
    h_frf = r[:, None] + (omeg[None, :] ** 2) * u_mat

    return np.log(np.abs(h_frf))


def eigen(
    ks: npt.ArrayLike,
    ms: Annotated[np.ndarray, "(ndof,)"] | float | int,
) -> tuple[
    Annotated[np.ndarray, "(ndof,)"],
    Annotated[np.ndarray, "(ndof, ndof)"],
]:
    """
    Eigenvalue analysis
    """
    ks = np.asarray(ks, dtype=float)
    n_dof = len(ks)

    # mass matrix -----
    ms = np.full(n_dof, float(ms)) if isinstance(ms, (int, float)) else np.asarray(ms, dtype=float)
    m_mat = np.diag(ms)

    # stiffness matrix -----
    k_mat = np.zeros((n_dof, n_dof), dtype=float)
    k_mat[0, 0] += ks[0]
    for i in range(1, n_dof):
        k_mat[(i - 1) : (i + 1), (i - 1) : (i + 1)] += np.array(
            [[ks[i], -ks[i]], [-ks[i], ks[i]]], dtype=float
        )

    # eigenvalue analysis -----
    lam, phi = eigh(k_mat, m_mat)
    omega_nat = np.sqrt(lam)

    return omega_nat, phi
