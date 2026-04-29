import numpy as np
from scipy.linalg import eigh


def frfshearm2(ks, ms, zeta, damping="rayleigh", omega_target=None, dlf=0.005, fmax=5.12):
    """
    Frequency response function for an N-DOF shear building
    with Rayleigh damping: C = a0 M + a1 K.
    """
    ks = np.asarray(ks, dtype=float)
    N = len(ks)

    # mass matrix -----
    ms = np.full(N, float(ms)) if np.isscalar(ms) else np.asarray(ms, dtype=float)
    M = np.diag(ms)

    # stiffness matrix -----
    K = np.zeros((N, N), dtype=float)
    K[0, 0] += ks[0]
    for i in range(1, N):
        K[(i - 1) : (i + 1), (i - 1) : (i + 1)] += np.array(
            [[ks[i], -ks[i]], [-ks[i], ks[i]]], dtype=float
        )

    # eigenvalue analysis -----
    lam, Phi = eigh(K, M)
    omega_nat = np.sqrt(lam)

    # determine coefficents a0 & a1 -----
    if damping == "stiffness":
        if not np.isscalar(zeta):
            zeta = zeta[0]
        a1 = 2.0 * zeta / omega_nat[0]
        a0 = 0.0

    elif damping == "rayleigh":
        if omega_target is None:
            omega1, omega2 = omega_nat[0], omega_nat[1]
        else:
            omega1, omega2 = omega_target[0], omega_target[1]
        if np.isscalar(zeta):
            zeta1, zeta2 = float(zeta), float(zeta)
        else:
            zeta = np.asarray(zeta, dtype=float)
            zeta1, zeta2 = zeta[0], zeta[1]
        A = np.array([[1 / (2 * omega1), omega1 / 2], [1 / (2 * omega2), omega2 / 2]], dtype=float)
        b = np.array([zeta1, zeta2], dtype=float)
        a0, a1 = np.linalg.solve(A, b)

    else:
        raise ValueError(f"Unknown damping type: {damping!r}")

    # damping ratios -----
    zeta_r = a0 / (2 * omega_nat) + a1 * omega_nat / 2

    # FRF -----
    r = np.ones(N)
    Mr = M @ r
    g = Phi.T @ Mr
    omeg = (np.arange(dlf, fmax + dlf, dlf) * 2 * np.pi).astype(float)
    omega_nat_col = omega_nat[:, None]
    zeta_r_col = zeta_r[:, None]
    denom = (
        -(omeg[None, :] ** 2) + 2j * zeta_r_col * omega_nat_col * omeg[None, :] + omega_nat_col**2
    )
    v = (1.0 / denom) * g[:, None]
    U = Phi @ v
    H = r[:, None] + (omeg[None, :] ** 2) * U

    return np.log(np.abs(H))


def eigen(ks, ms):
    """
    Eigenvalue analysis
    """
    ks = np.asarray(ks, dtype=float)
    N = len(ks)

    # mass matrix -----
    ms = np.full(N, float(ms)) if np.isscalar(ms) else np.asarray(ms, dtype=float)
    M = np.diag(ms)

    # stiffness matrix -----
    K = np.zeros((N, N), dtype=float)
    K[0, 0] += ks[0]
    for i in range(1, N):
        K[(i - 1) : (i + 1), (i - 1) : (i + 1)] += np.array(
            [[ks[i], -ks[i]], [-ks[i], ks[i]]], dtype=float
        )

    # eigenvalue analysis -----
    lam, Phi = eigh(K, M)
    omega_nat = np.sqrt(lam)

    return omega_nat, Phi
