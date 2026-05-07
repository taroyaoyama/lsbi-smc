"""Shape-annotated type aliases used across LSBI-SMC.

Aliases use `Annotated[T, "(shape)"]` with explicit `TypeAlias` annotations.
The plain assignment form (rather than PEP 695 `type` statements) is used here
because Pylance/Pyright resolves `TypeAliasType` inconsistently for operator
methods on `Tensor`, which led to flaky `*`-operator diagnostics.
"""

from typing import Annotated, TypeAlias

import numpy as np
from torch import Tensor

# ============================================================
# Tensor (PyTorch)
# ============================================================

# --- MVAE: per-batch ---
Latent: TypeAlias = Annotated[Tensor, "(batch, z_dim)"]
FRFTensor: TypeAlias = Annotated[Tensor, "(batch, ch, depth, size)"]
Param: TypeAlias = Annotated[Tensor, "(batch, n_label)"]
Scalar: TypeAlias = Annotated[Tensor, "()"]

# --- Observation (single example, batch=1) ---
ObsFRF: TypeAlias = Annotated[Tensor, "(1, ch, depth, n_freq)"]
ObsLatent: TypeAlias = Annotated[Tensor, "(1, z_dim)"]

# --- SMC particles (n = pop_size) ---
Pop: TypeAlias = Annotated[Tensor, "(n, dim)"]
LP: TypeAlias = Annotated[Tensor, "(n,)"]
Mask: TypeAlias = Annotated[Tensor, "(n,) bool"]
Cov: TypeAlias = Annotated[Tensor, "(dim, dim)"]
Theta: TypeAlias = Annotated[Tensor, "(n, ndof)"]
SimLatent: TypeAlias = Annotated[Tensor, "(n, z_dim)"]
FlatParticles: TypeAlias = Annotated[Tensor, "(n, total_dim)"]

# --- Prior building blocks ---
ConstantVal: TypeAlias = Annotated[Tensor, "(1, dim)"]
DistVal: TypeAlias = Annotated[Tensor, "(n, dim)"]
DistMask: TypeAlias = Annotated[Tensor, "(n, dim) bool"]

# ============================================================
# numpy
# ============================================================

NDofVec: TypeAlias = Annotated[np.ndarray, "(ndof,)"]
EigvecMat: TypeAlias = Annotated[np.ndarray, "(ndof, ndof)"]
FRFArray: TypeAlias = Annotated[np.ndarray, "(ndof, n_freq)"]
TwoVec: TypeAlias = Annotated[np.ndarray, "(2,)"]
SimBatch: TypeAlias = Annotated[np.ndarray, "(n, ndof, 1, n_freq)"]
LHSamples: TypeAlias = Annotated[np.ndarray, "(n, ndof)"]
WeightVec: TypeAlias = Annotated[np.ndarray, "(n,)"]
PostSamples: TypeAlias = Annotated[np.ndarray, "(pop_size, ndof)"]

# --- Observation (numpy side, used in inference) ---
ObsParam: TypeAlias = Annotated[np.ndarray, "(1, ndof)"]
ObsFRFArray: TypeAlias = Annotated[np.ndarray, "(1, 1, 1, n_freq)"]
FRFArrayBatch: TypeAlias = Annotated[np.ndarray, "(batch, ch, depth, size)"]
FreqAxis: TypeAlias = Annotated[np.ndarray, "(size,)"]
