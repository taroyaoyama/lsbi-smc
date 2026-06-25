"""Generic fully-connected MVAE for vector-valued observations and parameters.

The original ``example_shear4dof.mvae`` uses a 1-D convolutional encoder tailored
to the 1024-point FRF signal. The numerical examples of Carrera & Papaioannou
(2024) instead have *small vector* observations (R^2, 20 deflection points,
70 tomography readings ...) and parameters of varying dimension (2, 10, 102,
236). This module provides an MVAE whose encoders/decoder are plain MLPs,
parameterised by ``obs_dim`` / ``param_dim`` / ``z_dim`` so it can be reused
across all four examples.

It keeps the exact same public surface the rest of the package relies on:

* ``enc_x(x) -> (z, mu, var)`` and ``enc_w(w) -> (z, mu, var)`` returning the
  3-tuple expected by :class:`lsbi_smc.likelihood.latentlik.MVAEBasedLogLikelihood`.
* ``model.loss(xo, xi, w, alp1, alp2, alp3) -> (loss, kl, rec)`` with the same
  argument convention as :meth:`lsbi_smc.example_shear4dof.mvae.MVAE.loss`
  (``xo`` = noisy reconstruction target, ``xi`` = clean encoder input,
  ``w`` = parameter input).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from lsbi_smc.shapes import Latent, Param, Scalar

# ----------------
# util
# ----------------


def reparameterization(mean: Latent, var: Latent) -> Latent:
    epsilon = torch.randn_like(mean)
    return mean + torch.sqrt(var) * epsilon


# --------------------
# loss functions
# --------------------


def gauss_unitgauss_kl(mean: Latent, var: Latent) -> Scalar:
    eps = 1e-8
    _kl = -0.5 * (1 + torch.log(var + eps) - mean**2 - var)
    return torch.sum(_kl, dim=1).mean()


def gauss_gauss_kl(mean1: Latent, var1: Latent, mean2: Latent, var2: Latent) -> Scalar:
    eps = 1e-8
    _var2 = var2 + eps
    _kl = torch.log(_var2) - torch.log(var1 + eps) + (var1 + (mean1 - mean2) ** 2) / _var2 - 1
    return 0.5 * torch.sum(_kl, dim=1).mean()


def rec_loss_norm_2d(
    x: torch.Tensor, mean: torch.Tensor, var: torch.Tensor
) -> Scalar:
    """Negative Gaussian log-likelihood reconstruction loss for (batch, dim) data."""
    return -torch.mean(
        torch.sum(
            -0.5 * ((x - mean) ** 2 / var + torch.log(var) + torch.log(torch.tensor(2 * torch.pi))),
            dim=1,
        )
    )


# --------------------
# building block
# --------------------


class ResBlockFC(nn.Module):
    """Residual MLP block with LeakyReLU and dropout (mirrors ResblockEncSmall)."""

    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.fc1 = nn.Linear(in_dim, out_dim)
        self.fc2 = nn.Linear(out_dim, out_dim)
        self.activation = nn.LeakyReLU(0.2, inplace=True)
        self.dropout = nn.Dropout(p=dropout)
        self.adjust_dim = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.fc1(x)
        out = self.activation(out)
        out = self.fc2(out)
        out = self.dropout(out)
        bypass = self.adjust_dim(x)
        return self.activation(out + bypass)


# --------------------
# Encoder / Decoder
# --------------------


class FCEncoder(nn.Module):
    """MLP encoder mapping a vector to a diagonal-Gaussian latent code."""

    def __init__(self, in_dim: int, z_dim: int, hidden: int = 128, depth: int = 3) -> None:
        super().__init__()
        blocks: list[nn.Module] = [ResBlockFC(in_dim, hidden)]
        for _ in range(depth - 1):
            blocks.append(ResBlockFC(hidden, hidden))
        self.body = nn.Sequential(*blocks)
        self.mu = nn.Linear(hidden, z_dim)
        self.var = nn.Sequential(nn.Linear(hidden, z_dim), nn.Softplus())
        self._initialize_weights()

    def forward(self, x: torch.Tensor) -> tuple[Latent, Latent, Latent]:
        h = self.body(x)
        mu = self.mu(h)
        var = self.var(h) + 1e-6
        z = reparameterization(mu, var)
        return z, mu, var

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)


class FCDecoder(nn.Module):
    """MLP decoder mapping a latent code to a diagonal-Gaussian over observations."""

    def __init__(self, z_dim: int, out_dim: int, hidden: int = 128, depth: int = 3) -> None:
        super().__init__()
        blocks: list[nn.Module] = [ResBlockFC(z_dim, hidden)]
        for _ in range(depth - 1):
            blocks.append(ResBlockFC(hidden, hidden))
        self.body = nn.Sequential(*blocks)
        self.mu = nn.Linear(hidden, out_dim)
        self.var = nn.Sequential(nn.Linear(hidden, out_dim), nn.Softplus())
        self._initialize_weights()

    def forward(self, z: Latent) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.body(z)
        mu = self.mu(h)
        var = self.var(h) + 1e-6
        return mu, var

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)


# --------------------
# MVAE
# --------------------


class MVAEFC(nn.Module):
    """Multimodal VAE with two encoders (observation / parameter) and a shared decoder.

    ``enc_x`` embeds the observation, ``enc_w`` embeds the parameter, and both are
    decoded back to the observation space. The bidirectional latent-KL terms align
    the two latent spaces so that, after training, the analytically tractable
    latent-space likelihood of :mod:`lsbi_smc.likelihood.latentlik` can be used.
    """

    def __init__(
        self,
        obs_dim: int,
        param_dim: int,
        z_dim: int,
        hidden: int = 128,
        depth: int = 3,
    ) -> None:
        super().__init__()
        self.enc_x = FCEncoder(obs_dim, z_dim, hidden, depth)
        self.enc_w = FCEncoder(param_dim, z_dim, hidden, depth)
        self.dec = FCDecoder(z_dim, obs_dim, hidden, depth)

    def encode(
        self, x: torch.Tensor, w: Param
    ) -> tuple[Latent, Latent, Latent, Latent, Latent, Latent]:
        z1, mu1, var1 = self.enc_x(x)
        z2, mu2, var2 = self.enc_w(w)
        return z1, mu1, var1, z2, mu2, var2

    def decode(self, z1: Latent, z2: Latent) -> tuple[torch.Tensor, ...]:
        x_mu1, x_var1 = self.dec(z1)
        x_mu2, x_var2 = self.dec(z2)
        return x_mu1, x_var1, x_mu2, x_var2

    def loss(
        self,
        xo: torch.Tensor,
        xi: torch.Tensor,
        w: Param,
        alp1: float = 1.0,
        alp2: float = 10.0,
        alp3: float = 10.0,
    ) -> tuple[Scalar, Scalar, Scalar]:
        z1, mu1, var1, z2, mu2, var2 = self.encode(xi, w)
        x_mu1, x_var1, x_mu2, x_var2 = self.decode(z1, z2)

        kl1 = gauss_unitgauss_kl(mu1, var1)
        kl2 = gauss_unitgauss_kl(mu2, var2)
        kl_x1x2 = gauss_gauss_kl(mu1, var1, mu2, var2)
        kl_x2x1 = gauss_gauss_kl(mu2, var2, mu1, var1)
        rec_xx = rec_loss_norm_2d(xo, x_mu1, x_var1)
        rec_wx = rec_loss_norm_2d(xo, x_mu2, x_var2)

        loss = kl1 + kl2 + alp1 * (kl_x1x2 + kl_x2x1) + alp2 * rec_xx + alp3 * rec_wx
        return loss, (kl1 + kl2 + kl_x1x2 + kl_x2x1), (rec_xx + rec_wx)
