from __future__ import annotations

from typing import Literal, overload

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import torch
import torch.nn as nn
from matplotlib.figure import Figure
from torch import Tensor
from torch.nn.utils import spectral_norm
from torch.utils.data import DataLoader

# ----------------
# util
# ----------------


def reparameterization(mean: Tensor, var: Tensor, device: torch.device) -> Tensor:
    epsilon = torch.randn_like(mean)
    return mean + torch.sqrt(var) * epsilon


# --------------------
# loss functions
# --------------------


def gauss_gauss_kl(mean1: Tensor, var1: Tensor, mean2: Tensor, var2: Tensor) -> Tensor:
    eps = 1e-8
    _var2 = var2 + eps
    _kl = torch.log(_var2) - torch.log(var1 + eps) + (var1 + (mean1 - mean2) ** 2) / _var2 - 1
    return 0.5 * torch.sum(_kl, dim=1).mean()


def gauss_unitgauss_kl(mean: Tensor, var: Tensor) -> Tensor:
    eps = 1e-8
    _kl = -0.5 * (1 + torch.log(var + eps) - mean**2 - var)
    return torch.sum(_kl, dim=1).mean()


def rec_loss_norm_4d(x: Tensor, mean: Tensor, var: Tensor) -> Tensor:
    return -torch.mean(
        torch.sum(
            -0.5 * ((x - mean) ** 2 / var + torch.log(var) + torch.log(torch.tensor(2 * torch.pi))),
            dim=(1, 2, 3),
        )
    )


def rec_loss_norm_2d(x: Tensor, mean: Tensor, var: Tensor) -> Tensor:
    return -torch.mean(
        torch.sum(
            -0.5 * ((x - mean) ** 2 / var + torch.log(var) + torch.log(torch.tensor(2 * torch.pi))),
            dim=1,
        )
    )


# --------------------
# residual blocks
# --------------------


class ResblockEnc(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, pooling_size: tuple[int, int]) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=(1, 3), stride=(1, 1), padding=(0, 1), bias=True
        )
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=True
        )
        nn.init.xavier_uniform_(self.conv1.weight.data, 1.0)

        self.model = nn.Sequential(
            nn.ReLU(),
            spectral_norm(self.conv1),
            nn.ReLU(),
            spectral_norm(self.conv2),
            nn.Dropout(p=0.5),
            nn.AvgPool2d(kernel_size=pooling_size, stride=pooling_size, padding=0),
        )

        self.bypass_conv = nn.Conv2d(
            in_channels, out_channels, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), bias=True
        )
        nn.init.xavier_uniform_(self.bypass_conv.weight.data, 1.0)

        self.bypass = nn.Sequential(
            spectral_norm(self.bypass_conv),
            nn.AvgPool2d(kernel_size=pooling_size, stride=pooling_size, padding=0),
        )

        self._initialize_weights()

    def forward(self, x: Tensor) -> Tensor:
        out_x = self.model(x) + self.bypass(x)
        return out_x

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)


class FirstResblockEnc(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, pooling_size: tuple[int, int]) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=(1, 3), stride=(1, 1), padding=(0, 1), bias=True
        )
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=True
        )
        nn.init.xavier_uniform_(self.conv1.weight.data, 1.0)
        nn.init.xavier_uniform_(self.conv2.weight.data, 1.0)

        self.model = nn.Sequential(
            spectral_norm(self.conv1),
            nn.ReLU(),
            spectral_norm(self.conv2),
            nn.Dropout(p=0.5),
            nn.AvgPool2d(kernel_size=pooling_size, stride=pooling_size, padding=0),
        )

        self.bypass_conv = nn.Conv2d(
            in_channels, out_channels, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), bias=True
        )
        nn.init.xavier_uniform_(self.bypass_conv.weight.data, 1.0)

        self.bypass = nn.Sequential(
            spectral_norm(self.bypass_conv),
            nn.AvgPool2d(kernel_size=pooling_size, stride=pooling_size, padding=0),
        )

        self._initialize_weights()

    def forward(self, x: Tensor) -> Tensor:
        out_x = self.model(x) + self.bypass(x)
        return out_x

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)


class ResblockEncSmall(nn.Module):
    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(in_dim, out_dim)
        self.fc2 = nn.Linear(out_dim, out_dim)
        self.activation = nn.LeakyReLU(0.2, inplace=True)
        self.dropout = nn.Dropout(p=0.2)

        # Ensure dimensions match for bypass addition
        self.adjust_dim = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        out = self.fc1(x)
        out = self.activation(out)
        out = self.fc2(out)
        out = self.dropout(out)
        bypass_out = self.adjust_dim(x)  # Adjust dimension only if needed
        return self.activation(out + bypass_out)


class ResblockDec(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        up_sample: tuple[int, int] | float,
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=(1, 3), stride=(1, 1), padding=(0, 1), bias=True
        )
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=True
        )
        nn.init.xavier_uniform_(self.conv1.weight.data, 1.0)
        nn.init.xavier_uniform_(self.conv2.weight.data, 1.0)

        self.model = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Upsample(scale_factor=up_sample, mode="bilinear", align_corners=True),
            self.conv1,
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            self.conv2,
        )

        self.bypass_conv = nn.Conv2d(
            in_channels, out_channels, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), bias=True
        )
        nn.init.xavier_uniform_(self.bypass_conv.weight.data, 1.0)

        self.bypass = nn.Sequential(
            nn.Upsample(scale_factor=up_sample, mode="bilinear", align_corners=True),
            self.bypass_conv,
        )

        self._initialize_weights()

    def forward(self, x: Tensor) -> Tensor:
        out_x = self.model(x) + self.bypass(x)
        return out_x

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)


# --------------------
# Encoders & Decoders
# --------------------


class Encoder(nn.Module):
    device: torch.device

    def __init__(self, z_dim: int, ch: int, size: int, depth: int) -> None:
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.size = size
        self.depth = depth

        self.enc_block_1 = FirstResblockEnc(ch, ch * 2, pooling_size=(1, 2))  # 1024 -> 512
        self.enc_block_2 = ResblockEnc(ch * 2, ch * 4, pooling_size=(1, 2))  #  512 -> 256
        self.enc_block_3 = ResblockEnc(ch * 4, ch * 8, pooling_size=(1, 2))  #  256 -> 128
        self.enc_block_4 = ResblockEnc(ch * 8, ch * 16, pooling_size=(1, 2))  #  128 ->  64
        self.enc_block_5 = ResblockEnc(ch * 16, ch * 32, pooling_size=(1, 2))  #   64 ->  32

        self.blocks = nn.Sequential(
            self.enc_block_1,
            self.enc_block_2,
            self.enc_block_3,
            self.enc_block_4,
            self.enc_block_5,
            nn.LeakyReLU(0.2, inplace=True),
        )

        conv_output_size = ch * 2**5 * int(self.size / 2**5) * self.depth

        self.fc = nn.Sequential(
            nn.Linear(conv_output_size, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.mu = nn.Sequential(nn.Linear(128, z_dim))

        self.var = nn.Sequential(nn.Linear(128, z_dim), nn.Softplus())

        self._initialize_weights()

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        encoded = self.blocks(x)
        encoded = self.fc(encoded.view(-1, encoded.shape[1] * encoded.shape[2] * encoded.shape[3]))
        mu = self.mu(encoded)
        var = self.var(encoded)
        z = reparameterization(mu, var, self.device)
        return z, mu, var

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)


class EncoderW(nn.Module):
    device: torch.device

    def __init__(self, z_dim: int, n_label: int) -> None:
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Residual blocks for small dimensions
        self.res_block1 = ResblockEncSmall(n_label, z_dim)
        self.res_block2 = ResblockEncSmall(z_dim, z_dim * 2)
        self.res_block3 = ResblockEncSmall(z_dim * 2, z_dim * 2)
        self.res_block4 = ResblockEncSmall(z_dim * 2, z_dim)
        self.res_block5 = ResblockEncSmall(z_dim, z_dim)

        # Output layers for mean and variance
        self.mu = nn.Sequential(nn.Linear(z_dim, z_dim))
        self.var = nn.Sequential(nn.Linear(z_dim, z_dim), nn.Softplus())

    def forward(self, w: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        out = self.res_block1(w)
        out = self.res_block2(out)
        out = self.res_block3(out)
        out = self.res_block4(out)
        out = self.res_block5(out)
        mu = self.mu(out)
        var = self.var(out)
        z = reparameterization(mu, var, self.device)
        return z, mu, var


class Decoder(nn.Module):
    def __init__(self, z_dim: int, ch: int, size: int, depth: int) -> None:
        super().__init__()
        self.ch = ch
        self.size = size
        self.depth = depth

        self.fc = nn.Sequential(
            nn.Linear(z_dim, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, ch * 2**5 * int(size / 2**5) * self.depth),
            nn.BatchNorm1d(ch * 2**5 * int(size / 2**5) * self.depth),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.dec_block_3 = ResblockDec(ch * 32, ch * 16, up_sample=(1, 2))  #  32 ->  64
        self.dec_block_4 = ResblockDec(ch * 16, ch * 8, up_sample=(1, 2))  #  64 -> 128
        self.dec_block_5 = ResblockDec(ch * 8, ch * 4, up_sample=(1, 2))  # 128 -> 256
        self.dec_block_6 = ResblockDec(ch * 4, ch * 2, up_sample=(1, 2))  # 256 -> 512

        self.blocks = nn.Sequential(
            self.dec_block_3,
            self.dec_block_4,
            self.dec_block_5,
            self.dec_block_6,
        )

        self.decoder_mu = nn.Sequential(ResblockDec(ch * 2, ch, up_sample=(1, 2)))  # 512 -> 1024
        self.decoder_var = nn.Sequential(
            ResblockDec(ch * 2, ch, up_sample=(1, 2)),  # 512 -> 1024
            nn.Softplus(),
        )

        self._initialize_weights()

    def forward(self, z: Tensor) -> tuple[Tensor, Tensor]:
        xx = self.fc(z)
        decoded = self.blocks(xx.view(-1, self.ch * 2**5, self.depth, int(self.size / 2**5)))
        mu = self.decoder_mu(decoded)
        var = self.decoder_var(decoded)
        return mu, var

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)


# --------------------
# MVAE
# --------------------


class MVAE(nn.Module):
    def __init__(self, z_dim: int, ch: int, size: int, nlabel: int, depth: int) -> None:
        super().__init__()
        self.enc_x = Encoder(z_dim, ch, size, depth)
        self.enc_w = EncoderW(z_dim, nlabel)
        self.dec = Decoder(z_dim, ch, size, depth)

    def encode(self, x: Tensor, w: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        z1, mu1, var1 = self.enc_x(x)
        z2, mu2, var2 = self.enc_w(w)
        return z1, mu1, var1, z2, mu2, var2

    def decode(self, z1: Tensor, z2: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        x_mu1, x_var1 = self.dec(z1)
        x_mu2, x_var2 = self.dec(z2)
        return x_mu1, x_var1, x_mu2, x_var2

    @overload
    def forward(
        self,
        x: Tensor,
        w: Tensor,
        return_loss: Literal[True],
        alp1: float = ...,
        alp2: float = ...,
        alp3: float = ...,
    ) -> tuple[Tensor, Tensor, Tensor]: ...

    @overload
    def forward(
        self,
        x: Tensor,
        w: Tensor,
        return_loss: Literal[False] = ...,
        alp1: float = ...,
        alp2: float = ...,
        alp3: float = ...,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]: ...

    def forward(
        self,
        x: Tensor,
        w: Tensor,
        return_loss: bool = False,
        alp1: float = 1.0,
        alp2: float = 10.0,
        alp3: float = 10.0,
    ) -> tuple[Tensor, Tensor, Tensor] | tuple[Tensor, Tensor, Tensor, Tensor]:
        z1, mu1, var1, z2, mu2, var2 = self.encode(x, w)
        x_mu1, x_var1, x_mu2, x_var2 = self.decode(z1, z2)

        if return_loss:
            # KL terms
            kl1 = gauss_unitgauss_kl(mu1, var1)
            kl2 = gauss_unitgauss_kl(mu2, var2)
            kl_x1x2 = gauss_gauss_kl(mu1, var1, mu2, var2)
            kl_x2x1 = gauss_gauss_kl(mu2, var2, mu1, var1)
            # reconstruction terms
            rec_xx = rec_loss_norm_4d(x, x_mu1, x_var1)
            rec_wx = rec_loss_norm_4d(x, x_mu2, x_var2)
            # total loss
            loss = kl1 + kl2 + alp1 * (kl_x1x2 + kl_x2x1) + alp2 * rec_xx + alp3 * rec_wx
            return loss, (kl1 + kl2 + kl_x1x2 + kl_x2x1), (rec_xx + rec_wx)

        return x_mu1, x_var1, x_mu2, x_var2

    def loss(
        self,
        xo: Tensor,
        xi: Tensor,
        w: Tensor,
        alp1: float = 1.0,
        alp2: float = 10.0,
        alp3: float = 10.0,
    ) -> tuple[Tensor, Tensor, Tensor]:
        z1, mu1, var1, z2, mu2, var2 = self.encode(xi, w)
        x_mu1, x_var1, x_mu2, x_var2 = self.decode(z1, z2)
        # KL terms
        kl1 = gauss_unitgauss_kl(mu1, var1)
        kl2 = gauss_unitgauss_kl(mu2, var2)
        kl_x1x2 = gauss_gauss_kl(mu1, var1, mu2, var2)
        kl_x2x1 = gauss_gauss_kl(mu2, var2, mu1, var1)
        # reconstruction terms
        rec_xx = rec_loss_norm_4d(xo, x_mu1, x_var1)
        rec_wx = rec_loss_norm_4d(xo, x_mu2, x_var2)
        # total loss
        loss = kl1 + kl2 + alp1 * (kl_x1x2 + kl_x2x1) + alp2 * rec_xx + alp3 * rec_wx
        return loss, (kl1 + kl2 + kl_x1x2 + kl_x2x1), (rec_xx + rec_wx)


# --------------------
# VAE
# --------------------


class VAE(nn.Module):
    def __init__(self, z_dim: int, ch: int, size: int, depth: int) -> None:
        super().__init__()
        self.enc = Encoder(z_dim, ch, size, depth)
        self.dec = Decoder(z_dim, ch, size, depth)

    def encode(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        z, mu, var = self.enc(x)
        return z, mu, var

    def decode(self, z: Tensor) -> tuple[Tensor, Tensor]:
        x_mu, x_var = self.dec(z)
        return x_mu, x_var

    @overload
    def forward(self, x: Tensor, return_loss: Literal[True]) -> tuple[Tensor, Tensor, Tensor]: ...

    @overload
    def forward(self, x: Tensor, return_loss: Literal[False] = ...) -> tuple[Tensor, Tensor]: ...

    def forward(
        self, x: Tensor, return_loss: bool = False
    ) -> tuple[Tensor, Tensor, Tensor] | tuple[Tensor, Tensor]:
        # encode
        z, mu, var = self.encode(x)
        # decode
        x_mu, x_var = self.decode(z)

        if return_loss:
            # KL terms
            kl = gauss_unitgauss_kl(mu, var)
            # reconstruction terms
            rec = rec_loss_norm_4d(x, x_mu, x_var)
            # total loss
            loss = kl + 10 * rec
            # return: loss, (sum of KL), (sum of rec)
            return loss, kl, rec

        return x_mu, x_var


# --------------------
# plotter
# --------------------


def plot_frf(
    ids: list[int],
    chs: list[int],
    model: MVAE,
    loader: DataLoader,  # type: ignore[type-arg]
    dlf: float = 0.005,
) -> tuple[Figure, npt.NDArray[np.object_]]:
    device = next(model.parameters()).device
    # read data
    x, y, yn = next(iter(loader))
    x = x.to(device)
    y = y.to(device)
    yn = yn.to(device)
    # prediction
    model.eval()
    with torch.no_grad():
        y_rec, y_rec_var, y_pre, y_pre_var = model(y, x)
    # to cpu, numpy
    y_rec = y_rec.cpu().detach().numpy()
    y_pre = y_pre.cpu().detach().numpy()
    y_tar = yn.cpu().detach().numpy()
    y_rec_var = y_rec_var.cpu().detach().numpy()
    y_pre_var = y_pre_var.cpu().detach().numpy()
    f_axis = np.arange(dlf, dlf * (y_rec.shape[-1] + 1), dlf)
    # plot
    nrow, ncol = len(chs), len(ids)
    fig, ax = plt.subplots(
        nrow, ncol, figsize=(ncol * 2.5, nrow * 1.5), tight_layout=True, sharex=True, sharey=True
    )
    if nrow == 1:
        ax = ax[None, :]
    if ncol == 1:
        ax = ax[:, None]
    for i in range(nrow):
        for j in range(ncol):
            ax[i, j].plot(f_axis, y_tar[ids[j], chs[i], 0, :], color="tomato")
            ax[i, j].plot(f_axis, y_rec[ids[j], chs[i], 0, :], color="orange")
            ax[i, j].plot(f_axis, y_pre[ids[j], chs[i], 0, :], color="royalblue")
            ax[i, j].plot(
                f_axis,
                (y_rec + y_rec_var)[ids[j], chs[i], 0, :],
                linestyle="dashed",
                color="orange",
            )
            ax[i, j].plot(
                f_axis,
                (y_rec - y_rec_var)[ids[j], chs[i], 0, :],
                linestyle="dashed",
                color="orange",
            )
            ax[i, j].plot(
                f_axis,
                (y_pre + y_pre_var)[ids[j], chs[i], 0, :],
                linestyle="dashed",
                color="royalblue",
            )
            ax[i, j].plot(
                f_axis,
                (y_pre - y_pre_var)[ids[j], chs[i], 0, :],
                linestyle="dashed",
                color="royalblue",
            )
            if i == nrow - 1:
                ax[i, j].set_xlabel("Freq, Hz")
            if j == 0:
                ax[i, j].set_ylabel(f"Mag., ch.{chs[i]}")
    return fig, ax
