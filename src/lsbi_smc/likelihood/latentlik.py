import torch
import torch.nn as nn

from lsbi_smc.shapes import LP, ObsFRF, ObsLatent, SimLatent, Theta


def latent_space_loglik(
    mu_obs: ObsLatent,
    vr_obs: ObsLatent,
    mu_sim: SimLatent,
    vr_sim: SimLatent,
    alp: float = 0.0,
    tau: float = 1e-4,
    mu_pri: float = 0.0,
    vr_pri: float = 1.0,
    eps: float = 0.0,
) -> LP:
    """
    Latent-space-based likelihood approximation.
    """
    vr1 = torch.clamp(vr_obs, min=eps)
    vr2 = torch.clamp((1 + alp) * vr_sim + tau, min=eps)
    vr3 = torch.clamp(torch.as_tensor(vr_pri, dtype=vr1.dtype, device=vr1.device), min=eps)
    mu1 = mu_obs
    mu2 = mu_sim
    mu3 = torch.as_tensor(mu_pri, dtype=mu1.dtype, device=mu1.device)

    a = (1 / (2 * vr1)) + (1 / (2 * vr2)) - (1 / (2 * vr3))

    b = -(mu1 / vr1) - (mu2 / vr2) + (mu3 / vr3)
    c = (mu1**2 / (2 * vr1)) + (mu2**2 / (2 * vr2)) - (mu3**2 / (2 * vr3))
    d = torch.sqrt(vr3 / (2 * torch.pi * vr1 * vr2))

    lp = torch.log(d) + ((b**2 - 4 * a * c) / (4 * a)) + 0.5 * torch.log(torch.pi / a)
    lp = torch.nan_to_num(lp, neginf=-1e30, posinf=1e30)

    return torch.sum(lp, dim=1)


class MVAEBasedLogLikelihood:
    """
    MVAE-based log. likelihood function.
    """

    device: torch.device
    enc_w: nn.Module
    enc_x: nn.Module
    obs: ObsFRF
    mu_obs: ObsLatent
    vr_obs: ObsLatent

    def __init__(
        self,
        enc_w: nn.Module,
        enc_x: nn.Module,
        obs: ObsFRF,
        device: torch.device,
    ) -> None:
        self.device = device
        self.enc_w = enc_w.to(device)
        self.enc_x = enc_x.to(device)
        self.obs = obs.to(device)
        for p in self.enc_w.parameters():
            p.requires_grad_(False)
        for p in self.enc_x.parameters():
            p.requires_grad_(False)
        self.enc_w.eval()
        self.enc_x.eval()
        with torch.no_grad():
            _, mu_obs, vr_obs = self.enc_x(self.obs)
        self.mu_obs = mu_obs
        self.vr_obs = vr_obs

    @torch.no_grad()
    def __call__(self, theta: Theta, alp: float = 1.0, tau: float = 0.0) -> LP:
        theta = theta.to(self.device)
        _, mu_sim, vr_sim = self.enc_w(theta)
        return latent_space_loglik(self.mu_obs, self.vr_obs, mu_sim, vr_sim, alp=alp, tau=tau)
