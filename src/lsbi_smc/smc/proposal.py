from typing import TYPE_CHECKING, Annotated

import torch
import torch.distributions as dist
from torch import Tensor

if TYPE_CHECKING:
    from lsbi_smc.smc.smc import Particles


class ChingAndChenProposal:
    """
    Proposal Gaussian distribution given by Ching and Chen (2007).
    """

    b: float

    def __init__(self, b: float) -> None:
        self.b = b

    def cov_proposal(
        self,
        pop: Annotated[Tensor, "(n, dim)"],
        weights: Annotated[Tensor, "(n,)"],
    ) -> Annotated[Tensor, "(dim, dim)"]:
        device = pop.device
        w = torch.nan_to_num(weights, nan=0.0, posinf=0.0, neginf=0.0)
        s = w.sum()
        if s <= 0 or not torch.isfinite(s):
            w = torch.full_like(w, 1.0 / len(w))
        w = w / w.sum()
        mu = (pop * w.unsqueeze(1)).sum(dim=0)
        x_centered = pop - mu
        cov = x_centered.t().mm(torch.diag(w)).mm(x_centered)
        eps = 1e-6
        cov = cov * (self.b**2) + eps * torch.eye(pop.shape[1], device=device)
        return cov

    def __call__(self, particles: "Particles") -> Annotated[Tensor, "(n, dim)"]:
        assert particles.weights is not None
        device = particles.pop.device
        cov = self.cov_proposal(particles.pop, particles.weights)
        mvn = dist.MultivariateNormal(
            loc=torch.zeros(particles.dim, device=device), covariance_matrix=cov
        )
        return particles.pop + mvn.sample((particles.size,))
