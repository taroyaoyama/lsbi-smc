from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.distributions as dist
from torch import Tensor

if TYPE_CHECKING:
    from lsbi_smc.smc.smc import LikelihoodProtocol, Particles, PriorProtocol, ProposalProtocol


class RWMetropolisKernel:
    """
    Random Walk Metropolis-Hastings Kernel
    """

    proposal: ProposalProtocol

    def __init__(self, proposal: ProposalProtocol) -> None:
        self.proposal = proposal

    def __call__(
        self,
        particles: Particles,
        q: float,
        prior: PriorProtocol,
        likelihood: LikelihoodProtocol,
    ) -> tuple[Tensor, Tensor, Tensor]:
        device = particles.pop.device
        pop_new = self.proposal(particles).to(device)

        # support check
        within = prior.check_support(pop_new)
        pop_new[~within] = particles.pop[~within]
        lp_new = likelihood(pop_new)

        # MH acceptance ratio
        assert particles.lp is not None
        log_rat = (
            q * (lp_new - particles.lp)
            + prior.lp(pop_new).to(device)
            - prior.lp(particles.pop).to(device)
        )
        log_rat = torch.nan_to_num(log_rat, neginf=-1e30, posinf=1e30)

        # accept or reject
        u = torch.rand_like(log_rat)
        accept = (log_rat > torch.log(u + 1e-12)) & within

        return pop_new, lp_new, accept


class HMCKernel:
    """
    Hamiltonian Monte Carlo Kernal
    """

    L: int
    eps: float

    def __init__(self, leapfrog_steps: int, eps: float) -> None:
        self.L = leapfrog_steps
        self.eps = eps

    def __call__(
        self,
        particles: Particles,
        q: float,
        prior: PriorProtocol,
        likelihood: LikelihoodProtocol,
    ) -> tuple[Tensor, Tensor, Tensor]:
        device = particles.pop.device
        dtype = particles.pop.dtype
        b_size, n_dim = particles.pop.shape

        def potential_energy(th: Tensor) -> Tensor:
            lp = q * likelihood(th) + prior.lp(th)
            return -lp

        # ---------------------
        # leapfrog integrator
        # ---------------------

        # momentum prior
        m = torch.ones(n_dim, device=device)
        mvn = dist.MultivariateNormal(torch.zeros(n_dim).to(device), torch.diag(m))

        # initialize
        th = particles.pop.detach().clone().to(device)
        pp = mvn.sample((len(th),)).to(device)
        with torch.no_grad():
            h0 = potential_energy(th) + 0.5 * (pp**2 / m).sum(dim=1)

        # leapfrog integrator
        for _ in range(self.L):
            th = th.detach().requires_grad_(True)
            g = torch.autograd.grad(potential_energy(th).sum(), th)[0]
            with torch.no_grad():
                pp = pp - 0.5 * self.eps * g
                th = th + self.eps * (pp * 1 / m)
            th = th.detach().requires_grad_(True)
            g = torch.autograd.grad(potential_energy(th).sum(), th)[0]
            with torch.no_grad():
                pp = pp - 0.5 * self.eps * g

        # M-H rule
        with torch.no_grad():
            h1 = potential_energy(th) + 0.5 * (pp**2 / m).sum(dim=1)
            dh = h1 - h0
            acc = torch.exp(-dh).clamp(max=1.0)
            u = torch.rand(b_size, device=device, dtype=dtype)
            accept = u < acc

        # support check
        within = prior.check_support(th)
        accept = accept & within

        # replace
        pop_new = particles.pop.clone()
        pop_new[accept] = th[accept]

        # evaluate likelihood
        lp_new = likelihood(pop_new)

        return pop_new, lp_new, accept
