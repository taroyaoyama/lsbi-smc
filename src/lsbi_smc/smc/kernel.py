import torch
import torch.distributions as D


class RWMetropolisKernel:
    """
    Random Walk Metropolis-Hastings Kernel
    """

    def __init__(self, proposal):
        self.proposal = proposal

    def __call__(self, particles, q, prior, likelihood):
        device = particles.pop.device
        pop_new = self.proposal(particles).to(device)

        # support check
        within = prior.check_support(pop_new)
        pop_new[~within] = particles.pop[~within]
        lp_new = likelihood(pop_new)

        # MH acceptance ratio
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

    def __init__(self, L, eps):
        self.L = L
        self.eps = eps

    def __call__(self, particles, q, prior, likelihood):
        device = particles.pop.device
        dtype = particles.pop.dtype
        B, d = particles.pop.shape

        def U(th):
            lp = q * likelihood(th) + prior.lp(th)
            return -lp

        # ---------------------
        # leapfrog integrator
        # ---------------------

        # momentum prior
        m = torch.ones(d, device=device)
        mvn = D.MultivariateNormal(torch.zeros(d).to(device), torch.diag(m))

        # initialize
        th = particles.pop.detach().clone().to(device)
        pp = mvn.sample((len(th),)).to(device)
        with torch.no_grad():
            H0 = U(th) + 0.5 * (pp**2 / m).sum(dim=1)

        # leapfrog integrator
        for _ in range(self.L):
            th = th.detach().requires_grad_(True)
            g = torch.autograd.grad(U(th).sum(), th)[0]
            with torch.no_grad():
                pp = pp - 0.5 * self.eps * g
                th = th + self.eps * (pp * 1 / m)
            th = th.detach().requires_grad_(True)
            g = torch.autograd.grad(U(th).sum(), th)[0]
            with torch.no_grad():
                pp = pp - 0.5 * self.eps * g

        # M-H rule
        with torch.no_grad():
            H1 = U(th) + 0.5 * (pp**2 / m).sum(dim=1)
            dH = H1 - H0
            acc = torch.exp(-dH).clamp(max=1.0)
            u = torch.rand(B, device=device, dtype=dtype)
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
