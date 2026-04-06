import torch
import numpy as np
import pandas as pd


class Particles:
    '''
    Class for particles.
    '''
    def __init__(self, pop_ini):
        self.pop = pop_ini
        self.dq = 0.0
        self.lp = None
        self.weights = None
        self.dim = len(pop_ini[0])
        self.size = len(pop_ini)
    
    def eval_weights(self, dq = None):
        if dq is not None:
            self.dq = dq
        z = self.dq * self.lp
        z = torch.nan_to_num(z, neginf = -1e30, posinf = 1e30)
        z = z - torch.max(z)
        w = torch.exp(z)
        w = torch.nan_to_num(w, nan = 0.0, posinf = 0.0, neginf = 0.0)
        s = w.sum()
        if not torch.isfinite(s) or s <= 0:
            w = torch.full_like(w, 1.0 / len(w))
        else:
            w = w / s
        w = torch.clamp(w, min = 0)
        s = w.sum()
        if s <= 0:
            w = torch.full_like(w, 1.0 / len(w))
        else:
            w = w / s
        self.weights = w

    def resample(self, dq = None):
        if dq is not None:
            self.dq = dq
        self.eval_weights()
        idx = torch.multinomial(self.weights, self.size, replacement = True)
        self.pop = self.pop[idx]
        self.lp = self.lp[idx]

    def replace(self, idx, pop_new, lp_new):
        self.pop[idx] = pop_new[idx]
        self.lp[idx] = lp_new[idx]


def ess(weights_np):
    s1 = weights_np.sum()
    s2 = (weights_np**2).sum()
    return (s1 * s1) / s2


def _ess_from_lp(delta_q, lp_np):
    z = delta_q * lp_np
    z -= z.max()
    w = np.exp(z)
    return ess(w)


def _find_next_q(q_prev, q_tar, lp_np, ess_tar, tol = 1e-6, maxit = 50):
    lo, hi = q_prev, q_tar
    if _ess_from_lp(hi - q_prev, lp_np) >= ess_tar:
        return hi
    for _ in range(maxit):
        mid = 0.5 * (lo + hi)
        if _ess_from_lp(mid - q_prev, lp_np) < ess_tar:
            hi = mid
        else:
            lo = mid
        if abs(hi - lo) < tol:
            break
    return 0.5 * (lo + hi)


class SMC:
    '''
    Class for Sequential Monte Calro Sampler.
    '''
    def __init__(self,
        pop_size,
        likelihood,
        prior,
        kernel,
        q_tar = 1.0
    ):
        self.pop_size = pop_size
        self.likelihood = likelihood
        self.prior = prior
        self.kernel = kernel
        self.pops = []
        self.q = [0]
        self.q_tar = q_tar

        self.device = likelihood.device

        # initialize particles
        pop_ini = self.prior.sample(self.pop_size).to(self.device)
        self.particles = Particles(pop_ini)
        self.particles.lp = self.likelihood(self.particles.pop).to(self.device)
        self.pops.append(pop_ini)
    
    def assign_pop_ini(self, pop):
        pop = pop.to(self.device)
        self.particles = Particles(pop)
        self.particles.lp = self.likelihood(self.particles.pop).to(self.device)
        self.pops = [pop]
    
    def run(self, ess_tar_ratio = 0.8, t_max = 100, print_summary = True, mcmc_iter = 1):
        '''
        Running Sequeintial Monte Carlo.
        '''
        ess_tar = ess_tar_ratio * self.pop_size
        t = 0

        while self.q[-1] < self.q_tar and t < t_max:

            # find next q
            lp_np = self.particles.lp.detach().cpu().numpy()
            q_new = _find_next_q(self.q[-1], self.q_tar, lp_np, ess_tar)
            self.q.append(q_new)

            # resampling
            self.particles.resample(self.q[-1] - self.q[-2])

            # reweighting
            self.particles.weights = torch.full(
                (self.particles.size,),
                1.0 / self.particles.size,
                device = self.device,
                dtype = self.particles.pop.dtype
            )
            self.particles.dq = 0.0

            # # move
            # pop_new, lp_new, accept = self.kernel(
            #     self.particles, self.q[-1], self.prior, self.likelihood
            # )

            # # replace
            # self.particles.replace(accept, pop_new, lp_new)
            # self.pops.append(self.particles.pop.detach())

            # move (run MCMC)
            for _ in range(mcmc_iter):
                pop_new, lp_new, accept = self.kernel(
                    self.particles, self.q[-1], self.prior, self.likelihood
                )
                self.particles.replace(accept, pop_new, lp_new)

            self.pops.append(self.particles.pop.detach())

            print(
                f'(SMC: Target ESS = {ess_tar_ratio:.2f} * N) Iteration {t:02d} | q_t = {self.q[-1]:.5f}'
            )
            t += 1

        if print_summary:
            print('\nResults:')
            print(self.summary())

    def summary(self):
        pop = self.pops[-1].detach().cpu().numpy()
        result = pd.DataFrame({
            'name': self.prior.names,
            'mean': np.mean(pop, axis = 0),
            'sd'  : np.std (pop, axis = 0),
            'q05' : np.percentile(pop,  5, axis = 0),
            'q25' : np.percentile(pop, 25, axis = 0),
            'q50' : np.percentile(pop, 50, axis = 0),
            'q75' : np.percentile(pop, 75, axis = 0),
            'q95' : np.percentile(pop, 95, axis = 0),
        })
        return result