import torch
import torch.distributions as D


class ChingAndChenProposal:
    '''
    Proposal Gaussian distribution given by Ching and Chen (2007).
    '''
    def __init__(self, b):
        self.b = b

    def cov_proposal(self, pop, weights):
        device = pop.device
        X = pop
        w = torch.nan_to_num(weights, nan = 0.0, posinf = 0.0, neginf = 0.0)
        s = w.sum()
        if s <= 0 or not torch.isfinite(s):
            w = torch.full_like(w, 1.0 / len(w))
        w = w / w.sum()
        mu = (X * w.unsqueeze(1)).sum(dim = 0)
        Xm = X - mu
        cov = Xm.t().mm(torch.diag(w)).mm(Xm)
        eps = 1e-6
        cov = cov * (self.b ** 2) + eps * torch.eye(X.shape[1], device = device)
        return cov

    def __call__(self, particles):
        device = particles.pop.device
        cov = self.cov_proposal(particles.pop, particles.weights)
        mvn = D.MultivariateNormal(loc = torch.zeros(particles.dim, device = device), covariance_matrix = cov)
        return particles.pop + mvn.sample((particles.size,))