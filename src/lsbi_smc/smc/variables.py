import torch
import torch.distributions as D


class ConstantVector:
    def __init__(self, value, device=None, dtype=torch.float32):
        t = torch.as_tensor(value, dtype=dtype)
        if device is not None:
            t = t.to(device)
        self._values = t.reshape(1, -1)
        self.depth = 0
        self.dim = self._values.shape[1]

    def values(self, n=1, device=None):
        v = torch.tile(self._values, (n, 1))
        if device is not None:
            v = v.to(device)
        return v


class Constant(ConstantVector):
    def __init__(self, value):
        super().__init__([value])


class Uniform:
    """
    Class for Uniform random variables.
    """

    def __init__(self, name, lower, upper):
        self.name = name
        self.parents = [lower, upper]
        self.depth = max([parent.depth for parent in self.parents]) + 1
        self._values = None
        self.lower = lower
        self.upper = upper
        self.dim = self.lower.dim

    def values(self, n=None):
        if self._values is None:
            raise ValueError(
                "Error: self.values is not initialized. Call sample() before accessing values."
            )
        if n is not None and len(self._values) != n:
            raise ValueError(f"Error: Expected {n} samples, but got {len(self._values)}.")
        return self._values

    def sample(self, n, detach=False):
        device = self.lower._values.device if hasattr(self.lower, "_values") else None
        lv = self.lower.values(n, device=device)
        uv = self.upper.values(n, device=device)
        sampler = D.Uniform(lv, uv)
        values = sampler.sample((1,))[0]
        if detach:
            return values
        else:
            self._values = values

    def lp(self, values=None):
        if values is None:
            values = self.values()
        n = len(values)
        lv = self.lower.values(n, device=values.device)
        uv = self.upper.values(n, device=values.device)
        sampler = D.Uniform(lv, uv)
        return sampler.log_prob(values)

    def check_support(self, values):
        n = len(values)
        lv = self.lower.values(n, device=values.device)
        uv = self.upper.values(n, device=values.device)
        return (values >= lv) & (values <= uv)


class Normal:
    """
    Class for Normal random variables.
    """

    def __init__(self, name, mu, sg):
        self.name = name
        self.parents = [mu, sg]
        self.depth = max([parent.depth for parent in self.parents]) + 1
        self._values = None
        self.mu = mu
        self.sg = sg
        self.dim = self.mu.dim

    def values(self, n=None):
        if self._values is None:
            raise ValueError(
                "Error: self.values is not initialized. Call sample() before accessing values."
            )
        if n is not None and len(self._values) != n:
            raise ValueError(f"Error: Expected {n} samples, but got {len(self._values)}.")
        return self._values

    def sample(self, n, detach=False):
        mv = self.mu.values(n)
        sv = self.sg.values(n)
        if sv.device != mv.device:
            sv = sv.to(mv.device)
        sampler = D.Normal(mv, sv)
        values = sampler.sample((1,))[0]
        if detach:
            return values
        else:
            self._values = values

    def lp(self, values=None):
        if values is None:
            values = self.values()
        n = len(values)
        mv = self.mu.values(n)
        sv = self.sg.values(n)
        if mv.device != values.device:
            mv = mv.to(values.device)
        if sv.device != values.device:
            sv = sv.to(values.device)
        sampler = D.Normal(mv, sv)
        return sampler.log_prob(values)

    def check_support(self, values):
        return torch.ones_like(values, dtype=torch.bool, device=values.device)


class HalfNormal(Normal):
    """
    Class for Half-normal random variables.
    """

    def __init__(self, name, sg):
        super().__init__(name, Constant(0.0), sg)

    def sample(self, n, detach=False):
        sv = self.sg.values(n)
        sampler = D.HalfNormal(sv)
        values = sampler.sample((1,))[0]
        if detach:
            return values
        else:
            self._values = values

    def lp(self, values=None):
        if values is None:
            values = self.values()
        n = len(values)
        sv = self.sg.values(n)
        if sv.device != values.device:
            sv = sv.to(values.device)
        sampler = D.HalfNormal(sv)
        return sampler.log_prob(values)

    def check_support(self, values):
        return values >= 0


class Laplace:
    """
    Class for Laplace random variables.
    """

    def __init__(self, name, mu, b):
        """
        Laplace(mu, b)   （location mu, scale b > 0）
        """
        self.name = name
        self.parents = [mu, b]
        self.depth = max([parent.depth for parent in self.parents]) + 1
        self._values = None
        self.mu = mu  # location
        self.b = b  # scale (>0)
        self.dim = self.mu.dim

    def values(self, n=None):
        if self._values is None:
            raise ValueError(
                "Error: self.values is not initialized. Call sample() before accessing values."
            )
        if n is not None and len(self._values) != n:
            raise ValueError(f"Error: Expected {n} samples, but got {len(self._values)}.")
        return self._values

    def sample(self, n, detach=False):
        mv = self.mu.values(n)
        bv = self.b.values(n)
        if bv.device != mv.device:
            bv = bv.to(mv.device)

        sampler = D.Laplace(mv, bv)
        values = sampler.sample((1,))[0]  # shape: (n, dim)

        if detach:
            return values
        else:
            self._values = values

    def lp(self, values=None):
        if values is None:
            values = self.values()
        n = len(values)

        mv = self.mu.values(n)
        bv = self.b.values(n)

        if mv.device != values.device:
            mv = mv.to(values.device)
        if bv.device != values.device:
            bv = bv.to(values.device)

        sampler = D.Laplace(mv, bv)
        return sampler.log_prob(values)

    def check_support(self, values):
        return torch.ones_like(values, dtype=torch.bool, device=values.device)


class Exponential:
    """
    Class for Exponential random variables.
    """

    def __init__(self, name, rate):
        self.name = name
        self.parents = [rate]
        self.depth = max([parent.depth for parent in self.parents]) + 1
        self._values = None
        self.rate = rate
        self.dim = self.rate.dim

    def values(self, n=None):
        if self._values is None:
            raise ValueError(
                "Error: self.values is not initialized. Call sample() before accessing values."
            )
        if n is not None and len(self._values) != n:
            raise ValueError(f"Error: Expected {n} samples, but got {len(self._values)}.")
        return self._values

    def sample(self, n, detach=False):
        rv = self.rate.values(n)
        sampler = D.Exponential(rv)
        values = sampler.sample((1,))[0]  # shape: (n, dim)
        if detach:
            return values
        else:
            self._values = values

    def lp(self, values=None):
        if values is None:
            values = self.values()
        n = len(values)
        rv = self.rate.values(n)
        if rv.device != values.device:
            rv = rv.to(values.device)
        sampler = D.Exponential(rv)
        return sampler.log_prob(values)

    def check_support(self, values):
        return values >= 0
