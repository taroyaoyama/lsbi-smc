from __future__ import annotations

from typing import Any

import torch
import torch.distributions as dist
from torch import Tensor


class ConstantVector:
    _values: Tensor
    depth: int
    dim: int

    def __init__(
        self,
        value: Any,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        t = torch.as_tensor(value, dtype=dtype)
        if device is not None:
            t = t.to(device)
        self._values = t.reshape(1, -1)
        self.depth = 0
        self.dim = self._values.shape[1]

    def values(self, n: int = 1, device: torch.device | str | None = None) -> Tensor:
        v = torch.tile(self._values, (n, 1))
        if device is not None:
            v = v.to(device)
        return v


class Constant(ConstantVector):
    def __init__(self, value: float | int) -> None:
        super().__init__([value])


class Uniform:
    """
    Class for Uniform random variables.
    """

    _values: Tensor | None
    name: str
    parents: list[Any]
    depth: int
    lower: Any
    upper: Any
    dim: int

    def __init__(self, name: str, lower: Any, upper: Any) -> None:
        self.name = name
        self.parents = [lower, upper]
        self.depth = max([parent.depth for parent in self.parents]) + 1
        self._values = None
        self.lower = lower
        self.upper = upper
        self.dim = self.lower.dim

    def values(self, n: int | None = None) -> Tensor:
        if self._values is None:
            raise ValueError(
                "Error: self.values is not initialized. Call sample() before accessing values."
            )
        if n is not None and len(self._values) != n:
            raise ValueError(f"Error: Expected {n} samples, but got {len(self._values)}.")
        return self._values

    def sample(self, n: int, detach: bool = False) -> Tensor | None:
        device = self.lower._values.device if hasattr(self.lower, "_values") else None
        lv = self.lower.values(n, device=device)
        uv = self.upper.values(n, device=device)
        sampler = dist.Uniform(lv, uv)
        values = sampler.sample((1,))[0]
        if detach:
            return values
        self._values = values
        return None

    def lp(self, values: Tensor | None = None) -> Tensor:
        if values is None:
            values = self.values()
        n = len(values)
        lv = self.lower.values(n, device=values.device)
        uv = self.upper.values(n, device=values.device)
        sampler = dist.Uniform(lv, uv)
        return sampler.log_prob(values)

    def check_support(self, values: Tensor) -> Tensor:
        n = len(values)
        lv = self.lower.values(n, device=values.device)
        uv = self.upper.values(n, device=values.device)
        return (values >= lv) & (values <= uv)


class Normal:
    """
    Class for Normal random variables.
    """

    _values: Tensor | None
    name: str
    parents: list[Any]
    depth: int
    mu: Any
    sg: Any
    dim: int

    def __init__(self, name: str, mu: Any, sg: Any) -> None:
        self.name = name
        self.parents = [mu, sg]
        self.depth = max([parent.depth for parent in self.parents]) + 1
        self._values = None
        self.mu = mu
        self.sg = sg
        self.dim = self.mu.dim

    def values(self, n: int | None = None) -> Tensor:
        if self._values is None:
            raise ValueError(
                "Error: self.values is not initialized. Call sample() before accessing values."
            )
        if n is not None and len(self._values) != n:
            raise ValueError(f"Error: Expected {n} samples, but got {len(self._values)}.")
        return self._values

    def sample(self, n: int, detach: bool = False) -> Tensor | None:
        mv = self.mu.values(n)
        sv = self.sg.values(n)
        if sv.device != mv.device:
            sv = sv.to(mv.device)
        sampler = dist.Normal(mv, sv)
        values = sampler.sample((1,))[0]
        if detach:
            return values
        self._values = values
        return None

    def lp(self, values: Tensor | None = None) -> Tensor:
        if values is None:
            values = self.values()
        n = len(values)
        mv = self.mu.values(n)
        sv = self.sg.values(n)
        if mv.device != values.device:
            mv = mv.to(values.device)
        if sv.device != values.device:
            sv = sv.to(values.device)
        sampler = dist.Normal(mv, sv)
        return sampler.log_prob(values)

    def check_support(self, values: Tensor) -> Tensor:
        return torch.ones_like(values, dtype=torch.bool, device=values.device)


class HalfNormal(Normal):
    """
    Class for Half-normal random variables.
    """

    def __init__(self, name: str, sg: Any) -> None:
        super().__init__(name, Constant(0.0), sg)

    def sample(self, n: int, detach: bool = False) -> Tensor | None:
        sv = self.sg.values(n)
        sampler = dist.HalfNormal(sv)
        values = sampler.sample((1,))[0]
        if detach:
            return values
        self._values = values
        return None

    def lp(self, values: Tensor | None = None) -> Tensor:
        if values is None:
            values = self.values()
        n = len(values)
        sv = self.sg.values(n)
        if sv.device != values.device:
            sv = sv.to(values.device)
        sampler = dist.HalfNormal(sv)
        return sampler.log_prob(values)

    def check_support(self, values: Tensor) -> Tensor:
        return values >= 0


class Laplace:
    """
    Class for Laplace random variables.
    """

    _values: Tensor | None
    name: str
    parents: list[Any]
    depth: int
    mu: Any
    b: Any
    dim: int

    def __init__(self, name: str, mu: Any, b: Any) -> None:
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

    def values(self, n: int | None = None) -> Tensor:
        if self._values is None:
            raise ValueError(
                "Error: self.values is not initialized. Call sample() before accessing values."
            )
        if n is not None and len(self._values) != n:
            raise ValueError(f"Error: Expected {n} samples, but got {len(self._values)}.")
        return self._values

    def sample(self, n: int, detach: bool = False) -> Tensor | None:
        mv = self.mu.values(n)
        bv = self.b.values(n)
        if bv.device != mv.device:
            bv = bv.to(mv.device)

        sampler = dist.Laplace(mv, bv)
        values = sampler.sample((1,))[0]  # shape: (n, dim)

        if detach:
            return values
        self._values = values
        return None

    def lp(self, values: Tensor | None = None) -> Tensor:
        if values is None:
            values = self.values()
        n = len(values)

        mv = self.mu.values(n)
        bv = self.b.values(n)

        if mv.device != values.device:
            mv = mv.to(values.device)
        if bv.device != values.device:
            bv = bv.to(values.device)

        sampler = dist.Laplace(mv, bv)
        return sampler.log_prob(values)

    def check_support(self, values: Tensor) -> Tensor:
        return torch.ones_like(values, dtype=torch.bool, device=values.device)


class Exponential:
    """
    Class for Exponential random variables.
    """

    _values: Tensor | None
    name: str
    parents: list[Any]
    depth: int
    rate: Any
    dim: int

    def __init__(self, name: str, rate: Any) -> None:
        self.name = name
        self.parents = [rate]
        self.depth = max([parent.depth for parent in self.parents]) + 1
        self._values = None
        self.rate = rate
        self.dim = self.rate.dim

    def values(self, n: int | None = None) -> Tensor:
        if self._values is None:
            raise ValueError(
                "Error: self.values is not initialized. Call sample() before accessing values."
            )
        if n is not None and len(self._values) != n:
            raise ValueError(f"Error: Expected {n} samples, but got {len(self._values)}.")
        return self._values

    def sample(self, n: int, detach: bool = False) -> Tensor | None:
        rv = self.rate.values(n)
        sampler = dist.Exponential(rv)
        values = sampler.sample((1,))[0]  # shape: (n, dim)
        if detach:
            return values
        self._values = values
        return None

    def lp(self, values: Tensor | None = None) -> Tensor:
        if values is None:
            values = self.values()
        n = len(values)
        rv = self.rate.values(n)
        if rv.device != values.device:
            rv = rv.to(values.device)
        sampler = dist.Exponential(rv)
        return sampler.log_prob(values)

    def check_support(self, values: Tensor) -> Tensor:
        return values >= 0
