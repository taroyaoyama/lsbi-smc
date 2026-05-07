from collections.abc import Sequence
from typing import Annotated

import numpy as np
import torch
from torch import Tensor

from lsbi_smc.smc.variables import DistVar


class HierarchicalPrior:
    """
    Class for flexible hierarchical prior constructed from directed acyclic graph.
    """

    variables: list[DistVar]
    names: list[str]
    index: list[str]
    dim: int
    n_current: int | None

    def __init__(self, variables: Sequence[DistVar]) -> None:
        # sort variables by depth
        depths = np.array([var.depth for var in variables])
        sorted_idx = np.argsort(depths, kind="stable")
        self.variables = [variables[i] for i in sorted_idx]

        # collect names of variables
        self.names = []
        for variable in self.variables:
            self.names += [variable.name + "[" + str(i) + "]" for i in range(variable.dim)]

        # for indexing
        self.index = []
        for variable in self.variables:
            self.index += [variable.name] * variable.dim

        # dimension (necessary?)
        self.dim = len(self.index)

        # current size of values.
        self.n_current = None

    def collect(self) -> Annotated[Tensor, "(n, total_dim)"]:
        theta = [variable.values() for variable in self.variables]
        return torch.cat(theta, dim=-1)

    def assign(self, theta: Annotated[Tensor, "(n, total_dim)"]) -> None:
        n = len(theta)
        self.n_current = n
        for variable in self.variables:
            variable._values = theta[:, np.array(self.index) == variable.name]

    def lp(
        self, values: Annotated[Tensor, "(n, total_dim)"] | None = None
    ) -> Annotated[Tensor, "(n,)"]:
        if values is None:
            values = self.collect()
        else:
            self.assign(values)
        assert self.n_current is not None
        lp = torch.zeros((self.n_current, 1), device=values.device, dtype=values.dtype)
        for variable in self.variables:
            lp += variable.lp(values[:, np.array(self.index) == variable.name])
        return lp[:, 0]

    def sample(self, n: int = 1) -> Annotated[Tensor, "(n, total_dim)"]:
        for variable in self.variables:
            variable.sample(n)
        self.n_current = n
        return self.collect()

    def to_dict(
        self, values: Annotated[Tensor, "(n, total_dim)"] | None = None
    ) -> dict[str, Annotated[Tensor, "(n, dim)"]]:
        if values is None:
            values = self.collect()
        dic: dict[str, Annotated[Tensor, "(n, dim)"]] = {}
        for variable in self.variables:
            dic[variable.name] = values[:, np.array(self.index) == variable.name]
        return dic

    def check_support(
        self, values: Annotated[Tensor, "(n, total_dim)"]
    ) -> Annotated[Tensor, "(n,) bool"]:
        support_checks = []
        for variable in self.variables:
            check_result = variable.check_support(values[:, np.array(self.index) == variable.name])
            support_checks.append(check_result)
        support_concat = torch.cat(support_checks, dim=-1)
        return torch.prod(support_concat, dim=1).bool()
