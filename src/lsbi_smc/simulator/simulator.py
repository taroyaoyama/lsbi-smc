from __future__ import annotations

import os
from collections.abc import Callable
from multiprocessing.dummy import Pool as ThreadPool
from typing import Annotated

import numpy as np


class Simulator:
    fun: Callable[
        [Annotated[np.ndarray, "(ndof,)"]],
        Annotated[np.ndarray, "(ndof, n_freq)"],
    ]
    chunksize: int
    llim: float
    ulim: float
    workers: int

    def __init__(
        self,
        fun: Callable[
            [Annotated[np.ndarray, "(ndof,)"]],
            Annotated[np.ndarray, "(ndof, n_freq)"],
        ],
        lims: list[float] | tuple[float, float],
        workers: int | None = None,
        chunksize: int = 8,
    ) -> None:
        self.fun = fun
        self.chunksize = chunksize
        self.llim = lims[0]
        self.ulim = lims[1]
        if workers is None:
            self.workers = os.cpu_count() or 4
        else:
            self.workers = workers

    def __call__(
        self,
        theta: Annotated[np.ndarray, "(n, ndof)"],
    ) -> Annotated[np.ndarray, "(n, ndof, 1, n_freq)"]:
        theta = theta * (self.ulim - self.llim) + self.llim
        with ThreadPool(processes=self.workers) as pool:
            sims_list = pool.map(self.fun, list(theta), self.chunksize)
        sims = np.stack(sims_list, axis=0)
        sims = sims[:, :, None, :].astype(np.float32, copy=False)
        return sims
