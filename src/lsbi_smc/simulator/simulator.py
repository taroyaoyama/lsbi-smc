import os
from multiprocessing.dummy import Pool as ThreadPool

import numpy as np


class Simulator:
    def __init__(self, fun, lims, workers=None, chunksize=8):
        self.fun = fun
        self.chunksize = chunksize
        self.llim = lims[0]
        self.ulim = lims[1]
        if workers is None:
            self.workers = os.cpu_count() or 4
        else:
            self.workers = workers

    def __call__(self, theta):
        theta = theta * (self.ulim - self.llim) + self.llim
        with ThreadPool(processes=self.workers) as pool:
            sims_list = pool.map(self.fun, list(theta), self.chunksize)
        sims = np.stack(sims_list, axis=0)
        sims = sims[:, :, None, :].astype(np.float32, copy=False)
        return sims
