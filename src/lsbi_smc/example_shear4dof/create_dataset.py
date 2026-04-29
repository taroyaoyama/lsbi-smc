import numpy as np
from scipy.stats import norm, qmc

from lsbi_smc.example_shear4dof.frfshearm import frfshearm2
from lsbi_smc.simulator.simulator import Simulator

LLIM, ULIM = 0.33, 3.00


# define simulator
def fun(x):
    return frfshearm2(
        x * 1000,
        ms=1.0,
        zeta=0.02,
        dlf=0.020,
        fmax=20.48,
        damping="rayleigh",
        omega_target=np.array([1.0, 20.0]) * 2 * np.pi,
    )


simulator = Simulator(fun, lims=[LLIM, ULIM], workers=2)

# run simulators
ndof = 4
n_sim = 100000
sampler = qmc.LatinHypercube(d=ndof)
x_sim = sampler.random(n_sim)
y_sim = simulator(x_sim)

# add noise
noise_level = 0.20
y_sim_n = y_sim + noise_level * norm.rvs(size=y_sim.shape)

# export
np.savez("train_data.npz", llim=LLIM, ulim=ULIM, x_sim=x_sim, y_sim=y_sim, y_sim_n=y_sim_n)
