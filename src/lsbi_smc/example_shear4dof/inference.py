import numpy as np
from scipy.stats import norm
from scipy import io
import torch
import torch.distributions as D
import random

from example_shear4dof.mvae import MVAE

from example_shear4dof.frfshearm import frfshearm2
from simulator.simulator import *

from smc.smc import *
from smc.prior import *
from smc.kernel import *
from smc.variables import *
from smc.proposal import *

from likelihood.latentlik import *

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# range of parameters
LLIM, ULIM = 0.33, 3.00

# standard normal dist. class
stdnorm = D.Normal(0.0, 1.0)

# normalizer
y_mn, y_sd = -2.1384575366973877, 2.809697389602661

# load mvae model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ndof, z_dim = 4, 8
model = MVAE(z_dim = z_dim, ch = 1, size = 1024, nlabel = ndof, depth = 1).to(device)
model.load_state_dict(
    torch.load(
        'mvae_best.pth', map_location = device
    )['model_state_dict']
)
model.eval()

# -------------------------
# synthetic observation
# -------------------------

LLIM, ULIM = 0.33, 3.00

# simulator
fun = lambda x: frfshearm2(
    x * 1000, ms = 1.0, zeta = 0.02, dlf = 0.020, fmax = 20.48, damping = 'rayleigh',
    omega_target = np.array([1.0, 20.0]) * 2 * np.pi
)
simulator = Simulator(fun, lims = [LLIM, ULIM], workers = 2)

# run
x_obs = np.full((1, ndof), (1.0 - LLIM) / (ULIM - LLIM)).astype(np.float32)
y_obs = simulator(x_obs)[:,[-1],:,:]
y_obs = y_obs + norm.rvs(size = y_obs.shape, random_state = 101) * 0.20
y_obs = (y_obs - y_mn) / y_sd
y_obs = y_obs.astype(np.float32)

# to torch
x_obs_tc = torch.from_numpy(x_obs).to(device)
y_obs_tc = torch.from_numpy(y_obs).to(device)

# --------------------
# define likelihood
# --------------------

class LogLikelihood(MVAEBasedLogLikelihood):
    '''
    Wrapper class to incorpolate 'tau' into mvae-based loglik.
    '''
    def __init__(self, enc_w, enc_x, obs, device):
        super().__init__(enc_w, enc_x, obs, device)
        self.n_call = 0
    
    def __call__(self, theta):
        theta = stdnorm.cdf(theta)
        self.n_call += len(theta)
        return super().__call__(theta, alp = 1.0, tau = 0.00)

loglikelihood = LogLikelihood(model.enc_w, model.enc_x, y_obs_tc, device)

# ---------------------
# Set prior
# ---------------------

k_labels = [f'k{i:02}' for i in range(1, ndof + 1)]
variables = [Normal(l, Constant(0.0), Constant(1.0)) for l in k_labels]
prior = HierarchicalPrior(variables)

# ---------------------
# LSBI-SMC
# ---------------------

# proposal
proposal = ChingAndChenProposal(b = 0.2)

# run!
pop_size = 2000

# run smc
loglikelihood = LogLikelihood(model.enc_w, model.enc_x, y_obs_tc, device)
smc1 = SMC(
    pop_size = pop_size,
    likelihood = loglikelihood,
    prior = prior,
    kernel = RWMetropolisKernel(proposal),
    q_tar = 1.0
)

smc1.run(ess_tar_ratio = 0.8, mcmc_iter = 10)

# extract posterior samples 
pop = smc1.pops[-1].detach().cpu().numpy()
pop = norm.cdf(pop)
pop = pop * (ULIM - LLIM) + LLIM

# save
dic = {'pop': pop, 'n_call': loglikelihood.n_call}
io.savemat('posterior.mat', dic)