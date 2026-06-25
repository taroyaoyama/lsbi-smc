"""Step 3: run LSBI-SMC inference on the two-dimensional example.

Loads the trained MVAE, encodes the single observation d = mu_L, and runs SMC
with the standard-normal prior and the latent-space likelihood. Writes
``posterior.npz`` with the posterior particles and the likelihood-call count.

The prior here is N(0, I), so SMC explores directly in the (standard-normal)
parameter space -- no iso-probabilistic transform is needed, unlike the
shear-building example whose prior lives on a bounded box.
"""

from __future__ import annotations

import argparse

import numpy as np
import torch

from lsbi_smc.example_carrera2d.problem import DIM, observation
from lsbi_smc.likelihood.latentlik import MVAEBasedLogLikelihood
from lsbi_smc.mvae_fc import MVAEFC
from lsbi_smc.shapes import LP, Theta
from lsbi_smc.smc.kernel import RWMetropolisKernel
from lsbi_smc.smc.prior import HierarchicalPrior
from lsbi_smc.smc.proposal import ChingAndChenProposal
from lsbi_smc.smc.smc import SMC
from lsbi_smc.smc.variables import Constant, Normal


class CountingLogLikelihood(MVAEBasedLogLikelihood):
    """Latent likelihood that counts theta evaluations and applies fixed alp/tau.

    SMC calls ``likelihood(theta)`` without extra arguments, so the latent-variance
    inflation knobs are stored on the instance: ``alp`` scales the simulated latent
    variance ((1+alp)*vr_sim) and ``tau`` adds a floor. Increasing them widens /
    de-sharpens the latent likelihood, which is the natural lever when the raw
    encoder variances make the posterior over-confident.
    """

    def __init__(self, *args, alp: float = 1.0, tau: float = 0.0, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.alp = alp
        self.tau = tau
        self.n_call = 0

    def __call__(self, theta: Theta, alp: float | None = None, tau: float | None = None) -> LP:
        self.n_call += len(theta)
        return super().__call__(
            theta,
            alp=self.alp if alp is None else alp,
            tau=self.tau if tau is None else tau,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=str, default="mvae_best.pth")
    parser.add_argument("--out", type=str, default="posterior.npz")
    parser.add_argument("--pop-size", type=int, default=2000)
    parser.add_argument("--ess-ratio", type=float, default=0.8)
    parser.add_argument("--mcmc-iter", type=int, default=10)
    parser.add_argument("--proposal-b", type=float, default=0.2)
    parser.add_argument("--alp", type=float, default=1.0, help="latent-variance scale (1+alp)*vr")
    parser.add_argument("--tau", type=float, default=0.0, help="latent-variance floor added to vr")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(args.model, map_location=device)
    cfg = ckpt["config"]
    y_mn, y_sd = ckpt["y_mn"], ckpt["y_sd"]

    model = MVAEFC(**cfg).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # observation d = mu_L, standardised with the training statistics.
    d = observation().reshape(1, DIM).astype(np.float32)
    d = (d - y_mn) / y_sd
    obs = torch.from_numpy(d).to(device)

    # standard-normal prior, one Normal variable per dimension.
    variables = [Normal(f"theta{i}", Constant(0.0), Constant(1.0)) for i in range(DIM)]
    prior = HierarchicalPrior(variables)

    loglik = CountingLogLikelihood(
        model.enc_w, model.enc_x, obs, device, alp=args.alp, tau=args.tau
    )
    smc = SMC(
        pop_size=args.pop_size,
        likelihood=loglik,
        prior=prior,
        kernel=RWMetropolisKernel(ChingAndChenProposal(b=args.proposal_b)),
        q_tar=1.0,
    )
    smc.run(ess_tar_ratio=args.ess_ratio, mcmc_iter=args.mcmc_iter)

    pop = smc.pops[-1].detach().cpu().numpy()
    np.savez(args.out, pop=pop, n_call=loglik.n_call, n_steps=len(smc.q) - 1)
    print(f"saved {args.out}: pop {pop.shape}, n_call = {loglik.n_call}, steps = {len(smc.q) - 1}")


if __name__ == "__main__":
    main()
