# Anticipated Q&A

Questions are grouped by audience profile. Each answer is sized to be delivered in **30–60 s**.
References to the relevant slide / file / paper section are included.

---

## 0. Basic check

### Q-0. You said "if the likelihood is known, the posterior follows" — why is likelihood evaluation so hard here?

**A.** Conceptually, Bayes' theorem $p(\theta\mid x_\mathrm{obs}) \propto L(\theta;\,x_\mathrm{obs})\,p(\theta)$ does mean the posterior is determined once we have the likelihood. The problem is two-fold. (i) The FRF is 1024-dimensional, so there is no natural way to write $p(x_\mathrm{obs}\mid\theta)$ explicitly (e.g., as a Gaussian). (ii) Even if we had a form, MCMC would need to evaluate it $10^5$+ times, each requiring an FE call, which is not feasible. The proposed method addresses both simultaneously via the latent-space likelihood.

→ See Slide 1, Slide 5.

---

## A. Questions from a structural engineering angle

### Q-A1. Why use a latent-space likelihood instead of one in FRF space directly?

**A.** Two reasons.

1. **Dimensionality reduction**: FRF is 1024-dim — placing a Gaussian likelihood directly leads to numerical issues with covariance handling. The latent space (~8-dim here) is much easier.
2. **No simulator calls at inference**: a direct likelihood would need an FE run for every new $\theta$, i.e. $10^5$+ FE calls in the MCMC loop. The latent-space likelihood needs only forward passes of the encoders, which is orders of magnitude cheaper.

→ See Slide 5, `src/lsbi_smc/likelihood/latentlik.py`.

---

### Q-A2. What exactly are these "equivalent solutions" and how should they be interpreted physically?

**A.** With a 4-DOF shear model and roof-only observation, **different combinations of story stiffnesses produce the same roof FRF**. Making one story stiffer and another softer can leave the roof response unchanged.

This is an **observational identifiability** issue. Adding intermediate-floor observations would mitigate it. The method's strength is that it honestly **represents this non-identifiability as multi-modality** in the posterior, rather than collapsing onto a single mode.

→ See Slide 8 (the A/B/C markers in the corner plot).

---

### Q-A3. How is damping handled?

**A.** In the current implementation, Rayleigh damping is assumed with **fixed** damping coefficients (only the stiffness parameters $k_i$ are inferred). It is straightforward to include damping in the estimation by adding more variables to `HierarchicalPrior`.

→ See Slide 7, `src/lsbi_smc/example_shear4dof/frfshearm.py`, and Slide 9 prior row.

---

### Q-A4. How realistic is this for real buildings? How does it cope with noise and non-stationarity?

**A.** This paper only verifies on the 4-DOF simulation benchmark. Applying this to real buildings adds several independent considerations:

- Non-white excitation (earthquakes)
- Sensor placement / count constraints
- Model error (FEM does not match the real building structurally)

Each is a separate research direction. The benefit of librarizing is that one can vary the likelihood model or sampler independently and benchmark them.

→ See Slide 10 (future work).

---

### Q-A5. How much does the simulator cost drop?

**A.** Offline data generation still needs $N$ FE runs, but these are **embarrassingly parallel**. During SMC inference itself, there are **no FE calls at all** — only encoder passes.

Quoting paper Table 2: NUTS takes **1782 s** vs SMC's **0.8 s** (with $N_s = 2000$). That is roughly a **2200× speed-up**. Even controlling for total compute, SMC also achieves a lower MMD.

→ See Slide 5, Slide 8, `src/lsbi_smc/simulator/simulator.py`.

---

### Q-A6. How is the equation of motion implemented?

**A.** As stated on Slide 7, $M\ddot{u} + C(\theta)\dot{u} + K(\theta)u = -M\boldsymbol{\iota}\ddot{u}_g$, transformed to the frequency domain. The observable is $\log|H(f)|$ at 1024 points for the roof absolute acceleration. $M$ uses equal masses, $C$ is Rayleigh with 0.02 ratio at 1 Hz and 30 Hz, and $K$ is parameterized by $\theta_i$.

→ See Slide 7, `src/lsbi_smc/example_shear4dof/frfshearm.py`.

---

## B. Questions from a machine-learning angle

### Q-B1. Why multimodal VAE — wouldn't a standard VAE suffice?

**A.** A standard VAE takes **only $x$ as input** and has one encoder, learning a representation of $x$. There's no built-in way to also obtain the latent representation of $\theta$.

MVAE takes **$(\theta, x)$ pairs**, has **two encoders and a shared decoder**, and adds a KL term $\alpha\, D_{KL}(q_{\phi_\theta} \,\|\, q_{\phi_x})$ that **aligns the two encoders' distributions**. This makes $\theta$ and $x_\mathrm{obs}$ comparable in the same latent space, which is the prerequisite for the latent-space likelihood.

→ See Slide 4, `src/lsbi_smc/example_shear4dof/mvae.py`.

---

### Q-B2. Why a two-direction KL? Wouldn't single-direction be enough?

**A.** A single-direction KL only forces one distribution to **cover or seek the mode** of the other, which doesn't guarantee they coincide. **Summing both directions** drives them symmetrically toward agreement.

→ See Slide 4, paper Sec. 3.2, `mvae.py` `loss` method.

---

### Q-B3. How is the latent dimension $d$ chosen?

**A.** Fixed by the paper. In principle it can be picked by the trade-off between reconstruction error and KL term, or even by downstream SMC convergence. The library allows it as a constructor argument, so this is exactly the kind of comparison the library should enable in the future.

→ See `mvae.py` constructor.

---

### Q-B4. How does this compare with more expressive likelihood models like normalizing flows?

**A.** Not compared yet. This is one of the prime future directions; the library's `LikelihoodProtocol` is designed exactly for such drop-in comparisons.

→ See Slide 9, Slide 10.

---

### Q-B5. Why SMC instead of HMC/NUTS?

**A.** For multi-modal posteriors, HMC/NUTS alone tends to get trapped in local modes — they would need parallel tempering to compete. SMC has annealing built in and supports parallel particles, which addresses both **parallelism** and **multi-modality** in one. The benchmarking against other samplers is left as future work.

→ See Slide 6, Slide 10.

---

## C. Questions from a Bayesian / statistical angle

### Q-C1. What approximation, precisely, is "analytic likelihood in latent space"?

**A.** We do not approximate $p(x_\mathrm{obs}\mid\theta)$ directly. Instead, using the two encoders and the latent prior we define

$$\hat{L}(\theta;\,x_\mathrm{obs}) = \int \frac{q_{\phi_x}(z\mid x_\mathrm{obs})\, q_{\phi_\theta}(z\mid \theta)}{p(z)} dz$$

All terms are Gaussian, so the integral is **closed-form via completing the square**. It is better described as "using a latent-space compatibility score as the likelihood" than as "approximating the true likelihood".

→ See Slide 5, paper Sec. 3.1, `latentlik.py` `latent_space_loglik`.

---

### Q-C2. How is the next $\beta$ chosen in SMC?

**A.** We pick the next $\beta_t$ so the **effective sample size** stays above a target fraction (e.g., $\gamma = 0.5$) of the current ESS. The search is done by **bisection**. This adapts the step size: small when the geometry is hard, larger when it is easy.

→ See Slide 6, `src/lsbi_smc/smc/smc.py` `_find_next_q`.

---

### Q-C3. What is the Ching & Chen (2007) proposal?

**A.** A common SMC trick: at each SMC stage, scale the random-walk proposal's covariance by the **weighted covariance of the current particle set**. The particle set itself tells you the proposal's shape — fewer hyperparameters to tune.

→ See Slide 6, `src/lsbi_smc/smc/proposal.py`.

---

### Q-C4. How are priors chosen? Any prior sensitivity?

**A.** The current setup uses physically reasonable uniform priors on $k_i$. Prior sensitivity was not analyzed explicitly. SMC's annealing schedule depends on the prior, so different priors yield different paths — and this is something the library easily enables.

→ See `src/lsbi_smc/smc/prior.py`, `variables.py`.

---

## D. Implementation / software questions

### Q-D1. What concretely changed compared with the paper's published code?

**A.** Three main things:

1. **Protocol-based interfaces**: sampler / kernel / proposal / likelihood / prior pulled out behind `LikelihoodProtocol`, `KernelProtocol`, `ProposalProtocol`, `PriorProtocol`, each in its own file/class.
2. **Numerical stabilization**: log-weight clipping, covariance regularization ($\epsilon I$), variable clamping, etc.
3. **Reproducibility / ops**: DAG-style declarative priors via `HierarchicalPrior`; AMP-enabled training script; `uv`-based environment management.

→ See Slide 9, `docs/08_code_walkthrough.md`, Protocol definitions in `src/lsbi_smc/smc/smc.py`.

---

### Q-D2. Are you planning to distribute this as a library?

**A.** Currently it is a research repo. Once the APIs stabilize and several benchmarks are added, PyPI publication is on the roadmap.

→ See Slide 10.

---

### Q-D3. Are there tests?

**A.** Currently only linting (ruff). Unit tests aren't yet in place. Adding smoke tests per Protocol implementation is one of the next steps.

→ See `pyproject.toml`, `CLAUDE.md`.

---

### Q-D4. Is GPU required?

**A.** Training benefits significantly from GPU, but SMC inference is fast enough on CPU. A Docker setup is included (`compose.yaml`, `Makefile`) for reproducibility.

---

## E. Tougher questions / weak points

### Q-E1. "Isn't this just a re-implementation of the paper?" (Slide 9 challenge)

**A.** Yes — there is no methodological novelty here. The positioning is that the paper's method is re-organized so it can be **used as a substrate for comparison research**. Running comparisons against alternative methods would require non-trivial rewrites of the original code, and lowering that barrier is the contribution. Actually running those comparisons is the next phase. We're being explicit about being at the "foundation" stage, not pretending otherwise.

---

### Q-E2. "If there's no comparison, isn't the contribution weak?"

**A.** Correct — the **comparison results** themselves are not here yet. What is here is

- Reproducing the existing method (true recovery + equivalent modes on the 4-DOF benchmark), and
- Designing a layout where each component can be swapped out.

The comparison experiments are the next phase, on top of this library.

---

### Q-E3. "Equivalent solutions = insufficient observation. How realistic is this?"

**A.** Agreed — under roof-only observation, multi-modality is unavoidable. The method's value is that it **honestly represents this non-identifiability as a multi-modal posterior**, rather than masking it. For real-world use, more sensor coverage would mitigate the issue, and the library is designed to absorb multi-sensor / multi-modality observations as a future extension.

---

### Q-E4. "MVAE training is itself the bottleneck — the offline FE runs are still large."

**A.** True. Offline data generation does require many FE runs. However, this is an **offline, one-shot, parallelizable** cost; the benefit is that the inference loop becomes FE-free. Reducing training data cost (active learning, transfer learning, sample efficiency) is on the list of future work.

---

## Q&A tactics

- **Be honest if unsure**: "We haven't checked that yet; it's on the future-work list."
- **Refer back to slides or code** when asked for numbers or settings (file names are in this Q&A).
- **Distinguish "method vs implementation"**: method-side questions point to the paper; implementation-side questions point to my contribution.
