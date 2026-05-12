# Speaker script (5 min, for reading aloud)

- **Format**: per-slide reading text + target dwell time
- **Total**: 300 s (5 min)
- **Notation**: 〔…〕 = stage directions (gesture, pause, slide click)

> While rehearsing, aim to deliver the **underlined parts** within the dwell time. Allow ~5 s of slack per slide.

---

## Slide 0: Title (10 s)

〔Stand up → bow〕

"My talk is titled **A Library Implementation of Latent-Space Bayesian Inference with MVAE and SMC**, using Bayesian FE model updating of buildings from observed vibration as a test case. I am 〔name〕. Thank you for having me."

---

## Slide 1: Background (50 s)

"Let me start with the background.

Accurate seismic response prediction for buildings requires a high-fidelity FE model. However, the FE model built at design time **does not match the as-built building** — aging, construction tolerances, contributions from non-structural elements, and so on cause discrepancies.

To close this gap, we want to **Bayesian-update** stiffness parameters $\theta$ from the observed vibration $x_\mathrm{obs}$.

Bayes' theorem tells us the posterior is proportional to **likelihood times prior**. In other words, **if we can evaluate the likelihood $L$, the posterior is within reach** via MCMC and friends.

But for this problem, **evaluating $L$ itself is the bottleneck** — and that is where the research starts."

---

## Slide 2: Goal and challenges (25 s)

"Our goal is to estimate the **posterior** of $\theta$ given the observed FRF $x_\mathrm{obs}$, efficiently. We use the same **4-DOF shear-building** benchmark as the paper for verification.

Three obstacles. First, FRF is 1024-dimensional, so defining a likelihood directly is hard. Second, standard MCMC needs hundreds of thousands of FE calls — not feasible. Third, multiple $\theta$ reproduce the same roof response, making the posterior multi-modal."

---

## Slide 3: Proposed method, overview (30 s)

"The method has an **offline + online** two-stage structure.

Offline, we sample $\theta$ from the prior, run FE analysis to build a dataset, and train an **MVAE** to produce a low-dimensional **approximate likelihood $\hat{L}$**. — **This is where machine learning lives**, and only here.

Online, given the observed FRF $x_\mathrm{obs}$, SMC samples the posterior using $\hat{L}$. **No FE call is made during this inference loop.**"

---

## Slide 4: MVAE — vs. standard VAE (40 s)

"MVAE is the core of the method.

One quick refresher on VAE: a **VAE compresses data into a low-dimensional latent variable and learns to reconstruct it** — a deep generative model.

A standard VAE takes a single input $x$, has one encoder, and the loss is reconstruction error plus a KL term against the prior. In **MVAE**, we feed $(\theta, x)$ pairs, use **two encoders**, and share the decoder.

The key difference is the loss: the additional KL term $\alpha\, D_{KL}(q_{\phi_\theta} \,\|\, q_{\phi_x})$ **aligns the two encoders' distributions**. As a result, $\theta$ and $x_\mathrm{obs}$ become **directly comparable in the same latent space**."

---

## Slide 5: Latent-space likelihood — derivation (35 s)

"Once trained, the likelihood evaluation can be rewritten as follows.

The true likelihood is $\int p(x_\mathrm{obs}\mid z)\, p(z\mid\theta)\, dz$, but this is intractable because of $x$'s dimensionality.

Within the MVAE framework, we rewrite it using **the two encoders and the latent prior**: $\int \frac{q_{\phi_x}(z\mid x_\mathrm{obs})\, q_{\phi_\theta}(z\mid \theta)}{p(z)} dz$.

Since all three terms are **Gaussian**, the integral admits a **closed form via completing the square** — the implementation just evaluates that closed-form expression.

So evaluating the likelihood for a new $\theta$ only requires **passes through the encoders** — **no FE call**."

---

## Slide 6: SMC — annealing, weights, MCMC (35 s)

"SMC is an **annealing** sampler that steps the inverse temperature $\beta$ from 0 (prior) to 1 (posterior).

At each step, particles are reweighted with $w^{(n)}_t(\beta) = \hat{L}^{\beta - \beta_{t-1}}$, and the next $\beta$ is found by **bisection** so that the effective sample size stays above target.

Then we run an MCMC move with a **Gaussian transition kernel** $K(\theta, \cdot) = \mathcal{N}(\cdot \mid \theta, \Sigma_t)$, where $\Sigma_t$ is the **adaptive covariance of Ching & Chen**. Acceptance probability is $\eta = \min\{1, p_t(\theta^*)/p_t(\theta)\}$.

Particles evolve independently, which makes the algorithm **naturally parallel on GPU**, and the annealing structure makes it **robust to multi-modal posteriors**."

---

## Slide 7: Benchmark — 4-DOF shear building (25 s)

"The benchmark is the same 4-DOF shear building as the paper.

It is a four-story model with **each floor only moving in one horizontal direction** — lumped mass, spring, and damper. Story stiffness is $k_i = \theta_i k$, with $\theta_i$ being the estimation target. Damping is **Rayleigh, with 0.02 ratio at 1 Hz and 30 Hz**. Observation is **base excitation + roof FRF only**, which is the main source of multi-modality.

The equation of motion is the usual mass-damping-stiffness form. We transform it to the frequency domain and observe $\log|H(f)|$ of the roof absolute acceleration at 1024 points."

---

## Slide 8: Results (25 s)

"Here are the results.

The posterior's main mode concentrates around the true value $(1, 1, 1, 1)$, and the three known **equivalent solutions A, B, C** are also recovered as separate modes.

For **runtime**, citing the paper's Table 2: NUTS takes about 1782 seconds, whereas SMC with 2000 particles finishes in **just 0.8 seconds** — roughly a **2200× speed-up** thanks to GPU parallelism. SMC also outperforms NUTS in terms of MMD."

---

## Slide 9: What I worked on (40 s)

"That covered the existing method. **My contribution** comes next.

The method itself is exactly the one from the paper. **What I did was re-implement the framework in a structure that can be extended and replaced later.**

Concretely, I pulled five components — sampler, MCMC kernel, proposal, likelihood, prior — behind **Python Protocols**, so each can be swapped via the interface.

That said, the alternative methods themselves are not implemented yet — what I built is the **foundation** for future comparisons.

In addition to the Protocols, the re-implementation also includes module dependency cleanup, uv-based environment management with package publishing, and documentation."

---

## Slide 10: Summary and future work (30 s)

"To summarize.

I re-implemented LSBI-SMC in PyTorch, verified it on the 4-DOF benchmark, and structured it as **Protocol-based extensible components**.

Going forward, the plan is to compare with other samplers — NUTS, Nested Sampling — and with non-MVAE likelihood models, add more observation modalities, scale to larger DOF and nonlinear responses, and turn it into a public benchmark suite.

Thank you for your attention."

〔Bow〕

---

## Time table

| Slide                          | Time | Cumulative |
| ------------------------------ | ---- | ---------- |
| 0 Title                        | 10 s | 0:10       |
| 1 Background                   | 50 s | 1:00       |
| 2 Goal & challenges            | 25 s | 1:25       |
| 3 Overview                     | 30 s | 1:55       |
| 4 MVAE (vs VAE)                | 40 s | 2:35       |
| 5 Latent likelihood            | 35 s | 3:10       |
| 6 SMC                          | 35 s | 3:45       |
| 7 Benchmark                    | 25 s | 4:10       |
| 8 Results                      | 25 s | 4:35       |
| 9 My contribution              | 40 s | 5:15       |
| 10 Summary                     | 30 s | **5:45**   |

> Target 5:45. To hit a clean 5:00, trim slides 4, 5, 6.
> Likely cuts: the acceptance-ratio formula on slide 6, and one of the code-level items on slide 9.

---

## Delivery tips

- **Slide 1**: point at Bayes' theorem; read "left side = posterior, right side = likelihood × prior" slowly
- **Slide 4 (VAE comparison)**: walk the table top-down — "input", "encoder", "loss" — three rows that differ
- **Slide 5**: do not read the equations; gesture-driven, "the shape changes in three stages"
- **Slide 6**: present the equations as a triad — "weight, proposal, acceptance"; do not read symbol by symbol
- **Slide 8**: let the table speak — say "1782 s vs 0.8 s, 2200×" in tempo
- **Slide 9**: physically point at the Protocol column — "this is where a new method plugs in"
