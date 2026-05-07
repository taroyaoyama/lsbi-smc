# Sequential Monte Carlo (SMC) サンプラー

## 1. なぜ SMC が必要なのか

### MCMC の問題点
通常の MCMC（例：Random Walk MH）の問題：
- **多峰性に弱い**：局所的なモードにトラップされやすい
- **高次元で効率が悪い**：パラメータ空間が広いほど混合が遅い

### SMC の解決策
**段階的な温度スケジューリング（アニーリング）**：

$$\text{事前分布 } p(\theta) \xrightarrow{\beta_1} \xrightarrow{\beta_2} \cdots \xrightarrow{\beta_T=1} \text{事後分布 } p(\theta | x_{\text{obs}})$$

各ステップで尤度の影響を徐々に強めることで、多峰性の探索が可能になる。

---

## 2. SMC の基本アルゴリズム

### 中間分布（Tempered Distribution）

$$\pi_t(\theta) \propto \hat{L}(\theta; x_{\text{obs}})^{\beta_t} \cdot p(\theta), \quad 0 = \beta_0 < \beta_1 < \cdots < \beta_T = 1$$

- $\beta_t = 0$：事前分布 $p(\theta)$（尤度を全く考慮しない）
- $\beta_t = 1$：事後分布 $p(\theta | x_{\text{obs}})$

### メインループ

```
初期化：
    粒子 {θ^(n)}_{n=1}^N を事前分布 p(θ) からサンプリング

繰り返し（t = 1, 2, ..., T まで）:
    1. β_t を決定（ESS に基づく）
    2. リサンプリング（重み付きサンプリングで多様性を保つ）
    3. MCMC ムーブ（粒子を中間分布に従って移動）
```

---

## 3. テンパリングパラメータ β の決定

### 有効サンプルサイズ (ESS)

$$\text{ESS}(\beta) = \frac{\left(\sum_n w_n(\beta)\right)^2}{\sum_n w_n(\beta)^2}$$

重み：$w_n(\beta) \propto \hat{L}(\theta^{(n)})^{\beta - \beta_{t-1}}$

- ESS = N：全粒子が等重み（理想的）
- ESS = 1：1粒子に全重みが集中（非効率）

### 適応的な β 選択（二分探索）

目標 ESS（例：$0.8N$）を維持するよう $\beta_t$ を決定：

```python
# smc.py:65-77
def _find_next_q(q_prev, q_tar, lp_np, ess_tar, tol=1e-6, maxit=50):
    lo, hi = q_prev, q_tar
    if _ess_from_lp(hi - q_prev, lp_np) >= ess_tar:
        return hi  # 一気にターゲットまで到達可能
    for _ in range(maxit):
        mid = 0.5 * (lo + hi)
        if _ess_from_lp(mid - q_prev, lp_np) < ess_tar:
            hi = mid
        else:
            lo = mid
        if abs(hi - lo) < tol:
            break
    return 0.5 * (lo + hi)
```

**直感**：β を大きくするほど重みが偏る → ESS が下がる。ESS が目標値を下回らないよう β の増分を制限。

---

## 4. リサンプリング（Resampling）

重みに従って粒子をリサンプリング（多項サンプリング）：

```python
# smc.py:39-45
def resample(self, dq=None):
    self.eval_weights()
    idx = torch.multinomial(self.weights, self.size, replacement=True)
    self.pop = self.pop[idx]      # 重みが大きい粒子を多く複製
    self.lp = self.lp[idx]
```

**目的**：高重みの粒子を複製、低重みの粒子を削除 → 分布の代表性を維持

**問題**：リサンプリングで粒子の多様性が失われる（同じ粒子が複製される）→ MCMC ムーブで解決

---

## 5. MCMC ムーブ（Move Step）

### 5.1 ランダムウォーク Metropolis-Hastings (RW-MH)

```python
# kernel.py:RWMetropolisKernel
def __call__(self, particles, q, prior, likelihood):
    # 1. 提案
    pop_new = self.proposal(particles)

    # 2. 採択比の計算
    log_rat = q * (lp_new - particles.lp) + \
              prior.lp(pop_new) - prior.lp(particles.pop)

    # 3. 採択/棄却
    u = torch.rand_like(log_rat)
    accept = (log_rat > torch.log(u + 1e-12)) & within
```

採択確率：
$$\alpha = \min\left(1, \frac{\pi_t(\theta^*)}{\pi_t(\theta)}\right) = \min\left(1, \exp\left[\beta_t (\log\hat{L}(\theta^*) - \log\hat{L}(\theta)) + \log p(\theta^*) - \log p(\theta)\right]\right)$$

### 5.2 Ching & Chen 提案分布

Ching and Chen (2007) の適応的提案分布：

$$\boldsymbol{\Sigma}_t = b^2 \sum_n w_n (\theta^{(n)} - \bar{\theta})(\theta^{(n)} - \bar{\theta})^T$$

```python
# proposal.py:ChingAndChenProposal
def cov_proposal(self, pop, weights):
    mu = (X * w.unsqueeze(1)).sum(dim=0)     # 重み付き平均
    Xm = X - mu
    cov = Xm.t().mm(torch.diag(w)).mm(Xm)   # 重み付き共分散
    cov = cov * (self.b ** 2)                # スケール調整 (b=0.2)
```

**利点**：現在の粒子分布に適応した提案分布 → 効率的な探索

### 5.3 Hamiltonian Monte Carlo (HMC)

（`kernel.py` の `HMCKernel` クラス）

勾配情報を活用してより効率的な探索：

$$H(\theta, p) = U(\theta) + K(p), \quad U(\theta) = -\log \pi_t(\theta), \quad K(p) = \frac{1}{2}p^T M^{-1} p$$

**リープフロッグ積分器**でハミルトン方程式を解く：

```python
# kernel.py:67-74（リープフロッグ）
for _ in range(self.L):
    th = th.detach().requires_grad_(True)
    g = torch.autograd.grad(U(th).sum(), th)[0]  # 勾配計算
    pp = pp - 0.5 * self.eps * g                  # 半ステップの運動量更新
    th = th + self.eps * (pp / m)                  # 位置更新
    ...
    pp = pp - 0.5 * self.eps * g                  # 残り半ステップの運動量更新
```

PyTorch の自動微分（`torch.autograd.grad`）でニューラルネットの勾配を計算できる。

---

## 6. SMC のコア実装

```python
# smc.py:113-160
def run(self, ess_tar_ratio=0.8, t_max=100, mcmc_iter=1):
    ess_tar = ess_tar_ratio * self.pop_size  # 目標ESS = 0.8 × N

    while self.q[-1] < self.q_tar and t < t_max:
        # Step 1: 次の β を決定
        q_new = _find_next_q(self.q[-1], self.q_tar, lp_np, ess_tar)
        self.q.append(q_new)

        # Step 2: リサンプリング
        self.particles.resample(self.q[-1] - self.q[-2])  # Δβ 分だけ重み付け

        # Step 3: MCMCムーブ（mcmc_iter=10 回繰り返し）
        for _ in range(mcmc_iter):
            pop_new, lp_new, accept = self.kernel(
                self.particles, self.q[-1], self.prior, self.likelihood
            )
            self.particles.replace(accept, pop_new, lp_new)
```

---

## 7. 事前分布と変数クラス

### Uniform 事前分布（直接使用時）
```python
# variables.py:Uniform
θ_i ~ U(a, b)   # 一様分布
```

### inference.py での設定
```python
# 事前分布：標準正規分布（CDF変換と対応）
variables = [Normal(l, Constant(0.0), Constant(1.0)) for l in k_labels]
prior = HierarchicalPrior(variables)
```

HierarchicalPrior は複数の変数を管理し、以下を提供：
- `prior.sample(N)`：事前分布からの一括サンプリング
- `prior.lp(theta)`：対数事前確率
- `prior.check_support(theta)`：サポート内かどうかの確認

---

## 8. 結果の読み方

```python
# smc.py:166-177
def summary(self):
    pop = self.pops[-1].detach().cpu().numpy()
    result = pd.DataFrame({
        'name': self.prior.names,
        'mean': np.mean(pop, axis=0),     # 事後平均
        'sd'  : np.std (pop, axis=0),     # 事後標準偏差
        'q05' : np.percentile(pop,  5, axis=0),  # 5パーセンタイル
        ...
        'q95' : np.percentile(pop, 95, axis=0),  # 95パーセンタイル
    })
```

最終的な粒子 `smc.pops[-1]` が事後分布のサンプルになる。

---

## 9. SMC vs NUTS（論文 Table 2 より）

| アルゴリズム | 粒子数 Ns | 尤度評価回数 | MMD (低いほど良い) | 計算時間 |
|------------|----------|------------|-----------------|---------|
| SMC | 2000 | 362,000 | 0.051 ± 0.019 | **0.8秒** |
| NUTS | - | 558,871 | 0.629 ± 0.159 | 1782秒 |

**SMC の優位性**：
1. GPU並列処理で全粒子を同時評価 → 高速
2. 多峰性の探索に成功（NUTS は局所解に収束）
3. 計算時間が NUTS の約 2000 倍高速
