# kernel.py — MCMC カーネル

**ファイルパス**: `src/lsbi_smc/smc/kernel.py`  
**関連学術ドキュメント**: [07_smc.md](../07_smc.md), [02_probability.md](../02_probability.md)

---

## 概要

SMC の「ムーブステップ」で使用する2種類の MCMC カーネルを実装します。

| クラス | 手法 | 特徴 |
|--------|------|------|
| `RWMetropolisKernel` | ランダムウォーク Metropolis-Hastings | 勾配不要、Ching & Chen 提案と組み合わせ |
| `HMCKernel` | Hamiltonian Monte Carlo | 勾配を使って効率的に探索 |

ベンチマーク実装では `RWMetropolisKernel` が使用される。

### ムーブステップが SMC で果たす役割

カーネルは「中間分布 $\pi_{\beta_{k+1}}$ を **不変分布** とする遷移」を提供する。
不変性により、リサンプリングで $\pi_{\beta_{k+1}}$ に従っている粒子集合を、何ステップ動かしても分布は変わらず、**位置だけが散らばっていく**。これがリサンプリング後の粒子重複（多様性低下）を回復するメカニズム（[07_smc.md §7.1](../07_smc.md)）。

---

## `RWMetropolisKernel` クラス

### 初期化

```python
class RWMetropolisKernel:
    def __init__(self, proposal: ProposalProtocol):
        self.proposal = proposal  # ChingAndChenProposal など
```

### `__call__` メソッド

```python
def __call__(
    self,
    particles: Particles,
    q: float,                # 現在の逆温度 β
    prior: PriorProtocol,
    likelihood: LikelihoodProtocol,
) -> tuple[Tensor, Tensor, Tensor]:
    # 戻り値: (pop_new, lp_new, accept)
```

#### アルゴリズムの詳細

```python
# Step 1: 提案分布からサンプリング
pop_new = self.proposal(particles)       # θ* ← θ + N(0, Σ_proposal)

# Step 2: サポートチェックと棄却
within = prior.check_support(pop_new)   # サポート外の粒子を特定
pop_new[~within] = particles.pop[~within]  # サポート外は現在位置に戻す
lp_new = likelihood(pop_new)            # 提案の尤度評価

# Step 3: Metropolis-Hastings 採択比の計算
log_rat = (
    q * (lp_new - particles.lp)             # 尤度の比（温度スケール）
    + prior.lp(pop_new)                      # 提案の事前確率
    - prior.lp(particles.pop)                # 現在の事前確率
)
log_rat = nan_to_num(log_rat, neginf=-1e30, posinf=1e30)

# Step 4: 採択/棄却
u = torch.rand_like(log_rat)
accept = (log_rat > torch.log(u + 1e-12)) & within
```

#### 採択比の数式

$$\log\alpha = \beta \left[\log\hat{L}(\theta^*) - \log\hat{L}(\theta)\right] + \log p(\theta^*) - \log p(\theta)$$

採択確率：$\min(1, e^{\log\alpha})$

対称な提案分布（$q(\theta|\theta^*) = q(\theta^*|\theta)$）を使うため、提案密度の比はキャンセルされる。

**$\beta$ が掛かっている点**：通常の MH と違うのは、目標が事後分布 $\pi_1$ ではなく中間分布 $\pi_\beta$ だから。$\beta = 0$（事前分布相当）ならば尤度差は採択比に効かず、$\beta = 1$（事後分布）に近づくほど尤度差の影響が大きくなる。

#### `within` との AND

```python
accept = (log_rat > log(u)) & within
```

サポート外の粒子は MH ルールに関わらず棄却する（`within=False` で強制）。

---

## `HMCKernel` クラス

Hamiltonian Monte Carlo を実装。勾配情報を使って効率的に探索できるが、尤度関数が微分可能である必要がある。PyTorch の自動微分（`autograd`）を活用。

### 初期化

```python
class HMCKernel:
    def __init__(self, leapfrog_steps: int, eps: float):
        self.L = leapfrog_steps  # リープフロッグのステップ数
        self.eps = eps           # ステップサイズ
```

### `__call__` メソッド

ハミルトニアン：
$$H(\theta, p) = U(\theta) + K(p)$$
$$U(\theta) = -\beta \log\hat{L}(\theta) - \log p(\theta)$$
$$K(p) = \frac{1}{2} p^T M^{-1} p, \quad M = I$$

#### リープフロッグ積分

```python
def potential_energy(th):
    return -(q * likelihood(th) + prior.lp(th))

# 初期化
th = particles.pop.clone()
pp = MultivariateNormal(0, I).sample((n,))  # モーメンタムをサンプリング
h0 = potential_energy(th) + 0.5 * (pp**2).sum(dim=1)

# リープフロッグ L ステップ
for _ in range(self.L):
    th = th.detach().requires_grad_(True)
    g = autograd.grad(potential_energy(th).sum(), th)[0]   # ∇U
    pp = pp - 0.5 * eps * g    # 半ステップのモーメンタム更新
    th = th + eps * pp         # 位置更新
    g = autograd.grad(potential_energy(th).sum(), th)[0]
    pp = pp - 0.5 * eps * g    # 残り半ステップのモーメンタム更新
```

#### M-H 受理判定

```python
h1 = potential_energy(th) + 0.5 * (pp**2).sum(dim=1)
dh = h1 - h0
acc = exp(-dh).clamp(max=1.0)   # 採択確率
accept = (rand() < acc) & within
```

ハミルトニアンが保存されれば $H_1 = H_0$ → $\Delta H = 0$ → 採択確率 = 1（常に採択）。数値誤差がある場合は M-H ルールで補正。

---

## プロトコルインタフェース

これらのプロトコルは [smc.py:10-45](../../src/lsbi_smc/smc/smc.py#L10-L45) で定義されており、`kernel.py` は `TYPE_CHECKING` ブロックで型ヒント用にインポートしているのみ（[kernel.py:8-9](../../src/lsbi_smc/smc/kernel.py#L8-L9)）。プロトコル定義の本体は次の通り：

```python
class LikelihoodProtocol(Protocol):
    device: torch.device
    def __call__(self, theta: Tensor) -> Tensor: ...

class PriorProtocol(Protocol):
    names: list[str]
    def sample(self, n: int = 1) -> Tensor: ...
    def lp(self, values: Tensor | None = None) -> Tensor: ...
    def check_support(self, values: Tensor) -> Tensor: ...

class ProposalProtocol(Protocol):
    def __call__(self, particles: Particles) -> Tensor: ...

class KernelProtocol(Protocol):
    def __call__(
        self, particles, q, prior, likelihood
    ) -> tuple[Tensor, Tensor, Tensor]: ...
```

プロトコルを使うことで、カーネルを差し替え可能な設計になっている（`RWMetropolisKernel` を `HMCKernel` に変えるだけで使える）。

---

## RW-MH vs HMC の比較

| 項目 | RWMetropolisKernel | HMCKernel |
|------|-------------------|-----------|
| 勾配 | 不要 | 必要 |
| 採択率 | 中程度（粒子数に依存） | 高い（エネルギー保存） |
| 計算コスト | 低い | 高い（L 回の勾配計算） |
| 実装上の制約 | なし | 尤度が微分可能であること |
| ベンチマークでの使用 | ✓ | 未使用 |
