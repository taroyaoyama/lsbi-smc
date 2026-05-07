# proposal.py — Ching & Chen 提案分布

**ファイルパス**: `src/lsbi_smc/smc/proposal.py`  
**関連学術ドキュメント**: [07_smc.md](../07_smc.md)

---

## 概要

`ChingAndChenProposal` は SMC の MCMC ムーブステップで使用する**適応型ガウス提案分布**を実装します。

Ching and Chen (2007) の手法：現在の粒子群の重み付き共分散行列をスケーリングして提案分布を構成することで、粒子の現在の分布に適応した効率的な探索が可能になります。

---

## `ChingAndChenProposal` クラス

```python
class ChingAndChenProposal:
    b: float    # 提案分布のスケール係数

    def __init__(self, b: float):
        self.b = b
```

`inference.py` では `b=0.2` を使用（論文の推奨値）。

---

## `cov_proposal` メソッド（共分散行列の計算）

```python
def cov_proposal(self, pop: Tensor, weights: Tensor) -> Tensor:
    # 1. 重みの正規化
    w = nan_to_num(weights, 0.0)
    w = w / w.sum()

    # 2. 重み付き平均
    mu = (pop * w.unsqueeze(1)).sum(dim=0)   # shape: (dim,)

    # 3. 重み付き共分散
    x_centered = pop - mu                     # shape: (n, dim)
    cov = x_centered.T @ diag(w) @ x_centered  # shape: (dim, dim)

    # 4. スケーリングと正則化
    eps = 1e-6
    cov = cov * (self.b ** 2) + eps * I
    return cov
```

数式：
$$\boldsymbol{\Sigma}_{\text{proposal}} = b^2 \sum_{n=1}^N w_n (\theta^{(n)} - \bar{\theta})(\theta^{(n)} - \bar{\theta})^T + \varepsilon I$$

- $w_n$：正規化された粒子重み
- $\bar{\theta} = \sum_n w_n \theta^{(n)}$：重み付き平均
- $b = 0.2$：スケール係数（経験的に最適な値）
- $\varepsilon = 10^{-6}$：正則化項（行列が特異にならないように）

**`b=0.2` の意味**: 粒子の分散の $(0.2)^2 = 4\%$ のスケールで提案。現在の分布の広がりに比べて小さなステップを踏むことで、採択率を適度に保つ。

---

## `__call__` メソッド（提案の生成）

```python
def __call__(self, particles: Particles) -> Tensor:
    device = particles.pop.device
    cov = self.cov_proposal(particles.pop, particles.weights)
    mvn = dist.MultivariateNormal(
        loc=torch.zeros(particles.dim, device=device),
        covariance_matrix=cov
    )
    return particles.pop + mvn.sample((particles.size,))
```

### 処理フロー

```
現在の粒子群 pop: (n, dim)
  ↓
  共分散 Σ_proposal の計算
  ↓
  各粒子からの変位をサンプリング: Δθ ~ N(0, Σ_proposal)
  ↓
  提案: θ* = θ + Δθ
出力: pop_new (n, dim)
```

提案はゼロ平均の多変量正規分布からの変位。「現在位置から少しずれた場所」を提案する。

---

## 採択後の処理（`kernel.py` 内）

```python
# RWMetropolisKernel.__call__ での処理
pop_new = proposal(particles)        # 提案を生成（ここで呼ばれる）
lp_new = likelihood(pop_new)         # 提案の尤度を評価
log_rat = q*(lp_new - particles.lp) + prior.lp(pop_new) - prior.lp(particles.pop)
accept = (log_rat > log(u)) & within
```

提案を受け入れるかどうかは `RWMetropolisKernel` が Metropolis-Hastings ルールで判断する（対称な提案なので提案密度の比はキャンセルされる）。

---

## なぜ適応的な提案が重要か

固定スケールのランダムウォーク提案の問題：
- スケールが大きすぎると採択率が低下（ほとんど棄却される）
- スケールが小さすぎると探索が遅い

Ching & Chen の適応的提案の利点：
- 粒子の分布が変化（ステップが進むにつれて集中）しても自動的に調整
- SMC の各ステップで粒子が集まった後も適切なステップサイズを維持
- 重み情報を使うことでリサンプリング前の状態も反映

---

## 数値安定性の配慮

```python
w = torch.nan_to_num(weights, nan=0.0, posinf=0.0, neginf=0.0)
if s <= 0 or not torch.isfinite(s):
    w = torch.full_like(w, 1.0 / len(w))   # 重みが無効なら一様にフォールバック

cov = cov * (self.b**2) + eps * torch.eye(pop.shape[1], device=device)
# ↑ 正則化項 ε I で共分散行列が特異にならないようにする
```
