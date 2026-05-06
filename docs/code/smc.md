# smc.py — Sequential Monte Carlo サンプラー

**ファイルパス**: `src/lsbi_smc/smc/smc.py`  
**関連学術ドキュメント**: [06_smc.md](../06_smc.md), [01_math_prerequisites.md](../01_math_prerequisites.md)

---

## 概要

SMC（Sequential Monte Carlo）の本体実装です。事前分布から徐々に事後分布へアニーリングしながら、粒子群を事後分布の代表サンプルに収束させます。

このファイルには以下が含まれます：
- プロトコル定義（インタフェース）
- `Particles` クラス（粒子の状態管理）
- ESS 計算とβ探索の関数群
- `SMC` クラス（サンプラー本体）

---

## プロトコル定義

```python
class LikelihoodProtocol(Protocol):
    device: torch.device
    def __call__(self, theta: Tensor) -> Tensor: ...

class PriorProtocol(Protocol):
    names: list[str]
    def sample(self, n=1) -> Tensor: ...
    def lp(self, values=None) -> Tensor: ...
    def check_support(self, values) -> Tensor: ...

class ProposalProtocol(Protocol):
    def __call__(self, particles: Particles) -> Tensor: ...

class KernelProtocol(Protocol):
    def __call__(self, particles, q, prior, likelihood) -> tuple[...]: ...
```

Python の `Protocol` を使ったダックタイピング設計。実装クラスの継承関係なしにインタフェースを定義できる。

---

## `Particles` クラス

SMC の粒子群の状態を管理するデータコンテナ。

```python
class Particles:
    pop: Tensor           # 粒子位置  shape: (n, dim)
    dq: float             # 現在の重み付けに使ったΔβ
    lp: Tensor | None     # 対数尤度  shape: (n,)
    weights: Tensor | None # 正規化された重み  shape: (n,)
    dim: int              # パラメータ次元数
    size: int             # 粒子数 N
```

### `eval_weights` メソッド

```python
def eval_weights(self, dq=None):
    z = self.dq * self.lp
    z = nan_to_num(z, neginf=-1e30, posinf=1e30)
    z = z - z.max()          # 数値安定化（logsum-exp trick）
    w = exp(z)
    w = nan_to_num(w, nan=0.0, ...)
    s = w.sum()
    # 重みが全て無効な場合は一様分布にフォールバック
    w = full_like(w, 1/N) if not isfinite(s) or s <= 0 else w / s
    w = clamp(w, min=0)
    self.weights = w
```

重み付けの式：$w_n \propto \hat{L}(\theta^{(n)})^{\Delta\beta}$（対数スケール: $\Delta\beta \cdot \log\hat{L}(\theta^{(n)})$）

**logsum-exp トリック**: `z -= z.max()` で値を引いても exp の比率が変わらないため、数値的なオーバーフロー・アンダーフローを防ぐ。

### `resample` メソッド

```python
def resample(self, dq=None):
    self.eval_weights()
    idx = torch.multinomial(self.weights, self.size, replacement=True)
    self.pop = self.pop[idx]    # 高重みの粒子を複製、低重みは消える
    self.lp = self.lp[idx]
```

**多項サンプリング（Multinomial Sampling）**: 重みに比例した確率で粒子をリサンプリング。同じ粒子が複製されることがあるが、後の MCMC ムーブで多様性が回復する。

### `replace` メソッド

```python
def replace(self, idx: Tensor, pop_new: Tensor, lp_new: Tensor):
    self.pop[idx] = pop_new[idx]   # 採択された粒子のみ更新
    self.lp[idx] = lp_new[idx]
```

MCMC 採択された粒子のみ位置と尤度を更新。棄却された粒子は現在位置に留まる。

---

## ESS（有効サンプルサイズ）関数群

### `ess` 関数

```python
def ess(weights_np: NDArray) -> float:
    s1 = weights_np.sum()          # Σw_n
    s2 = (weights_np**2).sum()     # Σw_n²
    return (s1 * s1) / s2          # (Σw_n)² / Σw_n²
```

$$\text{ESS} = \frac{\left(\sum_n w_n\right)^2}{\sum_n w_n^2}$$

- 全粒子が等重み → ESS = N（最大値）
- 1粒子に全重みが集中 → ESS = 1（最小値）

### `_ess_from_lp` 関数

```python
def _ess_from_lp(delta_q: float, lp_np: NDArray) -> float:
    z = delta_q * lp_np
    z -= z.max()           # 数値安定化
    w = np.exp(z)
    return ess(w)
```

対数尤度から直接 ESS を計算（β を試行する際に使用）。

### `_find_next_q` 関数

```python
def _find_next_q(q_prev, q_tar, lp_np, ess_tar, tol=1e-6, maxit=50) -> float:
    lo, hi = q_prev, q_tar
    if _ess_from_lp(hi - q_prev, lp_np) >= ess_tar:
        return hi       # 一気にターゲットまで到達可能
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

**二分探索**で目標 ESS を達成する最大の β を探す。

- `ess_from_lp(hi - q_prev, lp_np) >= ess_tar` → 一気に終点まで進める
- そうでなければ二分探索で $\Delta\beta$ を絞り込む

---

## `SMC` クラス

### コンストラクタ

```python
class SMC:
    def __init__(
        self,
        pop_size: int,            # 粒子数 N
        likelihood: LikelihoodProtocol,
        prior: PriorProtocol,
        kernel: KernelProtocol,
        q_tar: float = 1.0,       # 目標逆温度（通常1.0）
    ):
        self.particles = Particles(prior.sample(pop_size))
        self.particles.lp = likelihood(self.particles.pop)
        self.pops = [pop_ini]     # 各ステップの粒子を履歴として保持
        self.q = [0]              # β の履歴
```

初期化時に:
1. 事前分布から粒子をサンプリング（$\beta_0 = 0$）
2. 全粒子の尤度を計算（初期尤度）
3. 粒子履歴と $\beta$ 履歴を初期化

### `run` メソッド（メインループ）

```python
def run(
    self,
    ess_tar_ratio: float = 0.8,   # 目標ESS割合
    t_max: int = 100,             # 最大ステップ数
    mcmc_iter: int = 1,           # MCMCムーブの繰り返し数
):
```

```
ループ（β < 1.0 かつ t < t_max）:
  1. β を決定: _find_next_q(β_prev, 1.0, lp, ESS_tar=0.8N)
  2. リサンプリング: particles.resample(Δβ)
  3. 重みをリセット: 一様重みに戻す（リサンプリング後は等重み）
  4. MCMC ムーブ（mcmc_iter 回）:
     a. 提案: pop_new = kernel(particles, β, ...)
     b. 採択: particles.replace(accept, pop_new, lp_new)
  5. 粒子をpops に記録
  6. 進行状況を表示
```

SMC の各ステップの図式：

```
β = 0 (事前分布)
  ↓ Δβ₁ → リサンプリング → MCMC×10
β = β₁
  ↓ Δβ₂ → リサンプリング → MCMC×10
β = β₂
  ↓ ...
β = 1.0 (事後分布) ← pops[-1] が事後サンプル
```

### `summary` メソッド

```python
def summary(self) -> pd.DataFrame:
    pop = self.pops[-1].detach().cpu().numpy()
    result = pd.DataFrame({
        'name': self.prior.names,
        'mean': np.mean(pop, axis=0),
        'sd':   np.std(pop, axis=0),
        'q05':  np.percentile(pop, 5, axis=0),
        'q25':  np.percentile(pop, 25, axis=0),
        'q50':  np.percentile(pop, 50, axis=0),
        'q75':  np.percentile(pop, 75, axis=0),
        'q95':  np.percentile(pop, 95, axis=0),
    })
    return result
```

最終粒子群（事後分布サンプル）の統計量を表示。`run()` の終了後に自動で呼ばれる。

---

## 数値安定性への配慮まとめ

| 箇所 | 処理 | 理由 |
|------|------|------|
| `eval_weights` | `nan_to_num(z, neginf=-1e30)` | 対数尤度が -inf のとき exp(-inf)=0 が正常 |
| `eval_weights` | `z -= z.max()` | オーバーフロー防止 |
| `eval_weights` | 一様重みへのフォールバック | 重み崩壊（全粒子の重みがゼロ）への対処 |
| `RWMetropolisKernel` | `nan_to_num(log_rat, ...)` | 採択比計算での inf 防止 |

---

## `assign_pop_ini` メソッド

```python
def assign_pop_ini(self, pop: Tensor):
    self.particles = Particles(pop)
    self.particles.lp = self.likelihood(self.particles.pop)
    self.pops = [pop]
```

初期粒子を手動で設定する場合（デフォルト: 事前分布からのサンプリング）。
