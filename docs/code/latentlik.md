# latentlik.py — 潜在空間ベース対数尤度

**ファイルパス**: `src/lsbi_smc/likelihood/latentlik.py`  
**関連学術ドキュメント**: [06_latent_likelihood.md](../06_latent_likelihood.md)

---

## 概要

学習済み MVAE の2つのエンコーダを使って、**パラメータ θ と観測 FRF の対数尤度** を潜在空間上で計算します。

高次元の FRF 空間での直接尤度計算を回避し、8次元の潜在空間での解析的な積分計算に置き換えることが核心です。

---

## 数学的背景

詳細な導出は [06_latent_likelihood.md](../06_latent_likelihood.md) を参照。要約すると：

$$\hat{L}(\theta; x_{\text{obs}}) = \int_Z \frac{q_{\phi_x}(z | x_{\text{obs}}) \cdot q_{\phi_\theta}(z | \theta)}{p(z)} dz$$

- $q_{\phi_x}(z|x_{\text{obs}}) = \mathcal{N}(z|\mu_1, \sigma_1^2)$：観測 FRF の潜在エンコーディング（`enc_x`）
- $q_{\phi_\theta}(z|\theta) = \mathcal{N}(z|\mu_2, \sigma_2^2)$：パラメータの潜在エンコーディング（`enc_w`）
- $p(z) = \mathcal{N}(z|0, 1)$：潜在変数の事前分布

3つのガウス分布の積・商の積分が解析的に計算できることを利用。

---

## `latent_space_loglik` 関数

```python
def latent_space_loglik(
    mu_obs: Tensor,      # 観測 FRF の潜在平均    shape: (1, z_dim)
    vr_obs: Tensor,      # 観測 FRF の潜在分散    shape: (1, z_dim)
    mu_sim: Tensor,      # シミュレーション θ の潜在平均  shape: (n, z_dim)
    vr_sim: Tensor,      # シミュレーション θ の潜在分散  shape: (n, z_dim)
    alp: float = 0.0,    # 分散スケーリング係数
    tau: float = 1e-4,   # 最小分散（数値安定性）
    mu_pri: float = 0.0, # 事前分布の平均
    vr_pri: float = 1.0, # 事前分布の分散
    eps: float = 0.0,    # 分散のクランプ下限
) -> Tensor:             # 対数尤度  shape: (n,)
```

### アルゴリズム詳細

#### 入力の準備

```python
vr1 = clamp(vr_obs, min=eps)
vr2 = clamp((1 + alp) * vr_sim + tau, min=eps)
vr3 = vr_pri = 1.0                 # N(0,1) の分散
mu1, mu2, mu3 = mu_obs, mu_sim, 0.0
```

`vr2 = (1 + alp) * vr_sim + tau`：シミュレーターの不確かさを調整する拡大項。

#### 係数の計算（ガウス積分の準備）

```python
a = (1 / (2*vr1)) + (1 / (2*vr2)) - (1 / (2*vr3))
b = -(mu1 / vr1) - (mu2 / vr2) + (mu3 / vr3)
c = (mu1**2 / (2*vr1)) + (mu2**2 / (2*vr2)) - (mu3**2 / (2*vr3))
d = sqrt(vr3 / (2*pi*vr1*vr2))
```

被積分関数の指数部分を `−az² + bz − c` と表したときの係数（各潜在次元 $i$ ごとに独立）。

#### ガウス積分の解析解

$$\int_{-\infty}^{\infty} e^{-az^2 + bz - c} dz = e^{b^2/(4a) - c} \cdot \sqrt{\pi/a}$$

対数を取ると：

```python
lp = log(d) + (b**2 - 4*a*c) / (4*a) + 0.5 * log(pi/a)
```

#### 数値安定化

```python
lp = torch.nan_to_num(lp, neginf=-1e30, posinf=1e30)
```

`a` がゼロや負になる極端なケース（`vr_obs` と `vr_sim` と `vr_pri` の組み合わせ）で NaN/inf が発生するのを防ぐ。

#### 潜在次元の合計

```python
return torch.sum(lp, dim=1)   # 各次元の対数尤度を足し合わせる
```

各潜在次元は独立と仮定しているため、対数尤度は和になる。

---

## `MVAEBasedLogLikelihood` クラス

```python
class MVAEBasedLogLikelihood:
    def __init__(self, enc_w, enc_x, obs, device):
```

### コンストラクタ（観測データの事前エンコード）

```python
def __init__(self, enc_w, enc_x, obs, device):
    self.enc_w = enc_w.to(device)
    self.enc_x = enc_x.to(device)
    self.obs = obs.to(device)

    # 推論中は重みを更新しない
    for p in self.enc_w.parameters():
        p.requires_grad_(False)
    for p in self.enc_x.parameters():
        p.requires_grad_(False)
    self.enc_w.eval()
    self.enc_x.eval()

    # 観測データを一度だけエンコード（重要な最適化）
    with torch.no_grad():
        _, mu_obs, vr_obs = self.enc_x(self.obs)
    self.mu_obs = mu_obs   # shape: (1, z_dim)
    self.vr_obs = vr_obs   # shape: (1, z_dim)
```

**重要な最適化**: 観測データは変化しないため、`__init__` 時に1回だけエンコード。SMC の何千回もの尤度評価で再エンコードのコストを節約する。

### `__call__` メソッド（実際の尤度計算）

```python
@torch.no_grad()
def __call__(self, theta: Tensor, alp=1.0, tau=0.0) -> Tensor:
    theta = theta.to(self.device)
    _, mu_sim, vr_sim = self.enc_w(theta)   # θ → 潜在エンコーディング
    return latent_space_loglik(
        self.mu_obs, self.vr_obs,
        mu_sim, vr_sim,
        alp=alp, tau=tau
    )
```

`@torch.no_grad()` デコレータで勾配計算を無効化（推論専用）。

---

## データフロー

```
観測 FRF (y_obs_tc)
   ↓ __init__ で1回だけ実行
   enc_x(y_obs_tc) → (_, mu_obs, vr_obs)
                           │
                           │ (固定)
                           ↓
粒子群 (theta) ────── __call__ で毎回実行
   ↓
   enc_w(theta) → (_, mu_sim, vr_sim)
   ↓
   latent_space_loglik(mu_obs, vr_obs, mu_sim, vr_sim)
   ↓
   対数尤度 shape: (n_particles,)
```

---

## パラメータ `alp` と `tau` の役割

`inference.py` では `alp=1.0, tau=0.0` で呼ばれる：

| パラメータ | 値 | 効果 |
|-----------|-----|------|
| `alp=1.0` | 1.0 | `vr_sim` を2倍にスケール → モデル誤差を吸収する余裕を持たせる |
| `tau=0.0` | 0.0 | 最小分散の付加なし（クランプで十分な数値安定性） |

詳細は [06_latent_likelihood.md](../06_latent_likelihood.md) を参照。
