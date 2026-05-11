# mvae.py — Multimodal Variational Autoencoder

**ファイルパス**: `src/lsbi_smc/example_shear4dof/mvae.py`  
**関連学術ドキュメント**: [05_mvae.md](../05_mvae.md), [03_ml_prerequisites.md](../03_ml_prerequisites.md)

---

## 概要

このモジュールは LSBI-SMC の中核であるニューラルネットワーク全体を実装しています。

- **残差ブロック**（エンコーダ用・デコーダ用）
- **損失関数**（KL ダイバージェンス・再構成損失）
- **`Encoder`**：FRF データ → 潜在空間（畳み込みResNet）
- **`EncoderW`**：構造パラメータ → 潜在空間（全結合ResNet）
- **`Decoder`**：潜在空間 → FRF（転置畳み込みResNet）
- **`MVAE`**：二つのエンコーダと一つの共通デコーダを統合
- **`VAE`**：参考用の単一エンコーダ VAE
- **`plot_frf`**：学習経過の可視化

---

## ユーティリティ関数

### `reparameterization`

```python
def reparameterization(mean: Tensor, var: Tensor, device: torch.device) -> Tensor:
    epsilon = torch.randn_like(mean)
    return mean + torch.sqrt(var) * epsilon
```

VAE の**再パラメータ化トリック**。確率的なサンプリングを微分可能にする。

- $z = \mu + \sqrt{\sigma^2} \cdot \epsilon$, $\epsilon \sim \mathcal{N}(0, I)$
- `var` は分散（$\sigma^2$）であることに注意（標準偏差ではない）

詳細: [03_ml_prerequisites.md](../03_ml_prerequisites.md)（VAE の再パラメータ化）

---

## 損失関数

### `gauss_gauss_kl` — ガウス間 KL ダイバージェンス

```python
def gauss_gauss_kl(mean1, var1, mean2, var2) -> Tensor:
```

$$D_{\text{KL}}(\mathcal{N}(\mu_1, \sigma_1^2) \| \mathcal{N}(\mu_2, \sigma_2^2)) = \frac{1}{2}\sum_i \left[\log\frac{\sigma_{2,i}^2}{\sigma_{1,i}^2} + \frac{\sigma_{1,i}^2 + (\mu_{1,i} - \mu_{2,i})^2}{\sigma_{2,i}^2} - 1\right]$$

MVAE では `enc_x` と `enc_w` の潜在表現を揃えるために使用（双方向 KL）。

### `gauss_unitgauss_kl` — 標準正規分布への KL

```python
def gauss_unitgauss_kl(mean, var) -> Tensor:
```

$$D_{\text{KL}}(\mathcal{N}(\mu, \sigma^2) \| \mathcal{N}(0, I)) = \frac{1}{2}\sum_i \left[-1 - \log\sigma_i^2 + \mu_i^2 + \sigma_i^2\right]$$

エンコーダの正則化項（標準 VAE の KL 損失と同じ）。`eps=1e-8` で数値安定化。

### `rec_loss_norm_4d` / `rec_loss_norm_2d` — 再構成損失

```python
def rec_loss_norm_4d(x, mean, var) -> Tensor:
    return -torch.mean(
        torch.sum(-0.5 * ((x - mean)**2 / var + log(var) + log(2π)), dim=(1,2,3))
    )
```

出力が $\mathcal{N}(\hat{x}_\mu, \hat{x}_{\sigma^2})$ と仮定したときの負の対数尤度（`4d`: 4次元テンソル用）。

$$-\log p(x|z) = \frac{1}{2}\sum_i \left[\frac{(x_i - \hat{x}_{\mu,i})^2}{\hat{x}_{\sigma,i}^2} + \log\hat{x}_{\sigma,i}^2 + \log 2\pi\right]$$

---

### `torch.sum(..., dim=...).mean()` の意味

上記すべての損失関数（`gauss_unitgauss_kl`, `gauss_gauss_kl`, `rec_loss_norm_4d`, `rec_loss_norm_2d`）に共通して、次のパターンが現れる：

```python
torch.sum(_kl, dim=1).mean()           # KL 系（潜在次元）
torch.sum(..., dim=(1, 2, 3)).mean()   # 再構成系（チャンネル × 深さ × 周波数）
```

これは「**特徴量次元については和、バッチ次元については平均**」というルールに従っている。

#### 形状の追跡（`gauss_unitgauss_kl` の例）

エンコーダ出力 `mean`, `var` の形状は `(batch, z_dim)` = `(batch, 8)`。

```python
_kl = -0.5 * (1 + torch.log(var + eps) - mean**2 - var)
# _kl.shape = (batch, 8)  ← 要素ごと演算なので形状は不変
```

`_kl[b, i]` は **バッチ `b` の潜在次元 `i` の KL 寄与分** を表す。

#### `torch.sum(_kl, dim=1)` — 次元方向の和

`dim=1` は「1番目の軸（潜在次元軸）に沿って和をとる」指定。`(batch, 8)` → `(batch,)` に縮約される。

```
_kl              shape: (batch, 8)
   ┌──────────────────────────────────┐
   │ kl₀ kl₁ kl₂ kl₃ kl₄ kl₅ kl₆ kl₇ │  ← サンプル0
   │ kl₀ kl₁ kl₂ kl₃ kl₄ kl₅ kl₆ kl₇ │  ← サンプル1
   │ ...                              │
   └──────────────────────────────────┘
            ↓ sum(dim=1)
   ┌────┐
   │ Σᵢ │  shape: (batch,)
   │ Σᵢ │
   │ ...│
   └────┘
```

**なぜ和なのか**：多変量ガウス（対角共分散）の KL は各次元の KL の和になるため。

$$D_{KL}(\mathcal{N}(\boldsymbol{\mu}, \mathrm{diag}(\boldsymbol{\sigma}^2)) \| \mathcal{N}(0, I)) = \sum_{i=1}^{z_{\dim}} D_{KL}(\mathcal{N}(\mu_i, \sigma_i^2) \| \mathcal{N}(0,1))$$

再構成損失の場合も、ピクセル（周波数点）独立を仮定した対角ガウスの対数尤度なので、`dim=(1,2,3)` で全要素の対数尤度を足し合わせる。

#### `.mean()` — バッチ方向の平均

`(batch,)` のテンソルに `.mean()` を適用するとスカラー（形状 `()`）になる。

```
(batch,) ─── .mean() ───→ ()   ← スカラー
```

**なぜ平均なのか**：損失関数を「1サンプルあたりの平均損失」として表すのが慣例。SGD の勾配ノルムがバッチサイズに依存しなくなり、学習率のチューニングがしやすくなる。

$$\mathcal{L}_{\text{KL}} = \frac{1}{N_{\text{batch}}} \sum_{n=1}^{N_{\text{batch}}} \left[\sum_{i=1}^{z_{\dim}} \mathrm{KL}_i^{(n)}\right]$$

#### まとめ

| 軸 | 操作 | 理由 |
|----|------|------|
| 特徴量次元（潜在 / FRF 要素） | **`sum`** | 確率の独立同時分布 → 対数尤度・KL は要素和になる |
| バッチ次元 | **`mean`** | 学習率がバッチサイズに依存しないように正規化 |

#### 数値例（`z_dim=2`, `batch=3` の場合）

```
_kl = [[0.3, 0.5],   # サンプル0: 次元0 で 0.3, 次元1 で 0.5
       [0.1, 0.2],   # サンプル1
       [0.4, 0.6]]   # サンプル2
```

`torch.sum(_kl, dim=1)`:
```
[0.3+0.5, 0.1+0.2, 0.4+0.6] = [0.8, 0.3, 1.0]   # 各サンプルの KL 値
```

`.mean()`:
```
(0.8 + 0.3 + 1.0) / 3 = 0.7   # バッチ平均 KL → スカラー
```

このスカラーがそのまま `loss` の構成要素として返される。

---

## 残差ブロック

### `FirstResblockEnc` / `ResblockEnc` — エンコーダ用残差ブロック（畳み込み）

```
入力 x
  ├── メインパス: [Conv(1×3)] → [ReLU] → [Conv(3×3)] → [Dropout(0.5)] → [AvgPool]
  └── バイパス:   [Conv(1×1)] → [AvgPool]
        ↓
メインパス + バイパス（残差接続）
```

- `FirstResblockEnc`：最初の層（ReLU が先頭にない）
- `ResblockEnc`：中間層（`ReLU` が先頭にある）
- `pooling_size=(1, 2)` で時系列方向のみダウンサンプリング（1024→512→256→...→32）
- **スペクトル正規化**（`spectral_norm`）：訓練の安定化のために重みを正規化

### `ResblockEncSmall` — エンコーダ用残差ブロック（全結合）

```
入力 x
  ├── メインパス: [Linear(in→out)] → [LeakyReLU(0.2)] → [Linear(out→out)] → [Dropout(0.2)]
  └── バイパス:   [Linear(in→out)]（次元が変わる場合のみ; 同じならIdentity）
        ↓
LeakyReLU(メインパス + バイパス)
```

`EncoderW`（パラメータエンコーダ）の各層に使用。小さい入力次元（4次元）に適した全結合型残差ブロック。

### `ResblockDec` — デコーダ用残差ブロック（アップサンプリング）

```
入力 x
  ├── メインパス: [BatchNorm] → [ReLU] → [Dropout(0.5)] → [Upsample(×2)] → [Conv(1×3)] → [BatchNorm] → [ReLU] → [Conv(3×3)]
  └── バイパス:   [Upsample(×2)] → [Conv(1×1)]
        ↓
メインパス + バイパス
```

`up_sample=(1, 2)` でバイリニア補間による時系列方向のアップサンプリング（32→64→128→...→1024）。

---

## `Encoder` クラス（FRF エンコーダ）

```python
class Encoder(nn.Module):
    def __init__(self, z_dim: int, ch: int, size: int, depth: int)
```

FRF データ `(batch, ch, depth, size)` を潜在表現 `(batch, z_dim)` に変換。

> 4 軸の意味（なぜ `Conv2d` を使うか、`depth` 軸の役割など）は [03_ml_prerequisites.md §4.4](../03_ml_prerequisites.md) を参照。型エイリアスは [`src/lsbi_smc/shapes.py`](../../src/lsbi_smc/shapes.py) に定義。

### アーキテクチャ

```
入力: (batch, 1, 1, 1024)
  ↓ FirstResblockEnc(1, 2, pool=(1,2))      → (batch, 2, 1, 512)
  ↓ ResblockEnc(2, 4, pool=(1,2))           → (batch, 4, 1, 256)
  ↓ ResblockEnc(4, 8, pool=(1,2))           → (batch, 8, 1, 128)
  ↓ ResblockEnc(8, 16, pool=(1,2))          → (batch, 16, 1, 64)
  ↓ ResblockEnc(16, 32, pool=(1,2))         → (batch, 32, 1, 32)
  ↓ LeakyReLU → Flatten                     → (batch, 1024)
  ↓ Linear(1024→512) → BN → LeakyReLU
  ↓ Linear(512→256)  → BN → LeakyReLU
  ↓ Linear(256→128)  → BN → LeakyReLU
  ↓ mu:  Linear(128→8)                      → (batch, 8)   [平均]
  ↓ var: Linear(128→8) → Softplus           → (batch, 8)   [分散, >0]
出力: z, mu, var  各 (batch, 8)
```

`forward` メソッドで `reparameterization` により `z` をサンプリング。

---

## `EncoderW` クラス（パラメータエンコーダ）

```python
class EncoderW(nn.Module):
    def __init__(self, z_dim: int, n_label: int)
```

構造パラメータ `(batch, ndof)` を潜在表現 `(batch, z_dim)` に変換。

### アーキテクチャ

```
入力: (batch, 4)
  ↓ ResblockEncSmall(4, 8)    → (batch, 8)
  ↓ ResblockEncSmall(8, 16)   → (batch, 16)
  ↓ ResblockEncSmall(16, 16)  → (batch, 16)
  ↓ ResblockEncSmall(16, 8)   → (batch, 8)
  ↓ ResblockEncSmall(8, 8)    → (batch, 8)
  ↓ mu:  Linear(8→8)          → (batch, 8)
  ↓ var: Linear(8→8) → Softplus → (batch, 8)
出力: z, mu, var  各 (batch, 8)
```

入力が4次元と小さいため、畳み込みではなく全結合 ResNet を使用。

---

## `Decoder` クラス（共通デコーダ）

```python
class Decoder(nn.Module):
    def __init__(self, z_dim: int, ch: int, size: int, depth: int)
```

潜在表現 `(batch, z_dim)` から FRF の分布 `(mu, var)` を生成。

### アーキテクチャ

```
入力: (batch, 8)
  ↓ Linear(8→128) → BN → LeakyReLU
  ↓ Linear(128→256) → BN → LeakyReLU
  ↓ Linear(256→512) → BN → LeakyReLU
  ↓ Linear(512→1024) → BN → LeakyReLU
  ↓ reshape: (batch, 32, 1, 32)
  ↓ ResblockDec(32, 16, up=(1,2))  → (batch, 16, 1, 64)
  ↓ ResblockDec(16, 8, up=(1,2))   → (batch, 8, 1, 128)
  ↓ ResblockDec(8, 4, up=(1,2))    → (batch, 4, 1, 256)
  ↓ ResblockDec(4, 2, up=(1,2))    → (batch, 2, 1, 512)
  ↓ decoder_mu:  ResblockDec(2,1,(1,2))           → (batch, 1, 1, 1024)
  ↓ decoder_var: ResblockDec(2,1,(1,2)) + Softplus → (batch, 1, 1, 1024)
出力: mu, var  各 (batch, 1, 1, 1024)
```

`enc_x` と `enc_w` の両方のサンプル `z` を受け取り、同一の重みでデコードする（**共有デコーダ**）。

---

## `MVAE` クラス

```python
class MVAE(nn.Module):
    def __init__(self, z_dim, ch, size, nlabel, depth)
```

ベンチマーク設定: `MVAE(z_dim=8, ch=1, size=1024, nlabel=4, depth=1)`

### `encode` メソッド

```python
def encode(self, x, w) -> (z1, mu1, var1, z2, mu2, var2):
    z1, mu1, var1 = self.enc_x(x)   # FRF → 潜在
    z2, mu2, var2 = self.enc_w(w)   # パラメータ → 潜在
```

### `decode` メソッド

```python
def decode(self, z1, z2) -> (x_mu1, x_var1, x_mu2, x_var2):
    x_mu1, x_var1 = self.dec(z1)   # FRF latent → FRF再構成
    x_mu2, x_var2 = self.dec(z2)   # θ latent → FRF予測
```

同一デコーダで2つの latent をそれぞれデコード。

### `loss` メソッド（学習時に使用）

```python
def loss(self, xo, xi, w, alp1=1.0, alp2=10.0, alp3=10.0):
    # xo: ノイズあり FRF (デコーダの目標)
    # xi: ノイズなし FRF (enc_x への入力)
    # w:  構造パラメータ (enc_w への入力)
```

損失の構成：

$$\mathcal{L} = \underbrace{L_{\text{KL1}} + L_{\text{KL2}}}_{\text{正則化}} + \alpha_1\underbrace{(L_{\text{KL12}} + L_{\text{KL21}})}_{\text{潜在空間整合}} + \alpha_2 \underbrace{L_{\text{rec}}}_{\text{再構成}} + \alpha_3 \underbrace{L_{\text{pred}}}_{\text{予測}}$$

`train.py` では `alp1=5.0`（潜在空間整合を強く強制）を使用。

| 損失項 | 計算対象 | 意味 |
|--------|---------|------|
| `kl1` | `gauss_unitgauss_kl(mu1, var1)` | `enc_x` → $\mathcal{N}(0,I)$ への KL |
| `kl2` | `gauss_unitgauss_kl(mu2, var2)` | `enc_w` → $\mathcal{N}(0,I)$ への KL |
| `kl_x1x2` | `gauss_gauss_kl(mu1, var1, mu2, var2)` | `enc_x` → `enc_w` への KL |
| `kl_x2x1` | `gauss_gauss_kl(mu2, var2, mu1, var1)` | `enc_w` → `enc_x` への KL |
| `rec_xx` | `rec_loss_norm_4d(xo, x_mu1, x_var1)` | FRF潜在から再構成（目標: ノイズあり） |
| `rec_wx` | `rec_loss_norm_4d(xo, x_mu2, x_var2)` | θ潜在からFRF予測（目標: ノイズあり） |

詳細は [05_mvae.md](../05_mvae.md) を参照。

### `forward` メソッド（推論時に使用）

```python
# return_loss=False (デフォルト): 再構成結果を返す
x_mu1, x_var1, x_mu2, x_var2 = model(y, x, return_loss=False)

# return_loss=True: 損失を返す (loss() と同等)
loss, kl_sum, rec_sum = model(y, x, return_loss=True)
```

---

## `VAE` クラス（参考用）

単一の `Encoder` と `Decoder` による標準 VAE。`MVAE` への理解の足がかりとして実装されている。実際のパイプラインでは使用しない。

---

## `plot_frf` 関数（可視化）

学習経過のモニタリング用。DataLoader からバッチを1つ取得し、再構成 FRF と予測 FRF を可視化する。

- **赤**: ノイズあり目標（`yn`）
- **オレンジ**: FRF エンコーダによる再構成（`enc_x` → `dec`）
- **青**: パラメータエンコーダによる予測（`enc_w` → `dec`）
- 破線: $\mu \pm \sigma$（不確かさの範囲）

---

## 初期化の方針

Xavier Uniform 初期化をすべての `Conv2d` と `Linear` 層に適用。バイアスはゼロ初期化。スペクトル正規化との組み合わせで訓練初期の安定性を確保。
