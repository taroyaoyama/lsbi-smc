# Multimodal Variational Autoencoder (MVAE)

## 1. 設計の動機

本問題では、二つの異なる「モダリティ（データの種類）」を扱う：
- **モダリティ1**：構造パラメータ $\theta \in \mathbb{R}^4$（各階の剛性比）
- **モダリティ2**：周波数応答関数 $x \in \mathbb{R}^{1024}$（FRF データ）

**目標**：$\theta$ と $x$ を同じ潜在空間 $z$ に埋め込み、「どちらからでも同じ latent が得られる」関係を学ぶ。

なぜこれが必要かは [06_latent_likelihood.md](06_latent_likelihood.md) で説明する。

---

## 2. MVAE のアーキテクチャ

```
パラメータ θ → [enc_w: Encoder_w] → (μ_w, σ²_w) → z_w
応答データ x → [enc_x: Encoder  ] → (μ_x, σ²_x) → z_x
                                              ↓
                              z → [dec: Decoder] → (x̂_μ, x̂_σ²)
```

```python
# mvae.py:361-367
class MVAE(nn.Module):
    def __init__(self, z_dim, ch, size, nlabel, depth):
        super().__init__()
        self.enc_x = Encoder(z_dim, ch, size, depth)   # FRFエンコーダ
        self.enc_w = Encoder_w(z_dim, nlabel)           # パラメータエンコーダ
        self.dec = Decoder(z_dim, ch, size, depth)      # 共通デコーダ
```

実際のパラメータ：
- `z_dim = 8`：潜在空間の次元数
- `ch = 1`：チャンネル数
- `size = 1024`：FRF の周波数点数
- `nlabel = 4`：構造パラメータの次元数

FRF テンソルの形状 `(batch, ch, depth, size)` の各軸の意味と、なぜ 1 次元データを `Conv2d` で扱うのかについては [03_ml_prerequisites.md §4.4](03_ml_prerequisites.md) を参照。

---

## 3. 各エンコーダの詳細

### enc_x: FRFデータエンコーダ（畳み込み型）

入力 $x \in \mathbb{R}^{1 \times 1 \times 1024}$（FRF データ）

```
入力: (batch, 1, 1, 1024)
     ↓ first_resblock_enc(1, 2, pooling=(1,2))   → (batch, 2, 1, 512)
     ↓ resblock_enc(2, 4, pooling=(1,2))          → (batch, 4, 1, 256)
     ↓ resblock_enc(4, 8, pooling=(1,2))          → (batch, 8, 1, 128)
     ↓ resblock_enc(8, 16, pooling=(1,2))         → (batch, 16, 1, 64)
     ↓ resblock_enc(16, 32, pooling=(1,2))        → (batch, 32, 1, 32)
     ↓ Flatten                                    → (batch, 1024)
     ↓ FC: 1024 → 512 → 256 → 128
     ↓ mu = Linear(128, 8)                        → (batch, 8)
     ↓ var = Linear(128, 8) → Softplus            → (batch, 8)  [正値]
出力: z, μ_x, σ²_x  各 (batch, 8)
```

**Softplus** を分散に使うのは「分散は必ず正」という制約を満たすため：
$$\text{Softplus}(x) = \log(1 + e^x) > 0$$

### enc_w: パラメータエンコーダ（全結合型）

入力 $\theta \in \mathbb{R}^4$（構造パラメータ）

```
入力: (batch, 4)
     ↓ resblock_enc_small(4, 8)   → (batch, 8)   [LeakyReLU残差ブロック]
     ↓ resblock_enc_small(8, 16)  → (batch, 16)
     ↓ resblock_enc_small(16, 16) → (batch, 16)
     ↓ resblock_enc_small(16, 8)  → (batch, 8)
     ↓ resblock_enc_small(8, 8)   → (batch, 8)
     ↓ mu = Linear(8, 8)          → (batch, 8)
     ↓ var = Linear(8, 8) → Softplus → (batch, 8)
出力: z, μ_w, σ²_w  各 (batch, 8)
```

### dec: 共通デコーダ（畳み込み転置型）

入力 $z \in \mathbb{R}^8$

```
入力: (batch, 8)
     ↓ FC: 8 → 128 → 256 → 512 → 1024
     ↓ reshape: (batch, 32, 1, 32)
     ↓ resblock_dec(32, 16, up=(1,2))   → (batch, 16, 1, 64)
     ↓ resblock_dec(16, 8, up=(1,2))    → (batch, 8, 1, 128)
     ↓ resblock_dec(8, 4, up=(1,2))     → (batch, 4, 1, 256)
     ↓ resblock_dec(4, 2, up=(1,2))     → (batch, 2, 1, 512)
     ↓ decoder_mu:  resblock_dec(2,1,(1,2)) → (batch, 1, 1, 1024)
     ↓ decoder_var: resblock_dec(2,1,(1,2)) → Softplus → (batch, 1, 1, 1024)
出力: x̂_μ, x̂_σ²  各 (batch, 1, 1, 1024)
```

デコーダが出力するのは **確率的な再構成**。出力は点推定ではなく分布 $\mathcal{N}(\hat{x}_\mu, \hat{x}_\sigma^2)$。

---

## 4. MVAE の損失関数

### 4.1 入力の準備

学習時には、各パラメータ $\theta^{(n)}$ に対して 2 種類の応答を用意する：
- $x^{(n)} = h(\theta^{(n)})$：ノイズなし（クリーン）応答
- $\tilde{x}^{(n)} = h(\theta^{(n)}) + \epsilon^{(n)}$：ノイズあり応答

```python
# create_dataset.py:23-25
noise_level = 0.20
y_sim_n = y_sim + noise_level * norm.rvs(size = y_sim.shape)
```

### 4.2 損失の構成

$$\mathcal{L} = \underbrace{L_{\text{KL1}} + L_{\text{KL2}}}_{\text{各エンコーダの正則化}} + \alpha \underbrace{(L_{\text{KL12}} + L_{\text{KL21}})}_{\text{エンコーダ間の整合性}} + \beta_1 \underbrace{L_{\text{rec}}}_{\text{FRF再構成}} + \beta_2 \underbrace{L_{\text{pred}}}_{\text{パラメータからの予測}}$$

```python
# mvae.py:401-409 の loss() メソッド
z1, mu1, var1, z2, mu2, var2 = self.encode(xi, w)  # xi=ノイズなし, w=パラメータ
x_mu1, x_var1, x_mu2, x_var2 = self.decode(z1, z2)

KL1 = gauss_unitgauss_kl(mu1, var1)        # enc_x → N(0,I) へのKL
KL2 = gauss_unitgauss_kl(mu2, var2)        # enc_w → N(0,I) へのKL
KL_x1x2 = gauss_gauss_kl(mu1, var1, mu2, var2)  # enc_x → enc_w へのKL
KL_x2x1 = gauss_gauss_kl(mu2, var2, mu1, var1)  # enc_w → enc_x へのKL
rec_xx = rec_loss_norm4D(xo, x_mu1, x_var1)  # FRFエンコーダで再構成（xo=ノイズあり）
rec_wx = rec_loss_norm4D(xo, x_mu2, x_var2)  # パラメータエンコーダで予測

loss = KL1 + KL2 + alp1*(KL_x1x2 + KL_x2x1) + alp2*rec_xx + alp3*rec_wx
```

### 各損失の役割

| 損失項 | 式 | 役割 |
|--------|-----|------|
| `KL1` | $D_{\text{KL}}(q_{\phi_x}(z\mid x) \,\Vert\, \mathcal{N}(0,I))$ | FRFエンコーダを正則化 |
| `KL2` | $D_{\text{KL}}(q_{\phi_\theta}(z\mid \theta) \,\Vert\, \mathcal{N}(0,I))$ | パラメータエンコーダを正則化 |
| `KL_x1x2` | $D_{\text{KL}}(q_{\phi_x} \,\Vert\, q_{\phi_\theta})$ | 両エンコーダの潜在表現を整合 |
| `KL_x2x1` | $D_{\text{KL}}(q_{\phi_\theta} \,\Vert\, q_{\phi_x})$ | 両エンコーダの潜在表現を整合 |
| `rec_xx` | $-\mathbb{E}_{z\sim q_{\phi_x}}[\log p_\eta(\tilde{x}\mid z)]$ | FRFから再構成できているか |
| `rec_wx` | $-\mathbb{E}_{z\sim q_{\phi_\theta}}[\log p_\eta(\tilde{x}\mid z)]$ | パラメータからFRFを予測できるか |

**`KL_x1x2 + KL_x2x1`**（α=5）が特に重要：これにより $q_{\phi_x}(z|x)$ と $q_{\phi_\theta}(z|\theta)$ が同じ潜在表現を持つように強制される。

### 4.3 再構成損失の計算

#### 4.3.1 出発点：負の対数尤度

再構成損失の出発点は次の式：

$$L_{\text{rec}} = -\mathbb{E}_{z \sim q_\phi(z|x)}[\log p_\eta(\tilde{x}|z)]$$

これは **「潜在変数 $z$ から復元したデコーダの分布 $p_\eta(\tilde{x}|z)$ のもとで、観測 $\tilde{x}$ が得られる確率の対数」を最大化したい** という意図を表す。

- 確率を最大化 ⇔ 負の対数尤度を最小化（最尤推定の基本パターン）
- 期待値は「$z$ を確率的にサンプリングする VAE では、サンプル次第で結果が変わるので平均を取る」という意味

実装では期待値は **1 個の MC サンプル** で近似する（再パラメータ化トリックで `z` を1つだけ取り出す。[03_ml_prerequisites.md](03_ml_prerequisites.md) 参照）。

#### 4.3.2 デコーダ出力の仮定

ここが式の形を決める核心。デコーダは点推定ではなく **「各ピクセル（周波数点）が独立な正規分布に従う」** とモデル化する：

$$p_\eta(\tilde{x}|z) = \prod_{i=1}^{D} \mathcal{N}\!\left(\tilde{x}_i \,\middle|\, \hat{x}_{\mu,i}, \hat{x}_{\sigma,i}^2\right), \qquad D = \text{ch} \times \text{depth} \times \text{size}$$

- **各点が独立**：共分散行列が対角（次元間の相関を仮定しない）
- **平均 $\hat{x}_{\mu,i}$ と分散 $\hat{x}_{\sigma,i}^2$ は両方ともデコーダの出力**：`Decoder` クラスが `decoder_mu` と `decoder_var` という 2 つのヘッドを持つのはこのため

なぜ独立を仮定するのか：
- 完全な共分散行列を出力すると $D \times D = 1024 \times 1024 \approx 10^6$ 個のパラメータが必要 → 非現実的
- 各点独立としても、潜在変数 $z$ を介して間接的に相関を表現できる（条件付き独立性）

#### 4.3.3 ガウス対数尤度の式

1 次元正規分布の確率密度関数：

$$\mathcal{N}(\tilde{x}_i \mid \hat{x}_{\mu,i}, \hat{x}_{\sigma,i}^2) = \frac{1}{\sqrt{2\pi \hat{x}_{\sigma,i}^2}} \exp\!\left(-\frac{(\tilde{x}_i - \hat{x}_{\mu,i})^2}{2 \hat{x}_{\sigma,i}^2}\right)$$

両辺の対数を取ると：

$$\log \mathcal{N}(\tilde{x}_i \mid \hat{x}_{\mu,i}, \hat{x}_{\sigma,i}^2) = -\frac{1}{2}\left[\frac{(\tilde{x}_i - \hat{x}_{\mu,i})^2}{\hat{x}_{\sigma,i}^2} + \log\hat{x}_{\sigma,i}^2 + \log(2\pi)\right]$$

**3 つの項の意味**：

| 項 | 役割 |
|----|------|
| $\dfrac{(\tilde{x}_i - \hat{x}_{\mu,i})^2}{\hat{x}_{\sigma,i}^2}$ | **マハラノビス距離項**：誤差を分散で割って正規化。「分散が大きい点は誤差が大きくても許す、分散が小さい点は厳しく咎める」 |
| $\log\hat{x}_{\sigma,i}^2$ | **分散ペナルティ項**：これがないとデコーダは $\hat{\sigma}^2 \to \infty$ にすることで誤差を見かけ上ゼロにできてしまう。分散を大きくするとこの項が増えるのでブレーキになる |
| $\log(2\pi)$ | 正規化定数（学習に直接の影響なし、定数オフセット） |

#### 4.3.4 独立性 → 和への分解

独立な分布の積の対数は、対数の和になる：

$$\log p_\eta(\tilde{x}|z) = \log \prod_{i=1}^{D} \mathcal{N}_i = \sum_{i=1}^{D} \log \mathcal{N}_i$$

各項を代入して整理すると、最終的な負の対数尤度：

$$\boxed{-\log p_\eta(\tilde{x}|z) = \frac{1}{2}\sum_i \left[\frac{(\tilde{x}_i - \hat{x}_{\mu,i})^2}{\hat{x}_{\sigma,i}^2} + \log\hat{x}_{\sigma,i}^2 + \log(2\pi)\right]}$$

これがコードと完全一致する。

#### 4.3.5 コードとの対応

```python
# mvae.py
def rec_loss_norm_4d(x, mean, var):
    return -torch.mean(
        torch.sum(
            -0.5 * ((x - mean) ** 2 / var
                    + torch.log(var)
                    + torch.log(torch.tensor(2 * torch.pi))),
            dim=(1, 2, 3),
        )
    )
```

| 数式 | コード |
|------|--------|
| $(\tilde{x}_i - \hat{x}_{\mu,i})^2 / \hat{x}_{\sigma,i}^2$ | `(x - mean) ** 2 / var` |
| $\log\hat{x}_{\sigma,i}^2$ | `torch.log(var)` |
| $\log(2\pi)$ | `torch.log(torch.tensor(2 * torch.pi))` |
| $\frac{1}{2}\sum_i [\cdots]$ | `-0.5 * (...)` の和（外側の `-torch.mean` で符号反転され NLL になる） |
| バッチ平均 | 外側の `torch.mean` |
| 全要素和（次元 1〜3） | `torch.sum(..., dim=(1, 2, 3))` |

`dim=(1, 2, 3)` は「チャンネル × 深さ × 周波数点」のすべての軸で和を取り、サンプルあたりの NLL を出している。詳しくは [code/mvae.md](code/mvae.md) の「`torch.sum(..., dim=...).mean()` の意味」を参照。

#### 4.3.6 直感：MSE との関係

もし分散を $\hat{x}_{\sigma,i}^2 = 1$（固定定数）と仮定すると、上の式は：

$$-\log p \propto \sum_i (\tilde{x}_i - \hat{x}_{\mu,i})^2 + \text{const}$$

これは **平均二乗誤差（MSE）と一致する**。つまり：

> **MVAE の再構成損失は「分散も学習する一般化された MSE」**

分散を学習することで「自信のある点は厳しく、自信のない点は緩く」という適応的な重み付けが可能になり、観測ノイズの大きさをデータから推定できる。

---

## 5. 学習の流れ

```
for epoch in range(epochs):
    for (θ, x_clean, x_noisy) in train_loader:
        loss = model.loss(x_noisy, x_clean, θ)
        # x_noisy → デコーダへの目標
        # x_clean → enc_x の入力
        # θ       → enc_w の入力
        loss.backward()
        optimizer.step()
```

**なぜクリーンとノイズを分けるのか**：
- エンコーダにはクリーンな信号を与えて正確な潜在表現を学習
- デコーダの目標にはノイズあり信号を使って「実測のような不確かさ」を再現できるよう学習

→ 実測データ（ノイズあり）をエンコードしたときも適切に機能する

---

## 6. 潜在空間の可視化（論文 Figure 4）

学習後、$\theta$ を `enc_w` でエンコードした潜在平均 $\mu_w \in \mathbb{R}^8$ に PCA を適用すると：
- 潜在空間が固有振動数と対応した構造を持つことがわかる
- 等価解（異なる $\theta$ だが同じ応答）は潜在空間上で観測値に近い位置に配置される

→ **MVAE が $\theta$ と $x$ の共通的な物理的特徴を学習できている証拠**
