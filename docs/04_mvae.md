# Multimodal Variational Autoencoder (MVAE)

## 1. 設計の動機

本問題では、二つの異なる「モダリティ（データの種類）」を扱う：
- **モダリティ1**：構造パラメータ $\theta \in \mathbb{R}^4$（各階の剛性比）
- **モダリティ2**：周波数応答関数 $x \in \mathbb{R}^{1024}$（FRF データ）

**目標**：$\theta$ と $x$ を同じ潜在空間 $z$ に埋め込み、「どちらからでも同じ latent が得られる」関係を学ぶ。

なぜこれが必要かは [05_latent_likelihood.md](05_latent_likelihood.md) で説明する。

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
| `KL1` | $D_{\text{KL}}(q_{\phi_x}(z|x) \| \mathcal{N}(0,I))$ | FRFエンコーダを正則化 |
| `KL2` | $D_{\text{KL}}(q_{\phi_\theta}(z|\theta) \| \mathcal{N}(0,I))$ | パラメータエンコーダを正則化 |
| `KL_x1x2` | $D_{\text{KL}}(q_{\phi_x} \| q_{\phi_\theta})$ | 両エンコーダの潜在表現を整合 |
| `KL_x2x1` | $D_{\text{KL}}(q_{\phi_\theta} \| q_{\phi_x})$ | 両エンコーダの潜在表現を整合 |
| `rec_xx` | $-\mathbb{E}_{z\sim q_{\phi_x}}[\log p_\eta(\tilde{x}|z)]$ | FRFから再構成できているか |
| `rec_wx` | $-\mathbb{E}_{z\sim q_{\phi_\theta}}[\log p_\eta(\tilde{x}|z)]$ | パラメータからFRFを予測できるか |

**`KL_x1x2 + KL_x2x1`**（α=5）が特に重要：これにより $q_{\phi_x}(z|x)$ と $q_{\phi_\theta}(z|\theta)$ が同じ潜在表現を持つように強制される。

### 4.3 再構成損失の計算

$$L_{\text{rec}} = -\mathbb{E}_{z \sim q_\phi(z|x)}[\log p_\eta(\tilde{x}|z)]$$

ガウス分布 $p_\eta(\tilde{x}|z) = \mathcal{N}(\hat{x}_\mu, \text{diag}(\hat{x}_\sigma^2))$ のとき：

$$-\log p_\eta(\tilde{x}|z) = \frac{1}{2}\sum_i \left[\frac{(\tilde{x}_i - \hat{x}_{\mu,i})^2}{\hat{x}_{\sigma,i}^2} + \log\hat{x}_{\sigma,i}^2 + \log(2\pi)\right]$$

```python
# mvae.py:32-35
def rec_loss_norm4D(x, mean, var):
    return -torch.mean(
        torch.sum(-0.5 * ((x - mean) ** 2 / var + torch.log(var)
                          + torch.log(torch.tensor(2 * torch.pi))), dim=(1, 2, 3))
    )
```

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
