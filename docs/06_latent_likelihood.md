# 潜在空間ベース尤度近似

## 1. 問題の本質

ベイズ更新では尤度 $L(\theta; x_{\text{obs}}) = p(x_{\text{obs}} | \theta)$ が必要。

しかし観測データが高次元（1024次元のFRF）の場合：
- 明示的な確率モデルを定義するのが困難
- シミュレーターが決定論的なため直接の尤度評価が不可能
- （$h(\theta)$ は決定論的なので $p(x|h(\theta))$ は密度ゼロになる可能性）

これが **尤度フリー推論（Likelihood-Free Inference）** が必要な理由。

---

## 2. 潜在変数による近似の導出

### Step 1: 潜在変数の導入

低次元潜在変数 $z \in \mathbb{R}^{n_z}$（$n_z = 8 \ll n_x = 1024$）を導入し、以下を仮定：

$$x \perp \theta \mid z$$

つまり、$z$ が与えられれば $x$ と $\theta$ は独立。

### Step 2: 周辺化による分解

$$L(\theta; x_{\text{obs}}) = p(x_{\text{obs}} | \theta) = \int_Z p(x_{\text{obs}} | z) p(z | \theta) dz$$

### Step 3: ベイズの定理の適用

$p(x|z) = p(z|x)p(x)/p(z)$ を使って：

$$= \int_Z \frac{p(z | x_{\text{obs}}) p(z | \theta)}{p(z)} dz \cdot \underbrace{p(x_{\text{obs}})}_{\text{定数 } c}$$

### Step 4: ニューラルネットワークで近似

$$\approx \hat{L}(\theta; x_{\text{obs}}) := \int_Z \frac{q_{\phi_x}(z | x_{\text{obs}}) \cdot q_{\phi_\theta}(z | \theta)}{p(z)} dz$$

- $q_{\phi_x}(z | x_{\text{obs}})$：FRF エンコーダ（`enc_x`）
- $q_{\phi_\theta}(z | \theta)$：パラメータエンコーダ（`enc_w`）
- $p(z) = \mathcal{N}(0, I)$：潜在変数の事前分布

---

## 3. 解析的な閉形式の導出

**三つの分布がすべてガウス分布**であることが鍵：

$$q_{\phi_x}(z | x) = \mathcal{N}(z | \mu_1, \sigma_1^2)$$
$$q_{\phi_\theta}(z | \theta) = \mathcal{N}(z | \mu_2, \sigma_2^2)$$
$$p(z) = \mathcal{N}(z | \mu_3, \sigma_3^2) = \mathcal{N}(z | 0, 1)$$

積分を計算する（各次元独立なので1次元で考える）：

$$\int_{-\infty}^{\infty} \frac{\mathcal{N}(z|\mu_1, \sigma_1^2) \cdot \mathcal{N}(z|\mu_2, \sigma_2^2)}{\mathcal{N}(z|\mu_3, \sigma_3^2)} dz$$

分子の積と分母は全てガウス型なので、この積分は**解析的に計算可能**。

### 導出の詳細

各ガウス分布の対数を取ると：

$$\log \text{numerator} = -\frac{(z-\mu_1)^2}{2\sigma_1^2} - \frac{(z-\mu_2)^2}{2\sigma_2^2} + \text{const}$$
$$\log \text{denominator} = -\frac{(z-\mu_3)^2}{2\sigma_3^2} + \text{const}$$

被積分関数の対数は $z$ の二次式：

$$\log f(z) = -az^2 - bz - c + \text{const}$$

ここで：

$$a = \frac{1}{2\sigma_1^2} + \frac{1}{2\sigma_2^2} - \frac{1}{2\sigma_3^2}$$
$$b = -\frac{\mu_1}{\sigma_1^2} - \frac{\mu_2}{\sigma_2^2} + \frac{\mu_3}{\sigma_3^2}$$

ガウス積分 $\int e^{-az^2 + bz} dz = \sqrt{\pi/a} \cdot e^{b^2/(4a)}$ を適用すると：

$$\hat{L}(\theta; x_{\text{obs}}) = d \cdot e^{(b^2 - 4ac)/(4a)} \cdot \sqrt{\pi/a}$$

```python
# latentlik.py: latent_space_loglik 関数
def latent_space_loglik(mu_obs, vr_obs, mu_sim, vr_sim, alp=0.0, tau=1e-4, mu_pri=0.0, vr_pri=1.0, eps=0.0):
    vr1 = mu_obs の分散 (観測FRFから)
    vr2 = (1 + alp) * vr_sim + tau  (パラメータから、不確かさ調整つき)
    vr3 = vr_pri = 1.0  (事前分布の分散)
    mu1 = mu_obs
    mu2 = mu_sim
    mu3 = mu_pri = 0.0

    a = 1/(2*vr1) + 1/(2*vr2) - 1/(2*vr3)
    b = -(mu1/vr1) - (mu2/vr2) + (mu3/vr3)
    c = mu1**2/(2*vr1) + mu2**2/(2*vr2) - mu3**2/(2*vr3)
    d = sqrt(vr3 / (2*pi*vr1*vr2))

    # 対数尤度（各潜在次元の和）
    lp = log(d) + (b**2 - 4*a*c)/(4*a) + 0.5*log(pi/a)
    return sum(lp, dim=1)  # 全 z_dim 次元の和
```

---

## 4. パラメータ `alp` と `tau` の役割

$$\sigma_2^2 = (1 + \alpha) \sigma_{\text{sim}}^2 + \tau$$

- **$\alpha$ (alp)**：シミュレーターの不確かさを増幅させるスケール係数
  - $\alpha = 0$：シミュレーターの予測分散をそのまま使用
  - $\alpha > 0$：モデル誤差を考慮して分散を広げる

- **$\tau$ (tau)**：数値的安定性のための最小分散値
  - $\tau = 10^{-4}$：分散がゼロになるのを防ぐ

---

## 5. 実装の全体フロー

```python
# inference.py での使い方

# Step 1: 観測データの潜在埋め込みを事前計算（1回だけ）
_, mu_obs, vr_obs = enc_x(y_obs)  # FRF → 潜在空間

# Step 2: 尤度計算（SMCの各イテレーションで呼ばれる）
def __call__(self, theta):
    theta = stdnorm.cdf(theta)  # 正規化変換（後述）
    _, mu_sim, vr_sim = enc_w(theta)  # パラメータ → 潜在空間
    return latent_space_loglik(mu_obs, vr_obs, mu_sim, vr_sim, alp=1.0, tau=0.0)
```

**重要な最適化**：観測データの潜在埋め込み $(\mu_{\text{obs}}, \sigma^2_{\text{obs}})$ は推論中に変わらないため、一度だけ計算して保存（`MVAEBasedLogLikelihood.__init__` で実施）。

---

## 6. 正規化変換（CDF変換）

```python
# inference.py:87
def __call__(self, theta):
    theta = stdnorm.cdf(theta)  # Φ(θ)：標準正規分布の CDF
    ...
```

SMCは無制約空間（$\mathbb{R}$）でサンプリングするが、実際のパラメータは $[0.33, 3.00]$ に制限されている。

**解決策**：標準正規分布 $\mathcal{N}(0,1)$ の CDF $\Phi$ を使って変換：

$$\theta_{\text{physical}} = \Phi(\theta_{\text{latent}}) \times (U - L) + L$$

- SMC は $\theta_{\text{latent}} \in \mathbb{R}$ を自由に探索
- 実際の計算には $\theta_{\text{physical}} \in [L, U] = [0.33, 3.00]$ を使用
- 事前分布は $\mathcal{N}(0, 1)$（一様分布を正規 CDF で変換した等価表現）

---

## 7. なぜこの手法が効率的なのか

### 従来手法との比較

| 手法 | シミュレーター呼び出し |
|------|---------------------|
| 従来の MCMC（直接尤度） | 推論中に何万〜何十万回 |
| ABC（Approximate Bayes Computation） | 推論中に何万〜何十万回 |
| **LSBI（本手法）** | **学習時のみ**（推論中は0回） |

### アモータイゼーション（Amortization）
- 訓練フェーズ：10万サンプルでシミュレーターを呼び出し → MVAE を学習
- 推論フェーズ：訓練済み MVAE を使って尤度を計算（シミュレーターは不要）

→ 新しい観測データが来ても、SMC のみ再実行すればよい（MVAE の再学習不要）
