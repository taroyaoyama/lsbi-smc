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

$$q_{\phi_x}(z | x_{\text{obs}}) = \mathcal{N}(z | \mu_1, \sigma_1^2)$$
$$q_{\phi_\theta}(z | \theta) = \mathcal{N}(z | \mu_2, \sigma_2^2)$$
$$p(z) = \mathcal{N}(z | \mu_3, \sigma_3^2) = \mathcal{N}(z | 0, 1)$$

エンコーダの出力は対角共分散ガウスなので、$z$ の各次元は独立に積分できる。
以下では1次元の場合を導出し、最後に各次元の和をとる。

評価する積分は：

$$I := \int_{-\infty}^{\infty} \frac{\mathcal{N}(z|\mu_1, \sigma_1^2) \cdot \mathcal{N}(z|\mu_2, \sigma_2^2)}{\mathcal{N}(z|\mu_3, \sigma_3^2)} dz$$

### Step 1: 正規化定数を被積分関数の外へ括り出す

各ガウス分布を **正規化定数 × 指数項** に分解する：

$$\mathcal{N}(z|\mu_i, \sigma_i^2) = \frac{1}{\sqrt{2\pi\sigma_i^2}} \exp\!\left(-\frac{(z-\mu_i)^2}{2\sigma_i^2}\right)$$

これを $I$ に代入し、$z$ に依存しない正規化定数を積分の外に出す：

$$I = \underbrace{\frac{\dfrac{1}{\sqrt{2\pi\sigma_1^2}} \cdot \dfrac{1}{\sqrt{2\pi\sigma_2^2}}}{\dfrac{1}{\sqrt{2\pi\sigma_3^2}}}}_{=:\, d} \int_{-\infty}^{\infty} \exp\!\left(-\frac{(z-\mu_1)^2}{2\sigma_1^2} - \frac{(z-\mu_2)^2}{2\sigma_2^2} + \frac{(z-\mu_3)^2}{2\sigma_3^2}\right) dz$$

ここで定義した前因子 $d$ を整理すると：

$$d = \sqrt{\frac{2\pi\sigma_3^2}{(2\pi\sigma_1^2)(2\pi\sigma_2^2)}} = \sqrt{\frac{\sigma_3^2}{2\pi \sigma_1^2 \sigma_2^2}}$$

これがコード [latentlik.py:32](../src/lsbi_smc/likelihood/latentlik.py#L32) の
`d = sqrt(vr3 / (2*pi*vr1*vr2))` に対応する。
要するに **$d$ は3つのガウス分布の正規化定数を組み合わせたもの**で、
積分の外に定数として括り出した結果として現れている。

### Step 2: 指数の中身を $z$ の二次式として整理

各二次項を展開する：

$$-\frac{(z-\mu_i)^2}{2\sigma_i^2} = -\frac{z^2}{2\sigma_i^2} + \frac{\mu_i z}{\sigma_i^2} - \frac{\mu_i^2}{2\sigma_i^2}$$

3つの寄与（$i=1,2$ は負、$i=3$ は正）をまとめると、指数の中身は $z$ の二次式 $-az^2 - bz - c$ となる：

$$
\begin{aligned}
a &= \frac{1}{2\sigma_1^2} + \frac{1}{2\sigma_2^2} - \frac{1}{2\sigma_3^2} \\
b &= -\frac{\mu_1}{\sigma_1^2} - \frac{\mu_2}{\sigma_2^2} + \frac{\mu_3}{\sigma_3^2} \\
c &= \frac{\mu_1^2}{2\sigma_1^2} + \frac{\mu_2^2}{2\sigma_2^2} - \frac{\mu_3^2}{2\sigma_3^2}
\end{aligned}
$$

これらは [latentlik.py:28-31](../src/lsbi_smc/likelihood/latentlik.py#L28-L31) の `a, b, c` と一致する。

> **収束条件**：積分が有限値に収束するには $a > 0$、すなわち
> $\dfrac{1}{\sigma_1^2} + \dfrac{1}{\sigma_2^2} > \dfrac{1}{\sigma_3^2}$ が必要。
> 本実装では $\sigma_3^2 = 1$（事前分布）でエンコーダが出力する $\sigma_1^2, \sigma_2^2$ は通常十分小さいので、この条件は自然に満たされる。

### Step 3: 平方完成によりガウス積分の標準形に帰着

積分は

$$I = d \cdot \int_{-\infty}^{\infty} \exp(-az^2 - bz - c)\, dz = d \cdot e^{-c} \int_{-\infty}^{\infty} \exp(-az^2 - bz)\, dz$$

指数部を $z$ について平方完成する：

$$-az^2 - bz = -a\!\left(z^2 + \frac{b}{a}z\right) = -a\!\left(z + \frac{b}{2a}\right)^2 + \frac{b^2}{4a}$$

これを代入して定数項を積分の外に出す：

$$\int_{-\infty}^{\infty} \exp(-az^2 - bz)\, dz = e^{b^2/(4a)} \int_{-\infty}^{\infty} \exp\!\left(-a\!\left(z + \tfrac{b}{2a}\right)^2\right) dz$$

平行移動 $u = z + b/(2a)$ により、標準ガウス積分

$$\int_{-\infty}^{\infty} e^{-au^2}\, du = \sqrt{\frac{\pi}{a}} \quad (a > 0)$$

に帰着する。したがって：

$$I = d \cdot e^{-c} \cdot e^{b^2/(4a)} \cdot \sqrt{\frac{\pi}{a}} = d \cdot \exp\!\left(\frac{b^2 - 4ac}{4a}\right) \cdot \sqrt{\frac{\pi}{a}}$$

（最後の等式では $b^2/(4a) - c = (b^2 - 4ac)/(4a)$ を使った。）

### Step 4: 対数をとり各次元の和をとる

対数をとると：

$$\log I = \log d + \frac{b^2 - 4ac}{4a} + \frac{1}{2}\log\frac{\pi}{a}$$

これが [latentlik.py:34](../src/lsbi_smc/likelihood/latentlik.py#L34) の式そのもの。

最後に、潜在変数の各次元 $j = 1, \ldots, n_z$ は独立なので、
全次元での対数尤度は単純な和：

$$\log \hat{L}(\theta; x_{\text{obs}}) = \sum_{j=1}^{n_z} \log I_j$$

これがコード [latentlik.py:37](../src/lsbi_smc/likelihood/latentlik.py#L37) の
`return torch.sum(lp, dim=1)` に対応する。

### コードとの対応

```python
# latentlik.py: latent_space_loglik 関数
def latent_space_loglik(mu_obs, vr_obs, mu_sim, vr_sim, alp=0.0, tau=1e-4, mu_pri=0.0, vr_pri=1.0, eps=0.0):
    vr1 = vr_obs                       # σ₁²: 観測 FRF の潜在分散
    vr2 = (1 + alp) * vr_sim + tau     # σ₂²: パラメータの潜在分散（不確かさ補正つき）
    vr3 = vr_pri = 1.0                 # σ₃²: 事前分布の分散
    mu1 = mu_obs                       # μ₁
    mu2 = mu_sim                       # μ₂
    mu3 = mu_pri = 0.0                 # μ₃

    a = 1/(2*vr1) + 1/(2*vr2) - 1/(2*vr3)                     # Step 2
    b = -(mu1/vr1) - (mu2/vr2) + (mu3/vr3)                    # Step 2
    c = mu1**2/(2*vr1) + mu2**2/(2*vr2) - mu3**2/(2*vr3)      # Step 2
    d = sqrt(vr3 / (2*pi*vr1*vr2))                            # Step 1: 正規化定数の前因子

    # 対数尤度 = log d + (b²-4ac)/(4a) + (1/2)log(π/a)
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

### 6.1 なぜ変換が必要か：3 つの空間

本実装では、$\theta$ が **3 つの異なる空間** を行き来する。
まずこれを整理することが鍵になる。

| 空間 | 範囲 | 用途 |
|------|------|------|
| **無制約空間** $\theta_{\text{latent}}$ | $\mathbb{R}$（実数全体） | SMC のサンプリング・MCMC カーネル |
| **正規化空間** $\theta_{\text{norm}}$ | $[0, 1]$ | MVAE の入力（学習時もこのスケール） |
| **物理空間** $\theta_{\text{phys}}$ | $[L, U] = [0.33, 3.00]$ | シミュレーター入力・最終出力 |

各空間は以下の写像で繋がっている：

$$\theta_{\text{latent}} \xrightarrow{\ \Phi(\cdot)\ } \theta_{\text{norm}} \xrightarrow{\ (U-L)\cdot + L\ } \theta_{\text{phys}}$$

ここで $\Phi$ は標準正規分布 $\mathcal{N}(0,1)$ の累積分布関数（CDF）。

**なぜ無制約空間でサンプリングするのか？**
SMC の中で動く RW-Metropolis カーネル（[kernel.py](../src/lsbi_smc/smc/kernel.py)）は
ガウス提案 $\theta' = \theta + \epsilon$ を使う。
もし直接 $[L, U]$ 上でサンプリングすると、提案が境界を越えるたびに
棄却または反射の処理が必要になる。
$\theta_{\text{latent}} \in \mathbb{R}$ なら**境界が存在しない**ので、
カーネルがシンプルに保てる。

### 6.2 確率積分変換（Probability Integral Transform）

CDF 変換のキモは、次の古典的な事実：

> $X \sim \mathcal{N}(0, 1)$ ならば、$\Phi(X) \sim \text{Uniform}(0, 1)$ である。

直観的には：
- $\Phi$ は $\mathbb{R}$ を $[0, 1]$ に**単調に押し込む** S 字曲線
- ガウス分布の「中央が密で裾が疎」という偏りと、$\Phi$ の「中央で傾きが急、裾で緩い」という形が**ちょうど打ち消し合う**
- 結果として、$\Phi(X)$ は $[0, 1]$ 上で一様分布になる

```
     θ_latent ~ N(0,1)         Φ(θ_latent) ~ Uniform(0,1)
        ▲                              ▲
        │  ╱╲                          │ ──────────
        │ ╱  ╲                         │
        │╱    ╲       ──Φ─→            │
        └──────────►                   └──────────►
       -3  0   3                       0    0.5    1
```

これを線形に伸ばせば、$\theta_{\text{phys}} = \Phi(\theta_{\text{latent}})(U-L) + L$ は
$[L, U]$ 上の一様分布になる。

### 6.3 事前分布の等価性

[inference.py:97](../src/lsbi_smc/example_shear4dof/inference.py#L97) では事前分布を
$\mathcal{N}(0, 1)$ と定義している：

```python
variables = [Normal(label, Constant(0.0), Constant(1.0)) for label in k_labels]
```

これは「$\theta_{\text{latent}} \sim \mathcal{N}(0,1)$」を意味する。
6.2 の事実から、これは**物理空間で見れば $\theta_{\text{phys}} \sim \text{Uniform}(L, U)$ という一様事前分布と等価**である。

つまり同じ「$[L, U]$ 上の無情報事前分布」を、座標系を変えて表現しているだけ：

| 座標系 | 事前分布の表現 |
|--------|---------------|
| 物理空間 $\theta_{\text{phys}}$ | $\text{Uniform}(L, U)$ |
| 無制約空間 $\theta_{\text{latent}}$ | $\mathcal{N}(0, 1)$ |

サンプリングが楽な座標系として後者を選んでいる、と理解すればよい。

### 6.4 コード上での2つの登場箇所

CDF 変換は inference スクリプト内で **2 箇所** に現れる。
役割が違うので注意する。

**(a) 尤度評価の直前** — [inference.py:87](../src/lsbi_smc/example_shear4dof/inference.py#L87)

```python
def __call__(self, theta):
    theta = stdnorm.cdf(theta)   # θ_latent (∈ℝ) → θ_norm (∈[0,1])
    return super().__call__(theta, alp=1.0, tau=0.00)
```

ここでは `enc_w` に入力するために $[0, 1]$ スケール（正規化空間）に変換する。
MVAE の学習時も入力は $[0, 1]$ だったので、推論時もそれに合わせる必要がある。

**(b) SMC 終了後の出力変換** — [inference.py:123-125](../src/lsbi_smc/example_shear4dof/inference.py#L123-L125)

```python
pop = smc1.pops[-1].detach().cpu().numpy()  # θ_latent
pop = norm.cdf(pop)                          # → θ_norm
pop = pop * (ULIM - LLIM) + LLIM             # → θ_phys
```

事後分布のサンプルを人間が読める物理スケール（剛性比）に戻す処理。
2 段階の変換 $\theta_{\text{latent}} \to \theta_{\text{norm}} \to \theta_{\text{phys}}$ がそのままコードに対応している。

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
