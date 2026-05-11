# Sequential Monte Carlo (SMC) サンプラー

本章は SMC の理論と実装を、[02_probability.md](02_probability.md) で導入した
「テンパリング・重点サンプリング・MCMC」を一本の流れに統合する形で解説する。
02 章では各要素と全体像を概念レベルで紹介したが、ここでは数式・アルゴリズム・
コードの 3 つの層で再度組み立て直す。

---

## 1. なぜ SMC が必要なのか

### 1.1 MCMC 単独の限界

通常の MCMC（例：Random Walk Metropolis-Hastings、[02 章 §8.7](02_probability.md)）には次の弱点がある：

- **多峰性に弱い**：尤度の地形にいくつも山があるとき、最初に到達したモードから抜け出せないことが多い（[02 章 §8.9](02_probability.md)）
- **バーンインが長い**：事前分布の散らばった初期点から事後分布の山に辿り着くまでに、棄却ばかりが続く期間が必要
- **高次元で混合が遅い**：パラメータが増えるほどランダムウォークが空間を覆い切るのに時間がかかる

これらは「**事前分布と事後分布の形があまりに違いすぎる**」ことに根ざしている。
事後分布は鋭い山を持つ複雑な分布だが、いきなりそこを目標にすると山を見つけられない。

### 1.2 SMC の発想：滑らかに繋ぐ

SMC は「**事前分布 → 事後分布を一気に飛ばず、間に滑らかな中間分布列を挟む**」ことで解決する。
このアイデアそのものは **テンパリング**（[02 章 §8.10](02_probability.md)）。
SMC はテンパリングを **粒子集合 + 重点サンプリング + MCMC** の枠組みで実行する手法、
と理解するとよい。

$$
\underbrace{p(\theta)}_{\beta_0 = 0\ \text{(事前)}}
\ \xrightarrow{\beta_1}\ \pi_{\beta_1}
\ \xrightarrow{\beta_2}\ \pi_{\beta_2}
\ \xrightarrow{\cdots}\ \pi_{\beta_{T-1}}
\ \xrightarrow{\beta_T = 1}\ \underbrace{p(\theta \mid x_{\text{obs}})}_{\text{事後}}
$$

各ステップで粒子集合を更新し、最後の $\beta = 1$ に到達した時点の粒子が事後分布のサンプルになる。

---

## 2. 中間分布列とテンパリング

### 2.1 中間分布の定義

「事後分布の尤度部分だけを $\beta$ 乗する」ことで、事前から事後を繋ぐ分布列を作る：

$$\pi_\beta(\theta) = \frac{1}{Z_\beta}\, \hat{L}(\theta; x_{\text{obs}})^{\beta} \cdot p(\theta), \qquad 0 = \beta_0 < \beta_1 < \cdots < \beta_T = 1$$

ここで $\hat{L}(\theta; x_{\text{obs}})$ は [06 章](06_latent_likelihood.md) の潜在空間近似尤度、
$Z_\beta$ は正規化定数。$\beta$ を変えると：

| $\beta$ | 分布 | 解釈 |
|---------|------|------|
| $0$ | $p(\theta)$ | 事前分布。データを全く反映しない、平坦で広がった分布 |
| 中間値 | $\propto \hat{L}^\beta p$ | データを部分的に反映。山は浅く、谷は渡りやすい |
| $1$ | $p(\theta \mid x_{\text{obs}})$ | 事後分布。データを完全に反映、山は鋭い |

### 2.2 地形のイメージ

$\beta$ は「データへの信頼度」のつまみだと思えばよい。$\beta$ を上げると尤度の影響が強まり、
データが支持する $\theta$ の周辺に「山」が育ち、それ以外の谷が深くなる。
$\beta$ が低いうちに粒子が空間を広く探検し、$\beta$ が上がるにつれて山に集まっていく
——これが SMC の動きの本質。

---

## 3. SMC のメインループ：全体像

ここからが本章の核心。02 章 §8.11 で示した擬似コードを、各ステップが「**何を達成して、なぜそれが正しいか**」も含めて再掲する。

```
[準備]
  観測 x_obs を固定する
  尤度 L̂(θ; x_obs) を MVAE から取得（[06 章]）

[初期化]                                          ─ q = β_0 = 0
  θ^(n) ~ p(θ)  for n = 1..N             ← 事前分布から N 粒子をサンプル
  log L̂^(n) := log L̂(θ^(n); x_obs) を全粒子について計算

[ループ: k = 0, 1, 2, ... ]
  ┌──────────────────────────────────────────────────────┐
  │ Step 1【次の温度を決める】                            │
  │   Δβ を二分探索で決定                                │
  │   （重点サンプリング重みの ESS が目標値 0.8N を満たす最大 Δβ）│
  │   β_{k+1} := β_k + Δβ                                │
  ├──────────────────────────────────────────────────────┤
  │ Step 2【重みを計算 → リサンプリング】                 │
  │   w^(n) ∝ exp(Δβ · log L̂^(n))                       │
  │   多項分布から N 粒子を重み付き再抽出                 │
  │   → 重複した粒子の集合になる（多様性は次のステップで回復）│
  │   重みを 1/N に等しくリセット                         │
  ├──────────────────────────────────────────────────────┤
  │ Step 3【MCMC ムーブで多様性を回復】                   │
  │   各粒子に対して、π_{β_{k+1}} を不変分布とする MH を   │
  │   mcmc_iter 回適用                                    │
  │   → 重複していた粒子が独立に少しずつ散らばる          │
  ├──────────────────────────────────────────────────────┤
  │ β_{k+1} = 1 になったら終了、そうでなければループ続行  │
  └──────────────────────────────────────────────────────┘

[出力]
  最終時点の粒子集合 {θ^(n)} が事後分布 p(θ | x_obs) の近似サンプル
```

**3 つの操作の役割分担：**

| 操作 | 何を達成する | 何を引き起こす |
|------|-------------|---------------|
| **重み付け** | $\pi_{\beta_k} \to \pi_{\beta_{k+1}}$ の橋渡し（情報の重み付け） | 重みの偏り → ESS 低下 |
| **リサンプリング** | 重みの偏りを「粒子の複製・削除」に変換 | 粒子の重複 → 多様性低下 |
| **MCMC ムーブ** | 各粒子を $\pi_{\beta_{k+1}}$ に従って動かして多様性回復 | 計算コスト（中間分布の評価 × mcmc_iter） |

3 つは相互補完的：単独ではどれも崩壊するが、組み合わせると相補的に欠点を埋め合う。

以下、それぞれを丁寧に見ていく。

---

## 4. 重み更新：隣接する中間分布間の重点サンプリング

### 4.1 出発点：粒子は今 $\pi_{\beta_k}$ に従っている

ループの $k$ 番目の入口では、前回までの処理で粒子集合 $\{\theta^{(n)}\}$ は近似的に
$\pi_{\beta_k}$ に従っている（初回は $\pi_0 = p(\theta)$ から直接サンプルしている）。
ここから $\pi_{\beta_{k+1}}$ に従う粒子集合を作りたい。

この場面は **重点サンプリングそのもの**（[02 章 §6](02_probability.md)）：

- 提案分布（粒子が今従っている分布） $= \pi_{\beta_k}$
- 目的分布（期待値を計算したい分布） $= \pi_{\beta_{k+1}}$
- 各粒子の重み $w^{(n)} \propto \pi_{\beta_{k+1}}(\theta^{(n)}) / \pi_{\beta_k}(\theta^{(n)})$

> **記号の注意**：02 章 §6 では提案分布を $q$、目的分布を $p$ と書いていたが、本章では $q$ は SMC の逆温度変数（コードの `q_tar`, `self.q`、$\beta$ と同義）を指す。混乱を避けるため、ここでは $q, p$ という別名を導入せず $\pi_{\beta_k}, \pi_{\beta_{k+1}}$ で直接書く。

### 4.2 重みの導出

$\pi_\beta(\theta) = Z_\beta^{-1}\, \hat{L}(\theta)^\beta\, p(\theta)$ を代入して比をとる：

$$
\frac{\pi_{\beta_{k+1}}(\theta)}{\pi_{\beta_k}(\theta)}
= \frac{Z_{\beta_{k+1}}^{-1}\, \hat{L}^{\beta_{k+1}}\, p(\theta)}{Z_{\beta_k}^{-1}\, \hat{L}^{\beta_k}\, p(\theta)}
= \underbrace{\frac{Z_{\beta_k}}{Z_{\beta_{k+1}}}}_{\text{粒子に依存しない定数}} \cdot \hat{L}(\theta)^{\Delta\beta}
$$

ここで $\Delta\beta = \beta_{k+1} - \beta_k$。事前分布 $p(\theta)$ は分子分母で消え、
正規化定数の比は **「合計 1 に正規化する」ときに自動的に吸収される**。
したがって実装で計算するのはシンプルに：

$$\boxed{w^{(n)} \propto \hat{L}(\theta^{(n)})^{\Delta\beta}}$$

### 4.3 対数スケールでの安定化

$\hat{L}^{\Delta\beta}$ は数値的にアンダーフロー/オーバーフローしやすいので、対数で扱う：

$$\log w^{(n)} = \Delta\beta \cdot \log \hat{L}(\theta^{(n)}) + \text{const}$$

正規化前に最大値を引いてから指数化することで安定する（[smc.py:72-82](../src/lsbi_smc/smc/smc.py#L72-L82)）：

```python
# Particles.eval_weights
z = self.dq * self.lp                              # log w (定数差は無視可)
z = torch.nan_to_num(z, neginf=-1e30, posinf=1e30)
z = z - torch.max(z)                                # 最大を 0 にシフト（exp の安定化）
w = torch.exp(z)
w = w / w.sum()                                     # 正規化（定数が消える）
```

---

## 5. ESS：重みの品質指標と $\Delta\beta$ の決定

### 5.1 ESS の定義

重みが偏ると重点サンプリングは破綻する（[02 章 §6](02_probability.md)）。
有効サンプルサイズ ESS（Effective Sample Size）はこれを定量化する：

$$\text{ESS}(\Delta\beta) = \frac{\bigl(\sum_n w^{(n)}\bigr)^2}{\sum_n (w^{(n)})^2}$$

正規化された重み（$\sum w = 1$）なら $\text{ESS} = 1 / \sum w^2$。

| ESS | 状態 |
|-----|------|
| $N$（最大） | 全粒子が等重み。$\pi_{\beta_k}$ と $\pi_{\beta_{k+1}}$ がほぼ同じ形 |
| $\sim N/2$ | 半数程度が有効。実用上はまだ許容 |
| $\sim 1$（最小） | ほぼ 1 粒子に重みが集中。重点サンプリングは破綻している |

### 5.2 なぜ ESS が「有効な粒子数」と呼べるのか

直感的には：$N$ 粒子のうち、有効に期待値計算に寄与しているのは
「等重みなら何個分の粒子と同じか」を測っている。

簡単な極端ケースでの確認：

- 全粒子が等重み（$w^{(n)} = 1/N$）：$\sum w^2 = N \cdot (1/N)^2 = 1/N$ → $\text{ESS} = N$
- 1 粒子に集中（$w^{(1)} = 1$、他は 0）：$\sum w^2 = 1$ → $\text{ESS} = 1$
- $k$ 粒子に均等集中（各 $1/k$、他は 0）：$\sum w^2 = k \cdot (1/k)^2 = 1/k$ → $\text{ESS} = k$

このように ESS は「実質的に効いている粒子数」を表す。

### 5.3 適応的 $\Delta\beta$：二分探索

$\Delta\beta$ を大きくとると一気に進むが ESS が崩れる。
小さくとると ESS は保てるが温度ステップ数が増える。
そこで **「ESS がちょうど目標値（例：$0.8N$）になる最大の $\Delta\beta$」を二分探索で見つける**：

```python
# smc.py:113-132
def _find_next_q(q_prev, q_tar, lp_np, ess_tar, tol=1e-6, maxit=50):
    lo, hi = q_prev, q_tar
    # 一気にターゲットまで上げて ESS が十分なら、そのまま採用
    if _ess_from_lp(hi - q_prev, lp_np) >= ess_tar:
        return hi
    # そうでなければ二分探索
    for _ in range(maxit):
        mid = 0.5 * (lo + hi)
        if _ess_from_lp(mid - q_prev, lp_np) < ess_tar:
            hi = mid                # ESS 足りない → Δβ を小さく
        else:
            lo = mid                # ESS 十分 → Δβ をもう少し大きく
        if abs(hi - lo) < tol:
            break
    return 0.5 * (lo + hi)
```

**なぜ二分探索で良いのか：** $\Delta\beta$ を 0 から大きくしていくと、
$\text{ESS}(\Delta\beta)$ はほぼ単調減少する（重みが指数的に偏っていくため）。
単調関数の根を探すので二分探索が使える。

---

## 6. リサンプリング：重みの偏りを粒子集合に反映

### 6.1 重みの偏りを「個数」に変換する

重みを付けた時点では、粒子は同じだが「価値」が違う状態。
このまま MCMC を回しても、価値の低い粒子に計算コストを使う羽目になる。
そこで **重みに比例して粒子を再抽出** する。

```python
# smc.py:84-92
def resample(self, dq=None):
    self.eval_weights()
    idx = torch.multinomial(self.weights, self.size, replacement=True)
    self.pop = self.pop[idx]      # 重み大きい粒子は何度も選ばれる
    self.lp = self.lp[idx]        # 対応する log-likelihood も付け替え
```

`torch.multinomial(..., replacement=True)` は **多項抽出**：
各粒子 $n$ が重み $w^{(n)}$ で選ばれ、それを $N$ 回独立に繰り返す。
結果として：

- 重み大の粒子は複数回コピーされる
- 重み小の粒子は消える
- 出てきた粒子は全て等重みとみなせる（だから [smc.py:204-210](../src/lsbi_smc/smc/smc.py#L204-L210) でリセット）

### 6.2 リサンプリングが引き起こす副作用

重み付けは数学的に正しいが、リサンプリングを挟むと **粒子の重複** が発生する。
複製された粒子は完全に同じ $\theta$ 値なので、独立な情報源としての価値は減る
（多様性の低下）。これを次のステップで MCMC が解消する。

> **注意：**「リサンプリングだけで $\pi_{\beta_{k+1}}$ のサンプルになっているのでは？」と思うかもしれない。確かに重み付け＋リサンプリングは形式的には $\pi_{\beta_{k+1}}$ から引いたことに対応する。だが「同じ $\theta$ の重複が多い集合」と「独立なサンプル」では実用上の有効粒子数が違うので、MCMC で揺らがせて事後分布の表現力を高める。

---

## 7. MCMC ムーブ：多様性の回復

### 7.1 何のための MCMC か

リサンプリング直後は「同じ $\theta$ が何度も登場する集合」になる。これを MCMC で散らすが、**散らした後も $\pi_{\beta_{k+1}}$ に従っていなければならない**。
そこで **「$\pi_{\beta_{k+1}}$ を不変分布とする」遷移カーネル** を使う。

不変性の意味：$\theta^{(n)} \sim \pi_{\beta_{k+1}}$ から始めて MH を 1 ステップ動かしても、結果はまだ $\pi_{\beta_{k+1}}$ に従う。
だから、リサンプリングで $\pi_{\beta_{k+1}}$ に従う集合になった後、何ステップ MH を回しても分布は変わらない——粒子の位置だけがバラけていく。

### 7.2 RW-MH カーネル

各粒子について以下を独立に実行：

1. 提案 $\theta^* = \theta + \epsilon$（$\epsilon$ は適応的共分散から引いたガウス、§7.3）
2. 採択比の計算：

$$\alpha = \min\!\left(1,\ \exp\bigl[\,\beta_{k+1} (\log \hat{L}(\theta^*) - \log \hat{L}(\theta)) + \log p(\theta^*) - \log p(\theta)\,\bigr]\right)$$

   ここで $\beta_{k+1}$（コード中では `q`）が掛かっている点が普通の MH と違う。
   これは目標が事後分布 $\pi_1$ ではなく中間分布 $\pi_{\beta_{k+1}}$ だから。
3. $u \sim \text{Uniform}(0,1)$ を引いて $u < \alpha$ なら採択、さもなければ棄却

実装は [kernel.py:39-48](../src/lsbi_smc/smc/kernel.py#L39-L48)：

```python
log_rat = (
    q * (lp_new - particles.lp)             # 尤度差 × 温度
    + prior.lp(pop_new) - prior.lp(particles.pop)  # 事前分布の差
)
u = torch.rand_like(log_rat)
accept = (log_rat > torch.log(u + 1e-12)) & within
```

`mcmc_iter` 回繰り返すことで、複製されていた粒子同士が徐々に分離していく。

### 7.3 Ching & Chen の適応的提案分布

#### 7.3.1 何が問題か：提案分布の設計ジレンマ

RW-MH カーネルの中で、提案 $\theta^* = \theta + \delta$ の **「揺らぎ $\delta$ をどんな分布から引くか」** を決める必要がある。最も素朴な選択は等方ガウス $\delta \sim \mathcal{N}(0, \sigma^2 I)$ だが、$\sigma$ を固定値にすると次のジレンマに陥る：

| $\sigma$ の選び方 | 結果 |
|------------------|------|
| 大きすぎる | 提案先がほぼ常に低確率領域に着地 → 採択率がほぼ 0 → 連鎖が動かない |
| 小さすぎる | 採択はされるが移動量が小さい → 空間を探索しきれない |
| **目標分布の形に合っていない** | 細長い谷を縦に動こうとしても弾かれる、相関のあるパラメータをバラバラに動かす、など |

さらに SMC では **温度 $\beta$ が上がるごとに目標分布の形が変わる**（事前分布の広がり → 事後分布の鋭い山）。固定の $\Sigma$ では序盤と終盤の両方で失敗する。

#### 7.3.2 アイデア：粒子集合自身が「地形」を教えてくれる

ここでの鍵となる観察：

> **現在の粒子集合 $\{\theta^{(n)}\}$ そのものが、目標分布 $\pi_{\beta_{k+1}}$ から（近似的に）引かれたサンプル集合になっている。**

ということは、その粒子集合の **散らばり方（共分散行列）** は、目標分布の局所的な形を表す近似である。たとえば：

- 粒子が広く散らばっていれば → 目標分布も広い
- 粒子がある方向に細長く分布していれば → 目標分布もその方向に伸びている
- 粒子が小さくまとまっていれば → 目標分布も鋭く局所化している

**ならば、その共分散をそのまま提案分布の共分散に使えばよい** ——というのが Ching and Chen (2007) のアイデア。提案を「現在の粒子雲と同じ形」に揃えることで、目標分布に「沿った」効率的な動きが得られる。

#### 7.3.3 具体的な計算手順

**Step 1：重み付き平均を計算**

$$\bar{\theta} = \sum_{n=1}^{N} w^{(n)} \theta^{(n)}$$

ここで $w^{(n)}$ は粒子の正規化重み（$\sum_n w^{(n)} = 1$）。
リサンプリング直後なら全粒子等重み $w^{(n)} = 1/N$ なので、通常の標本平均と一致する。

**Step 2：重み付き経験共分散行列**

$$S = \sum_{n=1}^{N} w^{(n)} (\theta^{(n)} - \bar{\theta})(\theta^{(n)} - \bar{\theta})^\top \in \mathbb{R}^{d \times d}$$

これは「粒子雲を中心から見たときの広がり方と相関構造」を表す $d \times d$ 行列。

**Step 3：スケールと正則化**

$$\Sigma_{\text{prop}} = b^2 \cdot S + \varepsilon I$$

- **$b$（スケール係数）**：粒子雲の広がりに対する「歩幅」。実装では $b = 0.2$ → 提案の標準偏差は粒子雲の標準偏差の 20%、分散比では $b^2 = 4\%$。
- **$\varepsilon I$（正則化項）**：$\varepsilon = 10^{-6}$。粒子が縮退して $S$ が特異になっても逆行列・コレスキー分解が成立するように。

**Step 4：提案分布の決定**

各粒子について、独立に：

$$\delta^{(n)} \sim \mathcal{N}(0, \Sigma_{\text{prop}}), \qquad \theta^{*(n)} = \theta^{(n)} + \delta^{(n)}$$

つまり、各粒子から見た提案は **「現在位置を中心とする多変量ガウス」**：

$$q(\theta^* \mid \theta) = \mathcal{N}\!\bigl(\theta^* \,\big|\, \theta,\ \Sigma_{\text{prop}}\bigr)$$

**全粒子で共通の $\Sigma_{\text{prop}}$ を使う点に注意**（粒子ごとには変えない）。なぜなら粒子雲全体が目標分布の形を表しているので、その「形」を全粒子共通の歩幅情報として使う方が統計的に安定する。

#### 7.3.4 コードの対応

```python
# proposal.py: ChingAndChenProposal
def cov_proposal(self, pop, weights):
    w = nan_to_num(weights, nan=0.0, posinf=0.0, neginf=0.0)
    if w.sum() <= 0 or not isfinite(w.sum()):
        w = full_like(w, 1.0 / len(w))   # 重み崩壊時は一様にフォールバック
    w = w / w.sum()                       # 正規化

    mu = (pop * w.unsqueeze(1)).sum(dim=0)       # Step 1: 重み付き平均
    Xm = pop - mu
    cov = Xm.t().mm(torch.diag(w)).mm(Xm)        # Step 2: 重み付き共分散
    cov = cov * (self.b ** 2) + eps * eye(d)     # Step 3: スケール + 正則化
    return cov                                    # Σ_prop

def __call__(self, particles):
    cov = self.cov_proposal(particles.pop, particles.weights)
    mvn = MultivariateNormal(zeros(d), covariance_matrix=cov)
    return particles.pop + mvn.sample((N,))       # Step 4: θ + δ
```

実装ファイルは [proposal.py](../src/lsbi_smc/smc/proposal.py)。

#### 7.3.5 なぜこれが SMC で効くのか

**(a) 温度に応じて自動でスケールが追従する**

| 段階 | 粒子雲の状態 | $S$ | $\Sigma_{\text{prop}} = b^2 S$ |
|------|-------------|-----|-------------------------------|
| 序盤 ($\beta \approx 0$) | 事前分布の広がり（広い） | 大 | 大 → 大胆な探索 |
| 中盤 | 山が見えてきて部分的に集中 | 中 | 中 |
| 終盤 ($\beta \approx 1$) | 事後分布の鋭い山に集中 | 小 | 小 → 細かい微調整 |

固定共分散だとどこかで必ず破綻するが、毎ステップで $S$ を計算し直すことで自動で歩幅を調整できる。

**(b) 異方性（パラメータ間の相関）も捉える**

事後分布が「$k_1$ と $k_2$ が正に相関」のような形だと、等方ガウス提案では非効率（相関方向に動きにくい）。Ching & Chen は粒子雲の共分散をそのまま使うので、相関方向に伸びた提案が自然に得られる。

**(c) 提案が対称：MH の採択比がシンプル**

$\Sigma_{\text{prop}}$ は $\theta$ に依存しない（粒子雲全体から決まる定数行列）ので：

$$q(\theta^* \mid \theta) = \mathcal{N}(\theta^* \mid \theta, \Sigma_{\text{prop}}) = \mathcal{N}(\theta \mid \theta^*, \Sigma_{\text{prop}}) = q(\theta \mid \theta^*)$$

つまり提案は **対称**。これにより §7.2 の MH 採択比で提案密度の比 $q(\theta|\theta^*) / q(\theta^*|\theta) = 1$ となり、計算する必要がない。

#### 7.3.6 $b = 0.2$ という値について

理論的には、ガウス標的に対する RW-MH の最適スケールは $b^2 = 2.38^2/d$ 程度（Roberts–Gelman–Gilks 1997 の最適採択率 ≈ 0.234 を達成する値）と知られている。本実装の $d = 4$ では $b \approx 1.19$ になるが、実際には：

- 中間分布は完全なガウスではない（複数モードや裾を持つ）
- 採択率と多様性回復速度のバランスは問題依存
- $b$ を小さめにとると採択率が上がる代わりに 1 ステップの移動量が減る

ため、実用では **$b = 0.1$ – $0.3$ あたりが経験的に良好**で、本実装は $b = 0.2$ を採用している。

### 7.4 HMC カーネル（代替）

[kernel.py:53-125](../src/lsbi_smc/smc/kernel.py#L53-L125) の `HMCKernel` は勾配情報を使う代替案：

- ハミルトニアン $H(\theta, p) = U(\theta) + K(p)$、$U = -\log \pi_{\beta}(\theta)$
- リープフロッグ積分で位相空間を進める
- 採択比は $\exp(-\Delta H)$

`torch.autograd.grad` で MVAE エンコーダの勾配を直接計算するため、
潜在空間尤度の構造を活用できる。本実装では RW-MH を主に使用しているが、
将来的に高次元化したときの選択肢として用意されている。

---

## 8. 実装との対応

### 8.1 `SMC.run()` の構造

メインループは [smc.py:180-234](../src/lsbi_smc/smc/smc.py#L180-L234)：

```python
def run(self, ess_tar_ratio=0.8, t_max=100, mcmc_iter=1):
    ess_tar = ess_tar_ratio * self.pop_size  # 目標 ESS = 0.8 × N

    while self.q[-1] < self.q_tar and t < t_max:
        # ─── Step 1: 次の β を決定 ─────────────────
        lp_np = self.particles.lp.detach().cpu().numpy()
        q_new = _find_next_q(self.q[-1], self.q_tar, lp_np, ess_tar)
        self.q.append(q_new)

        # ─── Step 2: 重み計算 → リサンプリング ─────
        self.particles.resample(self.q[-1] - self.q[-2])  # Δβ で重み付け
        self.particles.weights = torch.full(...)          # 重みを 1/N にリセット

        # ─── Step 3: MCMC ムーブを mcmc_iter 回 ────
        for _ in range(mcmc_iter):
            pop_new, lp_new, accept = self.kernel(
                self.particles, self.q[-1], self.prior, self.likelihood
            )
            self.particles.replace(accept, pop_new, lp_new)

        self.pops.append(self.particles.pop.detach())
```

### 8.2 モジュールごとの担当

| モジュール | クラス/関数 | 役割 |
|-----------|-----------|------|
| [smc.py](../src/lsbi_smc/smc/smc.py) | `SMC` | メインループ |
| [smc.py](../src/lsbi_smc/smc/smc.py) | `Particles` | 粒子集合・重みの管理 |
| [smc.py](../src/lsbi_smc/smc/smc.py) | `_find_next_q` | 二分探索で $\Delta\beta$ 決定 |
| [smc.py](../src/lsbi_smc/smc/smc.py) | `ess` / `_ess_from_lp` | ESS 計算 |
| [kernel.py](../src/lsbi_smc/smc/kernel.py) | `RWMetropolisKernel` | RW-MH ムーブ |
| [kernel.py](../src/lsbi_smc/smc/kernel.py) | `HMCKernel` | HMC ムーブ |
| [proposal.py](../src/lsbi_smc/smc/proposal.py) | `ChingAndChenProposal` | 適応的提案共分散 |
| [prior.py](../src/lsbi_smc/smc/prior.py) | `HierarchicalPrior` | 事前分布の評価・サンプリング |
| [variables.py](../src/lsbi_smc/smc/variables.py) | `Normal`, `Uniform`, ... | 個別の確率変数 |

---

## 9. 事前分布と変数クラス

### 9.1 `HierarchicalPrior`

複数の確率変数を DAG として束ね、結合事前分布として扱う。
[variables.py](../src/lsbi_smc/smc/variables.py) の各変数クラス（`Normal`, `Uniform`, `HalfNormal`, `Laplace`, `Exponential`）が以下のインタフェースを提供：

- `sample(n)`：$n$ サンプルを引く
- `lp(values)`：対数事前確率
- `check_support(values)`：サポート内か（範囲外の提案は MH で自動棄却）

### 9.2 inference.py での実例

```python
# inference.py:97-98
variables = [Normal(label, Constant(0.0), Constant(1.0)) for label in k_labels]
prior = HierarchicalPrior(variables)
```

ここでは **無制約空間** での事前分布として標準正規 $\mathcal{N}(0,1)$ を使っている。
これは物理空間で見れば $[L,U]$ 上の一様事前分布と等価
（CDF 変換、[06 章 §6](06_latent_likelihood.md)）。

---

## 10. 結果の読み方

最終粒子集合 `smc.pops[-1]` が事後分布のサンプル。
要約統計は [smc.py:240-254](../src/lsbi_smc/smc/smc.py#L240-L254) の `summary()` で出力される：

```python
def summary(self):
    pop = self.pops[-1].detach().cpu().numpy()
    return pd.DataFrame({
        'name': self.prior.names,
        'mean': np.mean(pop, axis=0),                # 事後平均
        'sd'  : np.std (pop, axis=0),                # 事後標準偏差
        'q05' : np.percentile(pop,  5, axis=0),      # 5パーセンタイル
        'q25' : np.percentile(pop, 25, axis=0),
        'q50' : np.percentile(pop, 50, axis=0),      # 中央値
        'q75' : np.percentile(pop, 75, axis=0),
        'q95' : np.percentile(pop, 95, axis=0),      # 95パーセンタイル
    })
```

注意：`pop` は無制約空間 $\theta_{\text{latent}}$ のまま。物理スケールに戻す変換は
[inference.py:123-125](../src/lsbi_smc/example_shear4dof/inference.py#L123-L125) で行う
（[06 章 §6.4](06_latent_likelihood.md)）。

---

## 11. SMC vs NUTS（論文 Table 2 より）

| アルゴリズム | 粒子数 $N_s$ | 尤度評価回数 | MMD（低いほど良い） | 計算時間 |
|------------|-----------|------------|-----------------|---------|
| SMC | 2000 | 362,000 | $0.051 \pm 0.019$ | **0.8 秒** |
| NUTS | -        | 558,871    | $0.629 \pm 0.159$ | 1782 秒 |

**SMC の優位性が出る理由：**

1. **GPU 並列性**：粒子は独立に評価できるので、全 $N$ 粒子を 1 度のフォワードで処理できる
2. **多峰性への耐性**：温度が低いうちに広く探索しているので、NUTS のような単一連鎖が陥る局所解に居着かない
3. **適応的 $\Delta\beta$**：必要なステップ数を計算機が自動で決めるので、無駄なイテレーションが少ない

---

## 12. 章のまとめ

SMC は、02 章で紹介した 3 つの要素を **互いの欠点を補い合う形** で組み合わせている：

| 要素 | 単独での弱点 | SMC で補う方法 |
|------|------------|---------------|
| 重点サンプリング | 提案と目的が違いすぎると ESS 崩壊 | $\Delta\beta$ を ESS が保てる範囲に制限 |
| リサンプリング | 粒子が重複し多様性が落ちる | 後続の MCMC ムーブで揺らがせる |
| MCMC | 単独では多峰性で詰まる | テンパリングで滑らかに進めるので局所解に陥らない |

そして本実装では、これらが [06 章](06_latent_likelihood.md) の **潜在空間尤度** と組み合わさることで、
推論中にシミュレーターを 1 度も呼ばずに事後分布が得られる、というアーキテクチャになっている。
