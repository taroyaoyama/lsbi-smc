# inference.py — SMC による事後分布推論

**ファイルパス**: `src/lsbi_smc/example_shear4dof/inference.py`  
**関連学術ドキュメント**: [07_smc.md](../07_smc.md), [06_latent_likelihood.md](../06_latent_likelihood.md), [00_overview.md](../00_overview.md)

---

## 概要

学習済みの MVAE を使って SMC で事後分布 $p(\theta | x_{\text{obs}})$ からサンプリングするスクリプトです。

このスクリプトが LSBI-SMC パイプラインの最終段階であり、`posterior.mat` に事後サンプルを出力します。

**実行方法**（プロジェクトルートから）：
```bash
uv run python src/lsbi_smc/example_shear4dof/inference.py
```

**前提**: `mvae_best.pth` が存在すること（`train.py` で生成）  
**出力**: `posterior.mat`（プロジェクトルート）

---

## 全体の処理フロー

```
1. グローバル定数の設定
      ↓
2. MVAE モデルのロード
      ↓
3. 合成観測データの生成
      ↓
4. LogLikelihood クラスのインスタンス化
      ↓
5. 事前分布の設定（標準正規）
      ↓
6. SMC の実行
      ↓
7. 無制約空間 → 正規化空間 → 物理パラメータへの逆変換
      ↓
8. posterior.mat に保存
```

---

## コードの詳細

### グローバル定数

```python
LLIM, ULIM = 0.33, 3.00            # 物理パラメータの下限・上限
stdnorm = dist.Normal(0.0, 1.0)    # 標準正規分布（CDF変換用）

# train.py で算出した標準化定数（ハードコード）
y_mn, y_sd = -2.1384575366973877, 2.809697389602661
```

`y_mn`, `y_sd` は学習データから計算した統計量。実際の使用では `train.py` 実行後の値をここに記録する。

### MVAE モデルのロード

```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
ndof, z_dim = 4, 8
model = MVAE(z_dim=8, ch=1, size=1024, nlabel=4, depth=1).to(device)
model.load_state_dict(
    torch.load('mvae_best.pth', map_location=device)['model_state_dict']
)
model.eval()  # Dropout / BatchNorm を評価モードに切り替え
```

`model.eval()` が重要：Dropout は推論時に無効化、BatchNorm は学習時の統計量を使用。

### 合成観測データの生成

```python
# "真値" θ = {1.0, 1.0, 1.0, 1.0} を正規化
x_obs = np.full((1, ndof), (1.0 - LLIM) / (ULIM - LLIM)).astype(np.float32)
# → x_obs = [[0.2509, 0.2509, 0.2509, 0.2509]]

# FRF 計算
y_obs = simulator(x_obs)[:, [-1], :, :]    # 屋根のみ: (1, 1, 1, 1024)

# ノイズ付加（固定シード101で再現性確保）
y_obs = y_obs + norm.rvs(size=y_obs.shape, random_state=101) * 0.20

# 標準化（train.py と同じスケール）
y_obs = (y_obs - y_mn) / y_sd
y_obs = y_obs.astype(np.float32)

# PyTorch テンソルに変換してGPUへ
y_obs_tc = torch.from_numpy(y_obs).to(device)
```

**合成観測を使う理由**: 真値が既知の場合に推定精度を定量的に評価できるため（ベンチマーク評価）。

### `LogLikelihood` クラス

```python
class LogLikelihood(MVAEBasedLogLikelihood):
    n_call: int

    def __init__(self, enc_w, enc_x, obs, device):
        super().__init__(enc_w, enc_x, obs, device)
        self.n_call = 0                     # 尤度評価回数のカウンター

    def __call__(self, theta: Tensor, alp=1.0, tau=0.00) -> Tensor:
        theta = stdnorm.cdf(theta)          # ★ CDF変換: θ_latent ∈ ℝ → θ_norm ∈ [0, 1]
        self.n_call += len(theta)           # 呼び出し回数を記録
        return super().__call__(theta, alp=1.0, tau=0.00)
```

**CDF変換（最重要）**:

本実装では $\theta$ が **3 つの空間** を行き来する：

| 空間 | 範囲 | 用途 |
|------|------|------|
| 無制約空間 $\theta_{\text{latent}}$ | $\mathbb{R}$ | SMC のサンプリング・MCMC カーネル |
| 正規化空間 $\theta_{\text{norm}}$ | $[0, 1]$ | MVAE の入力（学習時もこのスケール） |
| 物理空間 $\theta_{\text{phys}}$ | $[L, U] = [0.33, 3.00]$ | シミュレーター入力・最終出力 |

ここで使われているのは「**$\theta_{\text{latent}} \to \theta_{\text{norm}}$ への変換**」：

$$\theta_{\text{norm}} = \Phi(\theta_{\text{latent}})$$

$\Phi$ は標準正規分布の累積分布関数（CDF）。
**確率積分変換** によって、$\theta_{\text{latent}} \sim \mathcal{N}(0,1)$ なら $\Phi(\theta_{\text{latent}}) \sim \text{Uniform}(0,1)$ となる（[06_latent_likelihood.md §6.2](../06_latent_likelihood.md) 参照）。

```
θ_latent ∈ ℝ          （SMC が探索する空間。境界がないので MCMC が楽）
   ↓ stdnorm.cdf(θ)     [この __call__ メソッド]
θ_norm ∈ [0, 1]       （MVAE の enc_w に渡すスケール）
   ↓                    [後段で物理空間へ戻す: pop * (ULIM-LLIM) + LLIM]
θ_phys ∈ [0.33, 3.00] （物理パラメータ：剛性比）
```

**なぜ無制約空間でサンプリングするのか**: RW-MH の提案 $\theta' = \theta + \epsilon$ は無制約空間だと境界処理が不要で実装がシンプル。

詳細は [06_latent_likelihood.md §6](../06_latent_likelihood.md) を参照。

### 事前分布の設定

```python
k_labels = [f'k{i:02}' for i in range(1, ndof+1)]
# → ['k01', 'k02', 'k03', 'k04']

variables = [Normal(l, Constant(0.0), Constant(1.0)) for l in k_labels]
prior = HierarchicalPrior(variables)
```

SMC の探索空間は無制約（$\theta_{\text{latent}} \in \mathbb{R}^4$）なので、事前分布は $\mathcal{N}(0,1)^4$。

**事前分布の等価性**：

| 座標系 | 事前分布の表現 |
|--------|---------------|
| 無制約空間 $\theta_{\text{latent}}$ | $\mathcal{N}(0, 1)$ |
| 物理空間 $\theta_{\text{phys}}$ | $\text{Uniform}(L, U)$ |

つまり「物理空間で $[L, U]$ 上の無情報一様事前分布」を、サンプリングしやすい座標系で書き直したのが標準正規事前分布。CDF 変換の確率積分変換性質によって等価になる。

### SMC の実行

```python
proposal = ChingAndChenProposal(b=0.2)

loglikelihood = LogLikelihood(model.enc_w, model.enc_x, y_obs_tc, device)

smc1 = SMC(
    pop_size=2000,                       # 粒子数
    likelihood=loglikelihood,
    prior=prior,
    kernel=RWMetropolisKernel(proposal),
    q_tar=1.0,                           # 目標逆温度（事後分布に完全到達）
)

smc1.run(ess_tar_ratio=0.8, mcmc_iter=10)
```

| パラメータ | 値 | 意味 |
|-----------|-----|------|
| `pop_size` | 2000 | 粒子数 N |
| `ess_tar_ratio` | 0.8 | 目標ESS = 0.8 × N = 1600 |
| `mcmc_iter` | 10 | 各SMCステップでMCMCを10回実行 |
| `b` | 0.2 | Ching & Chen 提案分布のスケール係数 |

### 結果の取り出しと変換

```python
# SMC の最終状態（無制約空間での粒子）
pop = smc1.pops[-1].detach().cpu().numpy()   # shape: (2000, 4), θ_latent

# 2 段階変換: θ_latent → θ_norm → θ_phys
pop = norm.cdf(pop)                           # θ_latent ∈ ℝ → θ_norm ∈ [0, 1]
pop = pop * (ULIM - LLIM) + LLIM             # θ_norm → θ_phys ∈ [0.33, 3.00]
```

`smc1.pops[-1]` は最後の SMC ステップの粒子群 = 事後分布のサンプル（無制約空間表現）。
ここで [06_latent_likelihood.md §6.4](../06_latent_likelihood.md) で説明している 2 段階変換を逆向きに辿って物理スケールに戻す。

### 保存

```python
dic = {'pop': pop, 'n_call': loglikelihood.n_call}
io.savemat('posterior.mat', dic)
```

| キー | 形状 | 内容 |
|------|------|------|
| `pop` | `(2000, 4)` | 事後分布サンプル（物理パラメータ空間） |
| `n_call` | スカラー | 合計尤度評価回数 |

---

## 推論結果の解釈

`summary()` が自動で表示する統計量：

```
         name      mean        sd       q05       q25       q50       q75       q95
k01[0]   0.952   0.318     0.446     0.680     0.995     1.223     1.467
k02[0]   1.015   0.367     0.521     0.742     1.022     1.290     1.612
k03[0]   0.978   0.352     0.494     0.716     0.988     1.248     1.553
k04[0]   1.033   0.385     0.531     0.749     1.038     1.321     1.641
```

真値 θ = 1.0 付近に事後分布の中心が集まっていれば推定成功。等価解（異なる θ でも同じ FRF になる解）が複数存在するため、事後分布は多峰性になる場合がある。

---

## ファイル間の依存関係

```
inference.py
  ├── mvae.py        → MVAE クラス
  ├── latentlik.py   → MVAEBasedLogLikelihood
  ├── simulator.py   → Simulator
  ├── frfshearm.py   → frfshearm2
  ├── smc.py         → SMC
  ├── kernel.py      → RWMetropolisKernel
  ├── prior.py       → HierarchicalPrior
  ├── proposal.py    → ChingAndChenProposal
  └── variables.py   → Normal, Constant
```
