# コード全体のウォークスルー

## 1. プロジェクト全体の実行フロー

```
[Step 1] データセット生成
  src/lsbi_smc/example_shear4dof/create_dataset.py を実行
  → train_data.npz が生成される

[Step 2] MVAE 学習
  src/lsbi_smc/example_shear4dof/train.py を実行
  → mvae_best.pth が生成される

[Step 3] 推論
  src/lsbi_smc/example_shear4dof/inference.py を実行
  → posterior.mat が生成される（事後分布サンプル＋尤度評価回数）
```

---

## 2. Step 1: データセット生成（`create_dataset.py`）

```python
# ---- シミュレーター定義 ----
def fun(x):
    return frfshearm2(
        x * 1000,            # x は [0.33, 3.00] の範囲の正規化パラメータ
                             # 実際の剛性 k_i = x_i × 1000 N/m
        ms = 1.0,            # 各階質量 1 kg
        zeta = 0.02,         # 減衰比 2%
        dlf = 0.020,         # 周波数刻み 0.02 Hz
        fmax = 20.48,        # 最大周波数 20.48 Hz → 1024 点
        damping = 'rayleigh',
        omega_target = np.array([1.0, 20.0]) * 2 * np.pi
    )

simulator = Simulator(fun, lims=[LLIM, ULIM], workers=2)

# ---- ラテンハイパーキューブサンプリング ----
ndof = 4
n_sim = 100000  # 10万サンプル
sampler = qmc.LatinHypercube(d=ndof)
x_sim = sampler.random(n_sim)   # shape: (100000, 4) ← [0,1] の範囲
y_sim = simulator(x_sim)        # shape: (100000, 4, 1, 1024) ← FRF（全4階）

# ---- ノイズ付加 ----
noise_level = 0.20
y_sim_n = y_sim + noise_level * norm.rvs(size=y_sim.shape)

np.savez('train_data.npz', llim=LLIM, ulim=ULIM,
         x_sim=x_sim, y_sim=y_sim, y_sim_n=y_sim_n)
```

### Simulator クラス（`simulator/simulator.py`）
```python
class Simulator:
    def __call__(self, theta):
        # theta を [LLIM, ULIM] にスケール変換
        theta = theta * (self.ulim - self.llim) + self.llim
        # スレッドプールで並列実行
        with ThreadPool(processes=self.workers) as pool:
            sims_list = pool.map(self.fun, list(theta), self.chunksize)
        # (n, n_floors, 1, n_freq) の形状に整形
        sims = np.stack(sims_list, axis=0)
        sims = sims[:, :, None, :].astype(np.float32, copy=False)
        return sims
```

---

## 3. Step 2: MVAE 学習（`train.py`）

### データ前処理
```python
# 最後のチャンネル（屋根 RF）のみ使用
ch = [-1]
y_sim_tensor   = torch.from_numpy(y_sim[:,ch,:,:])    # クリーン: (100000,1,1,1024)
y_sim_n_tensor = torch.from_numpy(y_sim_n[:,ch,:,:])  # ノイズあり

# 標準化
y_mn, y_sd = y_sim_tensor.mean(), y_sim_tensor.std()
y_sim_tensor   = (y_sim_tensor   - y_mn) / y_sd
y_sim_n_tensor = (y_sim_n_tensor - y_mn) / y_sd

# 学習/検証分割（9:1）
n_train = int(0.9 * n_total)
train_dataset, valid_dataset = random_split(
    dataset, [n_train, n_valid], generator=torch.Generator().manual_seed(42)
)
```

**なぜ屋根だけを使うのか**：論文のベンチマーク設定（屋根のみ観測可能な想定）。

### モデル定義・コンパイル・学習ループ
```python
_mvae = MVAE(z_dim=8, ch=1, size=1024, nlabel=4, depth=1).to(device)
model = torch.compile(_mvae)   # XLA/Triton バックエンドで速度向上

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# AMP（Automatic Mixed Precision）: CUDA 利用時のみ有効
use_amp = device.type == "cuda"
scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

for epoch in range(epochs):
    model.train()
    for x, y, yn in train_loader:
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=use_amp):
            trl, kll, rcl = model.loss(yn, y, x, alp1=5.0)
        scaler.scale(trl).backward()
        scaler.step(optimizer)
        scaler.update()

    # 検証損失で早期終了（patience=20）
    if vl_loss < best_vl_loss:
        best_vl_loss = vl_loss
        epochs_no_improve = 0
        torch.save({"model_state_dict": _mvae.state_dict(), ...}, "mvae_best.pth")
    else:
        epochs_no_improve += 1
    if epochs_no_improve >= patience:
        break
```

**注意**: `torch.save` で保存するのは `_mvae`（compile前）の state_dict。

### 損失計算の内部（`MVAE.loss` メソッド）
```python
def loss(self, xo, xi, w, alp1=1.0, alp2=10.0, alp3=10.0):
    z1, mu1, var1, z2, mu2, var2 = self.encode(xi, w)
    # z1, mu1, var1 = enc_x(xi)  FRF エンコーダ出力
    # z2, mu2, var2 = enc_w(w)   パラメータエンコーダ出力

    x_mu1, x_var1, x_mu2, x_var2 = self.decode(z1, z2)

    KL1 = gauss_unitgauss_kl(mu1, var1)                    # enc_x → N(0,I)
    KL2 = gauss_unitgauss_kl(mu2, var2)                    # enc_w → N(0,I)
    KL_x1x2 = gauss_gauss_kl(mu1, var1, mu2, var2)         # enc_x → enc_w
    KL_x2x1 = gauss_gauss_kl(mu2, var2, mu1, var1)         # enc_w → enc_x

    rec_xx = rec_loss_norm4D(xo, x_mu1, x_var1)   # xo=ノイズあり FRF が目標
    rec_wx = rec_loss_norm4D(xo, x_mu2, x_var2)

    loss = KL1 + KL2 + alp1*(KL_x1x2 + KL_x2x1) + alp2*rec_xx + alp3*rec_wx
    return loss, (KL1 + KL2 + KL_x1x2 + KL_x2x1), (rec_xx + rec_wx)
```

---

## 4. Step 3: 推論（`inference.py`）

### 合成観測データの作成
```python
# "真値" θ = {1, 1, 1, 1} に対応する正規化値
x_obs = np.full((1, ndof), (1.0 - LLIM) / (ULIM - LLIM)).astype(np.float32)
y_obs = simulator(x_obs)[:, [-1], :, :]       # 屋根の FRF
y_obs = y_obs + norm.rvs(size=y_obs.shape, random_state=101) * 0.20
y_obs = (y_obs - y_mn) / y_sd                 # 標準化（学習時と同じスケール）
```

### 尤度クラス
```python
class LogLikelihood(MVAEBasedLogLikelihood):
    def __init__(self, enc_w, enc_x, obs, device):
        super().__init__(enc_w, enc_x, obs, device)
        self.n_call = 0          # 尤度評価回数カウンター

    def __call__(self, theta):
        theta = stdnorm.cdf(theta)   # 標準正規 CDF で [0,1] に変換
        self.n_call += len(theta)
        return super().__call__(theta, alp=1.0, tau=0.00)
```

**CDF 変換の理由**：SMC は $\theta_{\text{latent}} \in \mathbb{R}$ で探索（事前分布 $\mathcal{N}(0,1)$）し、
$\theta_{\text{physical}} = \Phi(\theta_{\text{latent}}) \in [0, 1]$ に変換してシミュレーターへ渡す。

### 事前分布の設定
```python
k_labels = [f"k{i:02d}" for i in range(1, ndof + 1)]
variables = [Normal(l, Constant(0.0), Constant(1.0)) for l in k_labels]
prior = HierarchicalPrior(variables)
```

### SMC の実行
```python
proposal = ChingAndChenProposal(b=0.2)
smc1 = SMC(
    pop_size=2000,
    likelihood=loglikelihood,
    prior=prior,
    kernel=RWMetropolisKernel(proposal),
    q_tar=1.0,
)
smc1.run(ess_tar_ratio=0.8, mcmc_iter=10)   # 各 SMC ステップで MCMC を 10 回実行
```

### 結果の取り出しと保存
```python
pop = smc1.pops[-1].detach().cpu().numpy()  # 潜在空間でのサンプル (2000, 4)
pop = norm.cdf(pop)                          # [0,1] に変換
pop = pop * (ULIM - LLIM) + LLIM            # 物理パラメータ [0.33, 3.00]

io.savemat("posterior.mat", {"pop": pop, "n_call": loglikelihood.n_call})
```

---

## 5. データ形状の整理

| 変数 | 形状 | 説明 |
|------|------|------|
| `x_sim` | (100000, 4) | 正規化パラメータ [0,1] |
| `y_sim` | (100000, 4, 1, 1024) | FRF（全4階、ノイズなし） |
| `y_sim_n` | (100000, 4, 1, 1024) | FRF（全4階、ノイズあり） |
| 学習時 `y` | (batch, 1, 1, 1024) | 屋根 FRF（クリーン） |
| 学習時 `yn` | (batch, 1, 1, 1024) | 屋根 FRF（ノイズあり） |
| `mu_obs` | (1, 8) | 観測 FRF の潜在平均 |
| `vr_obs` | (1, 8) | 観測 FRF の潜在分散 |
| SMC `pop` | (2000, 4) | 事後サンプル（潜在空間） |

---

## 6. 確率変数クラス（`smc/variables.py`）

`HierarchicalPrior` はこれらのクラスを DAG として組み合わせて複雑な事前分布を表現する。

| クラス | 概要 |
|-------|------|
| `Constant(v)` | スカラー定数（1次元） |
| `ConstantVector(v)` | ベクトル定数（多次元） |
| `Uniform(name, lower, upper)` | 一様分布 |
| `Normal(name, mu, sg)` | 正規分布 |
| `HalfNormal(name, sg)` | 半正規分布（`Normal` のサブクラス） |
| `Laplace(name, mu, b)` | ラプラス分布 |
| `Exponential(name, rate)` | 指数分布 |

各クラスは共通インタフェース `sample(n)` / `lp(values)` / `check_support(values)` を持つ。

---

## 7. 各クラスの責務まとめ

```
MVAEBasedLogLikelihood（latentlik.py）
  ├── enc_x: FRFエンコーダ（観測データ→潜在空間）
  ├── enc_w: パラメータエンコーダ（θ→潜在空間）
  ├── mu_obs, vr_obs: 観測データの潜在表現（コンストラクタで事前計算）
  └── __call__(theta, alp, tau): θを受け取り対数尤度を返す

SMC（smc.py）
  ├── Particles: 粒子の状態管理（pop, lp, weights）
  ├── _find_next_q: ESS基準でβを二分探索（モジュール関数）
  └── run(mcmc_iter=N): リサンプリング + MCMC×N回ムーブのループ

HierarchicalPrior（prior.py）
  ├── sample(n): 事前分布からのサンプリング
  ├── lp(theta): 対数事前確率
  └── check_support(theta): サポート内チェック

RWMetropolisKernel（kernel.py）
  └── __call__(): 提案→採択比計算→採択/棄却

HMCKernel（kernel.py）
  └── __call__(): リープフロッグ積分 + MH受理判定

ChingAndChenProposal（proposal.py）
  └── __call__(): 重み付き共分散に基づくガウス提案
```

---

## 8. 数値的安定性への配慮

```python
# smc.py: ログ尤度の NaN/inf 処理
z = torch.nan_to_num(z, neginf=-1e30, posinf=1e30)

# smc.py: 重みがゼロになった場合の対処
w = torch.full_like(w, 1.0 / len(w)) if not torch.isfinite(s) or s <= 0 else w / s

# latentlik.py: 分散の下限クリッピング
vr1 = torch.clamp(vr_obs, min=eps)
vr2 = torch.clamp((1 + alp) * vr_sim + tau, min=eps)

# proposal.py: 共分散行列の正則化
cov = cov * (self.b ** 2) + eps * torch.eye(X.shape[1], device=device)
```

---

## 9. 依存関係図

```
inference.py
    ↓ 使用
    ├── MVAE (mvae.py)
    │     ├── Encoder      ← FRF データを潜在空間へ（ConvResNet）
    │     ├── Encoder_w    ← パラメータを潜在空間へ（FC ResNet）
    │     └── Decoder      ← 潜在空間から FRF へ（ConvResNet）
    │
    ├── MVAEBasedLogLikelihood (latentlik.py)
    │     └── latent_space_loglik ← ガウス積分による解析的計算
    │
    ├── SMC (smc.py)
    │     ├── Particles ← 粒子の状態管理
    │     └── _find_next_q ← β の二分探索
    │
    ├── RWMetropolisKernel (kernel.py)
    │     └── ChingAndChenProposal (proposal.py)
    │
    └── HierarchicalPrior (prior.py)
          └── Normal, Uniform, HalfNormal, Laplace, Exponential (variables.py)
```
