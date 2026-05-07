# コード全体のウォークスルー

このドキュメントはコードベース全体の俯瞰図です。各ファイルの詳細解説は [docs/code/](code/) 配下の個別ドキュメントを参照してください。初学者向けの学習順序は [code/tutorial.md](code/tutorial.md) で案内しています。

---

## 1. 実行フロー（3ステップ）

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

[補助] 可視化
  src/lsbi_smc/example_shear4dof/plot_posterior.py を実行
  → posterior_plot.png が生成される
```

各スクリプトはプロジェクトルートから `uv run python <path>` で実行します（相対パスのファイル入出力のため）。

---

## 2. ファイル一覧と役割

### Step 1: データセット生成

| ファイル | 役割 | 詳細解説 |
|---------|------|---------|
| `frfshearm.py` | N自由度せん断建物のFRFを物理計算 | [code/frfshearm.md](code/frfshearm.md) |
| `simulator.py` | シミュレーターの並列実行ラッパー | [code/simulator.md](code/simulator.md) |
| `create_dataset.py` | LHSサンプリング + FRF生成 + ノイズ付加 | [code/create_dataset.md](code/create_dataset.md) |

### Step 2: MVAE 学習

| ファイル | 役割 | 詳細解説 |
|---------|------|---------|
| `mvae.py` | MVAE/VAE のネットワーク定義と損失関数 | [code/mvae.md](code/mvae.md) |
| `train.py` | 学習ループ、早期終了、モデル保存 | [code/train.md](code/train.md) |

### Step 3: 推論（SMC）

| ファイル | 役割 | 詳細解説 |
|---------|------|---------|
| `latentlik.py` | 潜在空間ベースの対数尤度の計算 | [code/latentlik.md](code/latentlik.md) |
| `inference.py` | SMCの設定と実行のメインスクリプト | [code/inference.md](code/inference.md) |
| `smc.py` | SMCサンプラー本体（粒子・ESS・β探索） | [code/smc.md](code/smc.md) |
| `kernel.py` | MCMCカーネル（RW-MH, HMC） | [code/kernel.md](code/kernel.md) |
| `proposal.py` | Ching & Chen 提案分布 | [code/proposal.md](code/proposal.md) |
| `prior.py` | 階層的事前分布 | [code/prior.md](code/prior.md) |
| `variables.py` | 確率変数クラス群（Normal, Uniformなど） | [code/variables.md](code/variables.md) |

### 補助：可視化

| ファイル | 役割 | 詳細解説 |
|---------|------|---------|
| `plot_posterior.py` | 事後分布のコーナープロット生成 | [code/plot_posterior.md](code/plot_posterior.md) |

---

## 3. 学術的背景の対応

各コードは数学的・工学的な理論にもとづいて実装されています。学術解説と対応するコードは以下の通り：

| 学術ドキュメント | 対応するコード |
|----------------|--------------|
| [00_overview.md](00_overview.md) — 全体概要 | プロジェクト全体 |
| [01_linear_algebra.md](01_linear_algebra.md) — 線形代数・NumPy/SciPy | [frfshearm.py](code/frfshearm.md), [simulator.py](code/simulator.md), 各コード共通 |
| [02_probability.md](02_probability.md) — ベイズ統計の基礎 | [variables.py](code/variables.md), [prior.py](code/prior.md), [smc.py](code/smc.md) |
| [03_ml_prerequisites.md](03_ml_prerequisites.md) — 深層学習・VAEの基礎 | [mvae.py](code/mvae.md), [train.py](code/train.md) |
| [04_structural_engineering.md](04_structural_engineering.md) — 構造工学の背景 | [frfshearm.py](code/frfshearm.md), [simulator.py](code/simulator.md), [create_dataset.py](code/create_dataset.md) |
| [05_mvae.md](05_mvae.md) — Multimodal VAE | [mvae.py](code/mvae.md), [train.py](code/train.md) |
| [06_latent_likelihood.md](06_latent_likelihood.md) — 潜在空間ベース尤度 | [latentlik.py](code/latentlik.md), [inference.py](code/inference.md) |
| [07_smc.md](07_smc.md) — Sequential Monte Carlo | [smc.py](code/smc.md), [kernel.py](code/kernel.md), [proposal.py](code/proposal.md), [prior.py](code/prior.md), [variables.py](code/variables.md) |

---

## 4. 依存関係図

```
inference.py（推論のエントリポイント）
  ├── mvae.py            ← MVAE クラス・エンコーダ・デコーダ
  ├── latentlik.py       ← MVAEBasedLogLikelihood
  │     └── （内部で）latent_space_loglik
  ├── simulator.py       ← Simulator（合成観測の生成）
  │     └── frfshearm.py ← frfshearm2（FRF計算）
  ├── smc.py             ← SMC サンプラー
  │     └── （内部で）Particles、_find_next_q、ess
  ├── kernel.py          ← RWMetropolisKernel、HMCKernel
  │     └── proposal.py  ← ChingAndChenProposal
  └── prior.py           ← HierarchicalPrior
        └── variables.py ← Constant, Normal, Uniform, ...
```

```
train.py（学習のエントリポイント）
  └── mvae.py            ← MVAE クラス
```

```
create_dataset.py（データ生成のエントリポイント）
  ├── simulator.py       ← Simulator
  └── frfshearm.py       ← frfshearm2
```

---

## 5. データ形状の整理

| 変数 | 形状 | 説明 |
|------|------|------|
| `x_sim` | (100000, 4) | 正規化パラメータ \[0, 1\] |
| `y_sim` | (100000, 4, 1, 1024) | FRF（全4階、ノイズなし、対数絶対値） |
| `y_sim_n` | (100000, 4, 1, 1024) | FRF（全4階、ノイズあり） |
| 学習時 `y` | (batch, 1, 1, 1024) | 屋根 FRF（クリーン、標準化済み） |
| 学習時 `yn` | (batch, 1, 1, 1024) | 屋根 FRF（ノイズあり、標準化済み） |
| `mu_obs`, `vr_obs` | (1, 8) | 観測 FRF の潜在表現（平均・分散） |
| `mu_sim`, `vr_sim` | (n, 8) | パラメータの潜在表現（平均・分散） |
| SMC `pop` | (2000, 4) | 事後サンプル（潜在空間または物理空間） |

詳細は各コードドキュメントを参照。

---

## 6. クラス・関数の責務まとめ

| クラス・関数 | 場所 | 責務 |
|------------|------|------|
| `Simulator` | [simulator.py](code/simulator.md) | 物理シミュレーターの並列ラッパー |
| `frfshearm2` | [frfshearm.py](code/frfshearm.md) | N-DOFせん断建物のFRF計算 |
| `Encoder` | [mvae.py](code/mvae.md) | FRF→潜在空間の畳み込みResNet |
| `EncoderW` | [mvae.py](code/mvae.md) | パラメータ→潜在空間の全結合ResNet |
| `Decoder` | [mvae.py](code/mvae.md) | 潜在空間→FRFの転置畳み込みResNet |
| `MVAE` | [mvae.py](code/mvae.md) | エンコーダ群とデコーダの統合 |
| `MVAEBasedLogLikelihood` | [latentlik.py](code/latentlik.md) | MVAEから対数尤度を計算 |
| `latent_space_loglik` | [latentlik.py](code/latentlik.md) | ガウス積分による解析的な対数尤度 |
| `Particles` | [smc.py](code/smc.md) | 粒子の状態（pop, lp, weights）管理 |
| `SMC` | [smc.py](code/smc.md) | SMCサンプラー本体 |
| `_find_next_q` | [smc.py](code/smc.md) | ESS基準でのβ二分探索 |
| `RWMetropolisKernel` | [kernel.py](code/kernel.md) | RW-MH MCMC カーネル |
| `HMCKernel` | [kernel.py](code/kernel.md) | Hamiltonian MC カーネル（拡張用） |
| `ChingAndChenProposal` | [proposal.py](code/proposal.md) | 重み付き共分散ベースの適応的提案 |
| `HierarchicalPrior` | [prior.py](code/prior.md) | DAG構造の階層事前分布 |
| `Constant`, `Normal` ほか | [variables.py](code/variables.md) | 確率変数ノード |

---

## 7. 数値的安定性への配慮

各箇所で施されている数値安定化の処理：

```python
# smc.py: 対数尤度の NaN/inf 処理
z = torch.nan_to_num(z, neginf=-1e30, posinf=1e30)

# smc.py: 重みがゼロになった場合のフォールバック
w = torch.full_like(w, 1.0 / len(w)) if not torch.isfinite(s) or s <= 0 else w / s

# latentlik.py: 分散の下限クランプ
vr1 = torch.clamp(vr_obs, min=eps)
vr2 = torch.clamp((1 + alp) * vr_sim + tau, min=eps)

# proposal.py: 共分散行列の正則化
cov = cov * (self.b ** 2) + eps * torch.eye(X.shape[1], device=device)

# mvae.py: 損失関数中の log への eps 加算
_kl = torch.log(_var2 + eps) - torch.log(var1 + eps) + ...
```

詳細は各コードドキュメントを参照。

---

## 8. 学習の手引き

このコードベースを理解するための学習順序は [code/tutorial.md](code/tutorial.md) を参照してください。背景知識から実装の細部まで段階的に案内しています。
