# LSBI-SMC 全体概要

## このプロジェクトは何をするものか

本プロジェクトは、**地震を受けるRC（鉄筋コンクリート）建物の有限要素（FE）モデルのパラメータを、観測データから確率論的に推定する**フレームワークの実装です。

論文タイトル：  
> "Finite element model updating of building structures under seismic excitation: A parallelized latent space–based Bayesian framework"  
> （東京大学 矢尾山 太郎 ほか、Taisei Corporation との共同研究）

---

## 一言で言うと

「観測した建物の振動データから、建物の剛性（どれだけ硬いか）を確率的に推定する」

---

## なぜ難しいのか

### 問題1: 高次元データ
周波数応答関数（FRF）は 1024 点の時系列データ。この高次元データに対して尤度（データの起こりやすさ）を定義・計算するのは困難。

### 問題2: シミュレーターのコスト
FE解析は計算コストが高い。ベイズ推定には通常、何十万回もシミュレーターを呼ぶ必要がある。

### 問題3: 多峰性
同じ屋根の振動応答を再現するパラメータの組み合わせが複数存在する（等価解）。

---

## 提案手法: LSBI-SMC

**L**atent space-based **B**ayesian **I**nference + **S**equential **M**onte **C**arlo

```
[Step 1] データセット生成
  事前分布 → パラメータサンプル θ → FEシミュレーター → 応答データ x

[Step 2] MVAEの学習
  (θ, x) ペアを使って、Multimodal Variational Autoencoder を学習
  → 潜在空間 z でθとxが共通表現を持つ

[Step 3] 推論 (SMC)
  実観測 xobs と潜在空間を使った尤度近似 L̂(θ; xobs) を計算
  → Sequential Monte Carlo で事後分布 p(θ|xobs) からサンプリング
```

---

## ファイル構成

```
lsbi-smc/
├── papers/
│   └── 2603_ned260329.pdf                  # 論文
│
├── src/lsbi_smc/                            # Pythonパッケージ
│   ├── example_shear4dof/                   # ベンチマーク例（4自由度せん断建物）
│   │   ├── frfshearm.py                     # FRF計算シミュレーター
│   │   ├── create_dataset.py                # 学習データ生成
│   │   ├── train.py                         # MVAE学習（AMP・早期終了対応）
│   │   ├── inference.py                     # 推論（SMC実行）
│   │   └── mvae.py                          # MVAE・VAEネットワーク定義
│   │
│   ├── likelihood/
│   │   └── latentlik.py                     # 潜在空間ベース尤度
│   │
│   ├── simulator/
│   │   └── simulator.py                     # シミュレーターのラッパー
│   │
│   └── smc/
│       ├── smc.py                           # SMCサンプラー本体
│       ├── kernel.py                        # MCMCカーネル (RW-MH, HMC)
│       ├── prior.py                         # 事前分布（HierarchicalPrior）
│       ├── proposal.py                      # 提案分布（Ching & Chen 2007）
│       └── variables.py                     # 確率変数クラス群
│
├── train_data.npz                           # 生成済み学習データ
└── mvae_best.pth                            # 学習済みモデル重み
```

---

## 各解説ドキュメント

| ファイル | 内容 |
|---------|------|
| [01_math_prerequisites.md](01_math_prerequisites.md) | ベイズ推論、確率論の基礎 |
| [02_ml_prerequisites.md](02_ml_prerequisites.md) | 深層学習・VAEの基礎 |
| [03_structural_engineering.md](03_structural_engineering.md) | 構造工学の背景（FEM、FRF、振動理論） |
| [04_mvae.md](04_mvae.md) | Multimodal VAE の詳細 |
| [05_latent_likelihood.md](05_latent_likelihood.md) | 潜在空間ベース尤度近似の導出 |
| [06_smc.md](06_smc.md) | Sequential Monte Carlo サンプラー |
| [07_code_walkthrough.md](07_code_walkthrough.md) | コード全体のウォークスルー |
