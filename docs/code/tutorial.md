# LSBI-SMC コードリーディングのチュートリアル

このドキュメントは、LSBI-SMC のコードベースを **初学者が無理なく読み進められる順序** で案内するチュートリアルです。各ステップで「読むべきドキュメント」と「読むべきコード」をリンクで示しています。

---

## 学習を始める前に

### このプロジェクトは何をするものか

> 観測した建物の振動データから、建物の剛性パラメータを確率的に推定する。

詳細: [00_overview.md](../00_overview.md)

### 全体像のスナップショット

```
[Step 1] create_dataset.py  →  train_data.npz       (シミュレーションでデータ生成)
[Step 2] train.py           →  mvae_best.pth        (MVAE の学習)
[Step 3] inference.py       →  posterior.mat        (SMC で事後分布推定)
[補助]   plot_posterior.py  →  posterior_plot.png  (事後分布の可視化)
```

---

## 学習ロードマップ

このプロジェクトを理解するためには、**3つの分野の知識** を組み合わせる必要があります：

```
構造工学（建物のFRF）
        +
深層学習（VAE）            →  LSBI-SMC
        +
ベイズ推定（SMC）
```

それぞれの分野の前提知識を確認しながら、段階的にコードへ降りていきます。

---

## Phase 0: プロジェクトの全体像を掴む（30分）

### 📖 読むドキュメント

1. **[../00_overview.md](../00_overview.md)** — プロジェクト全体の概要、なぜ必要なのかを理解
2. **[../08_code_walkthrough.md](../08_code_walkthrough.md)** — ファイル構成と依存関係の俯瞰

### 💻 軽く眺めるコード

ファイル構成を `find` で確認しておく：
```bash
find src/lsbi_smc -type f -name "*.py" | sort
```

→ ここで「あ、こういうファイルがあるんだ」程度の感覚を得る。

---

## Phase 1: 前提知識の確認（必要に応じて）

### 1-0. 線形代数・NumPy/SciPy の基礎

ベクトル・行列演算、固有値問題、ブロードキャスティングの理解が浅い場合：

📖 **[../01_linear_algebra.md](../01_linear_algebra.md)** を読む

キーワード:
- 一般化固有値問題（`scipy.linalg.eigh(K, M)`）
- ブロードキャスティング（`[:, None]` イディオム）
- Cholesky 分解、対数スケールでの数値安定性

### 1-1. 数学・ベイズ統計の基礎

ベイズの定理、MCMC、事後分布などの理解が浅い場合：

📖 **[../02_probability.md](../02_probability.md)** を読む

キーワード:
- 事前分布、尤度、事後分布
- MCMC、Metropolis-Hastings
- ガウス分布、KLダイバージェンス

### 1-2. 機械学習・VAEの基礎

VAE（変分オートエンコーダ）の理解が浅い場合：

📖 **[../03_ml_prerequisites.md](../03_ml_prerequisites.md)** を読む

キーワード:
- エンコーダ・デコーダ
- 再パラメータ化トリック
- ELBO、KL正則化、再構成損失
- 残差接続（ResNet）

### 1-3. 構造工学の基礎

FRF・固有値解析・レイリー減衰の理解が浅い場合：

📖 **[../04_structural_engineering.md](../04_structural_engineering.md)** を読む

キーワード:
- せん断建物モデル
- 質量行列、剛性行列
- 固有モード解析
- 周波数応答関数（FRF）

---

## Phase 2: Step 1（データ生成）のコードを読む（45分）

ここから実コードに入ります。**もっとも単純な物理シミュレーション** から始めるのが理解しやすいルートです。

### 2-1. 物理シミュレーター本体

📖 **[frfshearm.md](frfshearm.md)** — `frfshearm2` 関数の詳細  
💻 [src/lsbi_smc/example_shear4dof/frfshearm.py](../../src/lsbi_smc/example_shear4dof/frfshearm.py)

**ポイント**:
- 入力: 各階の剛性 `ks`、出力: 対数 FRF
- 内部では `scipy.linalg.eigh` で固有値解析
- レイリー減衰係数 `a₀, a₁` を連立方程式で決定
- モード重ね合わせ法で FRF を組み立て

### 2-2. シミュレーターの並列実行ラッパー

📖 **[simulator.md](simulator.md)** — `Simulator` クラス  
💻 [src/lsbi_smc/simulator/simulator.py](../../src/lsbi_smc/simulator/simulator.py)

**ポイント**:
- `[0, 1]` の正規化パラメータ → 物理パラメータに変換
- `multiprocessing.dummy`（スレッドプール）で並列実行
- 出力形状を `(n, n_floors, 1, n_freq)` に整形

### 2-3. データセット生成スクリプト

📖 **[create_dataset.md](create_dataset.md)** — 100,000 サンプルの生成  
💻 [src/lsbi_smc/example_shear4dof/create_dataset.py](../../src/lsbi_smc/example_shear4dof/create_dataset.py)

**ポイント**:
- ラテンハイパーキューブで均一サンプリング
- ノイズを加えた `y_sim_n` も生成
- `train_data.npz` として保存

### ✅ Phase 2 のチェックポイント

ここまでで、「なぜ100,000 サンプルが必要か」「FRF とは何の出力か」を説明できれば次へ進めます。

---

## Phase 3: Step 2（MVAE学習）のコードを読む（90分）

このフェーズが最も重い部分です。深層学習の構造を理解する必要があります。

### 3-1. 学術的背景

📖 **[../05_mvae.md](../05_mvae.md)** — MVAE の数式と設計思想

**重要な疑問**:
- なぜ「Multimodal」なのか？
- なぜ2つのエンコーダの潜在表現を一致させるのか？

### 3-2. ニューラルネットワーク本体

📖 **[mvae.md](mvae.md)** — `Encoder`, `EncoderW`, `Decoder`, `MVAE`  
💻 [src/lsbi_smc/example_shear4dof/mvae.py](../../src/lsbi_smc/example_shear4dof/mvae.py)

**読む順序の推奨**:
1. ユーティリティ関数 `reparameterization`、`gauss_unitgauss_kl`、`gauss_gauss_kl`、`rec_loss_norm_4d`
2. 残差ブロック群 `ResblockEnc`, `FirstResblockEnc`, `ResblockEncSmall`, `ResblockDec`
3. `Encoder` クラス（FRF用、畳み込みResNet）
4. `EncoderW` クラス（パラメータ用、全結合ResNet）
5. `Decoder` クラス（共通デコーダ、転置畳み込みResNet）
6. `MVAE` クラス本体（`encode`, `decode`, `loss`）

**ポイント**:
- `Decoder` は2つのエンコーダで**共有**される（クロスモーダル学習の鍵）
- `var` 出力には常に `Softplus`（正値保証）
- 損失は KL 正則化 + KL 整合性 + 2種類の再構成損失

### 3-3. 学習スクリプト

📖 **[train.md](train.md)** — 学習ループ、早期終了  
💻 [src/lsbi_smc/example_shear4dof/train.py](../../src/lsbi_smc/example_shear4dof/train.py)

**ポイント**:
- 屋根階のみ（`ch=[-1]`）を学習データに使用
- 標準化定数 `y_mn, y_sd` の値を覚えておく（推論時に使う）
- `model.loss(yn, y, x, alp1=5.0)` の引数の意味を `train.md` で確認
- 早期終了で過学習を防ぐ

### ✅ Phase 3 のチェックポイント

「なぜ `enc_x` と `enc_w` の潜在表現を一致させると尤度近似が可能になるのか」を説明できれば次へ進めます（次のフェーズで詳しく学びます）。

---

## Phase 4: Step 3（推論）のコードを読む（120分）

LSBI-SMC の最も革新的な部分。**潜在空間ベース尤度** と **SMC サンプラー** を組み合わせます。

### 4-1. 潜在空間ベース尤度の理論

📖 **[../06_latent_likelihood.md](../06_latent_likelihood.md)** — 数学的導出

**重要な疑問**:
- なぜ高次元 FRF の尤度が直接計算できないのか？
- なぜ潜在空間で計算できるのか？
- 3つのガウス分布の積分が解析的に計算できるのはなぜか？

### 4-2. 潜在空間ベース尤度の実装

📖 **[latentlik.md](latentlik.md)** — `latent_space_loglik`, `MVAEBasedLogLikelihood`  
💻 [src/lsbi_smc/likelihood/latentlik.py](../../src/lsbi_smc/likelihood/latentlik.py)

**ポイント**:
- `MVAEBasedLogLikelihood.__init__` で観測 FRF を1回だけ事前エンコード
- 各 SMC ステップでは `enc_w(theta)` のみ実行（高速）
- ガウス積分の閉形式を `a, b, c, d` の係数で計算

### 4-3. SMC の理論

📖 **[../07_smc.md](../07_smc.md)** — Sequential Monte Carlo の原理

**重要な疑問**:
- なぜ普通のMCMCではダメなのか？（多峰性、高次元）
- アニーリング（β を 0→1 へ徐々に）の意味は？
- 隣り合う温度の分布の橋渡しがなぜ重点サンプリングで書けるのか？
- ESS（有効サンプルサイズ）が「実質的な粒子数」を表すとはどういうことか？
- 重み付け／リサンプリング／MCMC ムーブの 3 要素がどう補い合うのか？

### 4-4. 確率変数と事前分布

📖 **[variables.md](variables.md)** — `Constant`, `Normal`, `Uniform` など  
💻 [src/lsbi_smc/smc/variables.py](../../src/lsbi_smc/smc/variables.py)

📖 **[prior.md](prior.md)** — `HierarchicalPrior`  
💻 [src/lsbi_smc/smc/prior.py](../../src/lsbi_smc/smc/prior.py)

**ポイント**:
- DAG（有向非巡回グラフ）で階層構造を表現
- `depth` 属性でトポロジカルソート
- `sample()`, `lp()`, `check_support()` の3つの共通インタフェース

### 4-5. 提案分布

📖 **[proposal.md](proposal.md)** — `ChingAndChenProposal`  
💻 [src/lsbi_smc/smc/proposal.py](../../src/lsbi_smc/smc/proposal.py)

**ポイント**:
- 粒子の重み付き共分散をスケーリングして提案
- `b=0.2` は経験的最適値
- 適応的な探索が可能

### 4-6. MCMC カーネル

📖 **[kernel.md](kernel.md)** — `RWMetropolisKernel`, `HMCKernel`  
💻 [src/lsbi_smc/smc/kernel.py](../../src/lsbi_smc/smc/kernel.py)

**ポイント**:
- `RWMetropolisKernel`: ベンチマークで使用される標準実装
- `HMCKernel`: 勾配を使った高効率版（拡張用）
- どちらも `KernelProtocol` に準拠

### 4-7. SMC サンプラー本体

📖 **[smc.md](smc.md)** — `SMC` クラス、`Particles`、`_find_next_q`  
💻 [src/lsbi_smc/smc/smc.py](../../src/lsbi_smc/smc/smc.py)

**ポイント**:
- `Particles` クラス: 粒子の状態管理
- `_find_next_q`: ESS 基準で β を二分探索
- `run()` メソッド: SMC のメインループ
- 各ステップで MCMC を `mcmc_iter=10` 回実行

### 4-8. 推論のメインスクリプト

📖 **[inference.md](inference.md)** — 推論パイプラインの統合  
💻 [src/lsbi_smc/example_shear4dof/inference.py](../../src/lsbi_smc/example_shear4dof/inference.py)

**ポイント**:
- 合成観測の生成（真値 θ=1.0 + ノイズ）
- CDF 変換: 無制約空間 ℝ → 正規化空間 \[0,1\] → 物理空間 \[L, U\]（3 空間の往来）
- 学習時の標準化定数 `y_mn, y_sd` を再利用
- SMC 実行 → `posterior.mat` 保存

### ✅ Phase 4 のチェックポイント

`inference.py` のすべての行を「なぜそうしているのか」を説明できればプロジェクトの理解は完了です。

---

## Phase 5: 結果の可視化（30分）

📖 **[plot_posterior.md](plot_posterior.md)** — コーナープロット生成  
💻 [src/lsbi_smc/example_shear4dof/plot_posterior.py](../../src/lsbi_smc/example_shear4dof/plot_posterior.py)

**ポイント**:
- 事後分布の周辺分布（対角ヒストグラム）
- パラメータペアの相関（非対角散布図 + KDE）
- 真値と等価解の重ね合わせ表示

---

## Phase 6: 実際に動かしてみる（60分）

### 6-1. 環境セットアップ

```bash
# Docker（GPU環境）
make build && make up

# あるいは uv でローカル環境
uv sync
```

### 6-2. データ生成 → 学習 → 推論

```bash
# Step 1: データセット生成（数分）
uv run python src/lsbi_smc/example_shear4dof/create_dataset.py

# Step 2: MVAE 学習（GPUなら30分〜1時間程度）
uv run python src/lsbi_smc/example_shear4dof/train.py

# Step 3: 推論（数十秒〜数分）
uv run python src/lsbi_smc/example_shear4dof/inference.py

# 補助: 可視化
uv run python src/lsbi_smc/example_shear4dof/plot_posterior.py
```

### 6-3. パラメータをいじって遊ぶ

- `inference.py` の `pop_size=2000` → 1000 や 5000 に変えて挙動を観察
- `mcmc_iter=10` → 5 や 20 に変えて精度・速度のトレードオフを観察
- `b=0.2` → 0.1, 0.5 に変えて採択率の変化を観察

---

## 学習順序のまとめ

```
[Phase 0]  全体概要 (30分)
   00_overview → 08_code_walkthrough
                   ↓
[Phase 1]  前提知識（必要に応じて）
   01_linear_algebra / 02_probability / 03_ml_prerequisites / 04_structural_engineering
                   ↓
[Phase 2]  Step 1 のコード (45分)
   frfshearm → simulator → create_dataset
                   ↓
[Phase 3]  Step 2 のコード (90分)
   05_mvae → mvae.py → train.py
                   ↓
[Phase 4]  Step 3 のコード (120分)
   06_latent_likelihood → latentlik.py
   07_smc → variables → prior → proposal → kernel → smc → inference
                   ↓
[Phase 5]  可視化 (30分)
   plot_posterior
                   ↓
[Phase 6]  実際に動かす (60分)
   create_dataset → train → inference → plot_posterior
```

合計の目安: 約 7〜10 時間（背景知識のレベルによる）。

---

## つまずきやすいポイント

| つまずき | 対策 |
|---------|------|
| FRF の物理的な意味がわからない | [04_structural_engineering.md](../04_structural_engineering.md) を再読 |
| なぜ 2 つのエンコーダが必要かピンと来ない | [05_mvae.md](../05_mvae.md) Section 1 を再読 |
| 潜在空間で尤度が計算できる理屈がわからない | [06_latent_likelihood.md](../06_latent_likelihood.md) Section 2-3 を再読 |
| CDF 変換の意味がわからない | [06_latent_likelihood.md](../06_latent_likelihood.md) Section 6、[inference.md](inference.md) を再読 |
| なぜ RW-MH ではなく SMC を使うのか | [07_smc.md](../07_smc.md) Section 1 を再読 |
| ESS と β の関係がわからない | [07_smc.md](../07_smc.md) Section 3、[smc.md](smc.md) を再読 |
| データの形状がよくわからない | [08_code_walkthrough.md](../08_code_walkthrough.md) Section 5 を確認 |

---

## 次のステップ

このコードベースを理解した後の発展的な学習：

1. **論文を読む**: `papers/2603_ned260329.pdf` の Method/Results セクション
2. **HMC カーネルを試す**: `RWMetropolisKernel` を `HMCKernel` に置き換えて実行
3. **異なる事前分布を試す**: `Normal` から `Uniform` や `Laplace` への変更
4. **多自由度への拡張**: `ndof=4` を `8` や `16` にして同じパイプラインを動かす
5. **実観測データへの適用**: 合成観測を実際の建物センサーデータに置き換える
