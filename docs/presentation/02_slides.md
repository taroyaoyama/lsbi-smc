---
marp: true
theme: default
paginate: true
math: mathjax
size: 16:9
header: ''
footer: 'LSBI-SMC | 観測振動データに基づく建物FEモデル更新'
style: |
  section {
    font-family: "Hiragino Sans", "Yu Gothic", "Noto Sans CJK JP", sans-serif;
    font-size: 26px;
    padding: 50px 60px;
  }
  section.title {
    justify-content: center;
    text-align: center;
  }
  h1 {
    color: #1a3a5c;
    border-bottom: 3px solid #1a3a5c;
    padding-bottom: 8px;
  }
  h2 {
    color: #1a3a5c;
  }
  strong {
    color: #c0392b;
  }
  table {
    font-size: 22px;
  }
  blockquote {
    border-left: 4px solid #1a3a5c;
    color: #555;
  }
  footer {
    color: #888;
    font-size: 14px;
  }
---

<!-- _class: title -->

# 観測振動データに基づく建物FEモデル更新
## — 潜在空間ベイズ推論 + SMC の拡張可能実装 —

<br>

Yaoyama et al. (2026, NED) のフレームワーク実装と推論基盤のライブラリ化

<br><br>

**発表者**: （名前）
**所属**: （所属）
**日付**: 2026-XX-XX

---

# 研究背景 — FEモデル更新の必要性

- 建物の **地震応答予測** には高精度なFEモデルが前提
- しかし **設計時FEM ≠ 実建物**
  - 経年劣化、施工誤差、非構造部材の寄与 …
- **FE モデル更新 (model updating)**: 観測振動データから剛性等のパラメータ $\theta$ を逆推定
- 決定論的 fitting では不十分 → **不確かさを定量化するベイズ推定**が必要

<!--
SPEAKER: 5分の最初の50秒で「なぜこの研究をやるのか」を伝える。設計FEMと実建物のギャップ＝聴衆も日常的に感じている問題、と置いて入る。
-->

![bg right:35% w:90%](assets/building_fem.png)

---

# 目的と課題

**目的**: 観測FRF $x_\mathrm{obs}$ から、剛性パラメータ $\theta$ の **事後分布** $p(\theta \mid x_\mathrm{obs})$ を効率的に推定する

<br>

**3つの課題**:

|     | 課題                     | 内容                                                                   |
| --- | ------------------------ | ---------------------------------------------------------------------- |
| ①   | **高次元観測**           | FRF は 1024 点 → 尤度 $p(x_\mathrm{obs}\mid\theta)$ の直接定義が難しい |
| ②   | **シミュレータ高コスト** | 通常MCMCは数十万回FE解析を呼ぶ → 非現実的                              |
| ③   | **多峰性**               | 同じ屋根応答を再現する $\theta$ が複数存在（等価解）                   |

<!--
SPEAKER: 課題を3つに整理。①と②は計算機資源の話、③は構造系の方には実感がある「同じ応答を出す異なる剛性配分」の話、と分けて伝える。
-->

---

# 提案手法（既存）— 全体像

**オフライン学習 + オンライン推論** の2段構え

```
[オフライン]                              [オンライン]

事前分布 ─→ FE解析 ─→ (θ, FRF)
                          │
                          ▼
                       MVAE 学習
                          │
                          ▼
                     潜在空間 z      ←── 観測 x_obs
                          │                 │
                          └────→ SMC ←──────┘
                                  │
                                  ▼
                       事後分布 p(θ | x_obs)
```

1. データセット生成（FE並列実行）
2. **MVAE 学習** — $\theta$ と FRF を共通の潜在空間 $z$ に写す
3. **SMC 推論** — 潜在空間で尤度を計算してサンプリング

<!--
SPEAKER: ここから手法パート120秒。まず全体像で「2段構え」をひとことで掴ませる。詳細は次の3枚。
-->

---

# Step 2: MVAE — 2つのエンコーダを揃える

「$\theta$ も FRF $x$ も、**同じ潜在変数 $z$** から生成される」よう学習

- パラメータ用エンコーダ $\mathrm{enc}_w(\theta) \to (\mu_w, \sigma^2_w)$
- FRF 用エンコーダ $\mathrm{enc}_x(x) \to (\mu_x, \sigma^2_x)$
- 共通デコーダ $\mathrm{dec}(z) \to \hat{x}$

**損失**: 再構成項 + **2方向KL** $\mathrm{KL}(q_w \,\|\, q_x) + \mathrm{KL}(q_x \,\|\, q_w)$

→ 2つのエンコーダが**同じ $z$ 表現**を返すように揃う

<br>

> **VAE とは**: データを低次元の潜在変数 $z$ に圧縮し、そこから復元できるよう学習する深層学習手法。

![bg right:33% w:95%](assets/mvae_arch.png)

<!--
SPEAKER: ML弱め聴衆向け。図でenc_w / enc_x / decの3つのネットワークが見えればOK。引用ボックスでVAEの一言定義。
-->

---

# Step 3a: 潜在空間ベース尤度

1024次元のFRFを直接比較せず、**低次元の潜在空間で尤度を評価**

- 観測 $x_\mathrm{obs}$ を**1度だけ** $\mathrm{enc}_x$ で潜在化 → $(\mu_\mathrm{obs}, \sigma^2_\mathrm{obs})$
- 候補 $\theta$ を $\mathrm{enc}_w$ で潜在化 → $(\mu_w, \sigma^2_w)$
- 潜在事前 $\mathcal{N}(0, I)$ 上で **解析的に積分** → 閉形式

$$
\log p(x_\mathrm{obs}\mid\theta) \approx \log \int \mathcal{N}(z;\mu_\mathrm{obs},\sigma^2_\mathrm{obs})\, \mathcal{N}(z;\mu_w,\sigma^2_w)\, \mathcal{N}(z;0,I)\, dz
$$

→ **SMC ループ内で FE シミュレータを呼ばなくて済む**（最大のセールスポイント）

<!--
SPEAKER: ここが本手法の "効く" 理由。「FEを毎回呼ばずに尤度が出る」と1文で結ぶ。
-->

![bg right:30% w:95%](assets/latent_compare.png)

---

# Step 3b: SMC — 焼きなまし型サンプリング

逆温度 $q$ を $0$（事前）→ $1$（事後）まで**段階的に上げる**

各ステップで:
1. 重みづけ（尤度の $\Delta q$ 乗で更新）
2. **リサンプリング**
3. MCMC kernel（RW-Metropolis）で多様化

特徴:
- $q$ の刻みは **ESS（実効サンプル数）** に基づき適応決定
- 粒子は独立に進化 → **自然に並列化可能**
- 焼きなましのおかげで **多峰性（等価解）に強い**

![bg right:35% w:95%](assets/smc_evolution.png)

<!--
SPEAKER: 多峰性に強いという点を強調。構造系聴衆の関心とつながる。
-->

---

# ベンチマーク — 4自由度せん断建物

**設定**（論文と同じ）:

- 4層せん断モデル、各層剛性 $k_i \in [0.33, 3.00]$（正規化）
- 観測: 屋根のFRF（1024点、ホワイトノイズ励振、付加ノイズあり）
- 真値: $(k_1, k_2, k_3, k_4) = (1.0, 1.0, 1.0, 1.0)$
- 訓練データ: 事前から $N$ サンプル → FE並列実行
- SMC: 粒子数 $N_p$, MCMC step数 $N_m$

![bg right:45% w:90%](assets/shear4dof_frf.png)

<!--
SPEAKER: 設定をテンポよく。図で4階建てモデルと観測FRF例を見せる。
-->

---

# 結果 — 事後分布のコーナープロット

![bg right:55% w:95%](../../posterior_plot.png)

- メインモード: $k\approx(1,1,1,1)$ 付近に集中（真値を回収）
- 等価解 A/B/C も**別モード**として再現
- $k_4$（屋根層）は強く拘束、$k_1$ も well-identified、$k_2, k_3$ はやや広がる

→ 論文で報告されている挙動を**実装上でも再現**

<!--
SPEAKER: 結果をひと言で。「真値に当たった」「等価解も拾えた」の2点に絞る。
-->

---

# 本研究の貢献 — 拡張しやすい形での再実装

手法そのものは既存。本研究は **現方式のまま、後から差し替えやすい構造に再実装**

| モジュール   | 現在の中身              | 設計上の拡張余地（未実装） |
| ------------ | ----------------------- | -------------------------- |
| サンプラー   | SMC                     | HMC, NUTS, Nested Sampling |
| MCMCカーネル | RW-Metropolis           | MALA, HMC など             |
| 提案分布     | Ching & Chen (2007)     | 適応的MH, 学習ベース       |
| 尤度モデル   | MVAE 潜在尤度           | 標準VAE, NF, 直接尤度      |
| 事前分布     | HierarchicalPrior (DAG) | 任意の確率変数の組み合わせ |

**再実装上の整備**: 数値安定化 / 並列化整理 / 事前分布の DAG 宣言

> 代替手法そのものは未実装 — **今後の比較研究のための土台**を作った段階

<!--
SPEAKER: ここが自分の貢献。背伸びせず「土台を作った」と正直に。
-->

---

# まとめ・今後の展望

**やったこと**:
- LSBI-SMC（既存手法）を PyTorch で再実装、4DOFベンチマークで動作確認
- 各構成要素を後から差し替え可能なモジュール構成に整理

**今後の展望（ライブラリ拡張）**:
- 他サンプラー（NUTS / Nested Sampling）との比較実験
- MVAE 以外の尤度モデル（標準VAE / Normalizing Flow / 直接尤度）の比較
- 観測モダリティの追加（複数センサ、加速度 + 変位）
- より大規模 DOF・非線形応答への拡張
- 公開ベンチマーク化（4DOF 以外の検証問題を同梱）

<br>

<center>

**ご清聴ありがとうございました**

</center>

<!--
SPEAKER: 最後の30秒。やったこと2点 + 今後5点をテンポよく。
-->
