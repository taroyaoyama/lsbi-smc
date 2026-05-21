---
marp: true
theme: default
paginate: true
math: mathjax
size: 16:9
header: ''
footer: '工学逆問題における不確実性定量化のための潜在空間ベイズ推論フレームワーク構築'
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
  img[alt~="center"] {
    display: block;
    margin: 0 auto;
  }
  pre {
    font-size: 20px;
  }
---

<!-- _class: title -->

# 工学逆問題における不確実性定量化のための
# 潜在空間ベイズ推論フレームワーク構築

<br>

A Latent-Space Bayesian Inference Framework for
Uncertainty Quantification in Engineering Inverse Problems

<br><br>

指導教員　村上健太 准教授
03-250919　笠井優作

---

# 1. はじめに

**工学では「観測データからモデルパラメータを推定」する逆問題が遍在**

- 構造物の剛性・減衰の同定、材料定数・損傷度の推定、流体・熱輸送パラメータの推定
- 診断・リスク評価・意思決定には、推定の **不確実性定量化** が不可欠

**ベイズ推論** は事後分布として不確実性を直接与える:

$$
p(\theta \mid x_\mathrm{obs}) \;\propto\; \underbrace{L(\theta;\,x_\mathrm{obs})}_{\text{尤度}} \cdot \underbrace{p(\theta)}_{\text{事前}}
$$

しかし **高コストなシミュレータ呼び出し** が実用上の壁となる。

→ 本研究: これらを回避する **潜在空間ベイズ推論** を起点に、**各構成要素を差し替え可能** とした推論フレームワークを構築。

<!--
SPEAKER: 35秒。工学逆問題 → 不確実性定量化 → ベイズ → シミュレータの壁 → 本研究の枠組み、と一直線に。
-->

---

# 2-1. 潜在空間ベイズ推論 — 定式化

ベイズの定理より:

$$
p(\theta \mid x_\mathrm{obs}) \;\propto\; L(\theta;\, x_\mathrm{obs})\, p(\theta) \tag{1}
$$

低次元潜在変数 $z$ を介して尤度を近似する。観測側エンコーダ $q_{\phi_x}$、パラメータ側エンコーダ $q_{\phi_\theta}$、潜在事前 $p(z) = \mathcal{N}(0,I)$ を **全てガウス** として構成すれば、

$$
\hat{L}(\theta;\, x_\mathrm{obs}) \;=\; \int \frac{q_{\phi_x}(z \mid x_\mathrm{obs})\, q_{\phi_\theta}(z \mid \theta)}{p(z)}\, dz \tag{2}
$$

は **閉形式で評価可能** となる。

![w:550 center](assets/latent_overlap.png)

<!--
SPEAKER: 60秒。(1) ベイズの定理を確認、(2) 潜在変数を介して尤度近似、ガウスにすれば閉形式、と式を指で追う。図は2エンコーダの重なりイメージ。
-->

---

# 2-1. 潜在空間ベイズ推論 — オフライン/オンライン

![w:1050 center](assets/pipeline.png)

- **オフライン**: 事前から $(\theta, x)$ サンプル → MVAE 学習 → 近似尤度 $\hat{L}$ を構築
- **オンライン**: SMC が $\hat{L}$ のみを呼び事後分布から粒子をサンプリング

→ 推論ループで **シミュレータを呼ばない**

<!--
SPEAKER: 50秒。図の左半分がオフライン (MVAE学習)、右半分がオンライン (SMC) と指す。「推論時にFEを呼ばない」が肝。
-->

---

# 2-2. フレームワーク実装 — Protocol で抽象化

文献 1) の具体実装を、各構成要素が **後から差し替え可能** なライブラリとして再整備:

<div style="display: flex; gap: 24px;">
<div style="flex: 1.05;">

| 概念的役割     | 現実装               |
| -------------- | -------------------- |
| 尤度モデル     | MVAE 潜在尤度        |
| サンプラー     | 焼きなまし型 SMC     |
| MCMC カーネル  | RW-Metropolis        |
| 提案分布       | Ching & Chen (2007)  |
| 事前分布       | HierarchicalPrior    |

</div>
<div style="flex: 1;">

```python
class LikelihoodProtocol(Protocol):
    def __call__(self, theta: Theta) -> LP: ...

class KernelProtocol(Protocol):
    def __call__(
        self, particles: Particles, q: float,
        prior: PriorProtocol,
        likelihood: LikelihoodProtocol,
    ) -> tuple[Pop, LP, Mask]: ...
```

**図 1** 代表的な Protocol 定義（抜粋）

</div>
</div>

具体実装は Protocol にのみ依存 → 他の尤度モデル（**NF・拡散モデル**）、サンプラー（**HMC・Nested Sampling**）への差し替えが簡単
→ 「**どの手法がどの問題に最適か**」を共通ベンチマーク上で **横断比較** できる構成

<!--
SPEAKER: 60秒。左の表で「概念的役割と現実装の対応」、右でコード抜粋を見せる。差し替え対象 (NF/拡散/HMC/NS) を口頭で。
-->

---

# 3. 検証 — 4自由度せん断建物

**ベンチマーク** (Yaoyama 2026 と同じ):

- 4 自由度せん断建物の **層剛性同定問題**
- 観測は **屋上 FRF** (1024 点) のみ
- $\theta_i \in [0.33, 3.00]$、真値 $\theta=(1,1,1,1)$
- **真値以外に 3 つの等価解** が存在することが既知

**学習設定**:
- 訓練データ: 事前分布から生成した $10^5$ サンプル
- MVAE: 潜在次元 8、早期終了付きで学習

→ 多峰性が現れる古典的ベンチマークでフレームワークの動作確認

<!--
SPEAKER: 25秒。設定を一気に。「等価解 3 つが既知」は次の結果スライドで効くので強調。
-->

---

# 3. 検証 — 結果

![bg right:46% w:95%](../../posterior_plot.png)

**SMC**: 粒子数 2000、GPU 上で実行

**事後分布散布図**（右図）:
- **真値** $(1,1,1,1)$ に対応する集中モード
- **既知の等価解 3 点** にも別モードが立つ

→ 観測の制約に由来する **多峰性を本フレームワークが正しく追跡** できることを確認

<!--
SPEAKER: 45秒。右の散布図を指しながら「メインモード=真値」「他3モード=等価解」と対応を示す。多峰性に対応できた、で締める。
-->

---

# 課題と今後

**現状の限界**:
- 検証は 4DOF せん断建物のみ
- 他の尤度モデル・サンプラーは未実装、横断比較はこれから

**今後**:

🎯 **尤度モデル拡張を最優先課題** とする
- **Normalizing Flow / 拡散モデル** 等を実装
- 同一ベンチマーク上で **横断比較**

→ 複雑系へ適用を広げ、**汎用ライブラリへ発展** させる

<br>

<center>

**ご清聴ありがとうございました**

</center>

<!--
SPEAKER: 20秒。「尤度モデル拡張」を一点突破で押し、汎用ライブラリで締める。
-->
