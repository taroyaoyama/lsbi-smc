# プレゼン構成案 v3（要旨準拠）

- **タイトル**: 「工学逆問題における不確実性定量化のための潜在空間ベイズ推論フレームワーク構築」
- **想定聴衆**: 構造工学系（FEM・振動は既知 / VAE・SMC は未知）
- **発表時間**: 5分（質疑除く）
- **言語**: 日本語

---

## v3 の設計思想

最終提出した要旨（[abstract/abstract_v2.md](abstract/abstract_v2.md)）の章立て・配分・流れに完全準拠する。

要旨の構造:

1. **はじめに** — 工学逆問題と不確実性定量化の必要性、ベイズ推論、シミュレータ呼び出しの壁、本研究の立ち位置
2. **フレームワーク概要**
   - 2-1 潜在空間ベイズ推論（数式 + MVAE/SMC のオフライン/オンライン構成）
   - 2-2 フレームワーク実装（Protocol で抽象化、差し替え可能）
3. **検証と課題** — 4自由度せん断建物の動作確認、結果の散布図、今後の方針

v2 にあった「既存アプローチ俯瞰の表」スライドは、要旨に対応する記述がないため落とす。Q&A 側に温存する。

---

## 配分（5分 = 300秒）

| #  | 章                                  | 時間  | スライド枚数 |
|----|-------------------------------------|-------|--------------|
| 0  | タイトル                            | 5s    | 1            |
| 1  | はじめに                            | 35s   | 1            |
| 2  | 潜在空間ベイズ推論 — 定式化         | 60s   | 1            |
| 3  | 潜在空間ベイズ推論 — オフライン/オンライン | 50s   | 1            |
| 4  | フレームワーク実装 — Protocol で抽象化 | 60s   | 1            |
| 5  | 検証 — 4自由度せん断建物            | 25s   | 1            |
| 6  | 検証 — 結果                         | 45s   | 1            |
| 7  | 課題と今後                          | 20s   | 1            |
|    | **合計**                            | **300s** | **8枚** |

---

## スライド別 要点

### Slide 0: タイトル
- メイン: 工学逆問題における不確実性定量化のための潜在空間ベイズ推論フレームワーク構築
- サブ: 4自由度せん断建物を題材とした検証

---

### Slide 1: はじめに（要旨 §1）
**要旨 §1 にそのまま対応**

- 工学では観測データからモデルパラメータを推定する逆問題が広く現れる
- 診断やリスク評価には推定の **不確実性定量化** が不可欠
- ベイズ推論は事後分布として不確実性を直接与えるが、**高コストなシミュレータ呼び出し** が実用上の壁
- 本研究: これらを回避する潜在空間ベイズ推論を起点に、**各構成要素を差し替え可能** とした推論フレームワークを構築

→ 図: なし（テキスト中心）

---

### Slide 2: 潜在空間ベイズ推論 — 定式化（要旨 §2-1 前半）

ベイズの定理:

$$
p(\theta \mid x_\mathrm{obs}) \;\propto\; L(\theta;\, x_\mathrm{obs})\, p(\theta) \tag{1}
$$

低次元潜在変数 $z$ を介して尤度を近似:

- 観測側エンコーダ $q_{\phi_x}(z \mid x_\mathrm{obs})$
- パラメータ側エンコーダ $q_{\phi_\theta}(z \mid \theta)$
- 潜在事前 $p(z) = \mathcal{N}(0, I)$

全てガウスとして構成すれば、

$$
\hat{L}(\theta;\, x_\mathrm{obs}) \;=\; \int \frac{q_{\phi_x}(z \mid x_\mathrm{obs})\, q_{\phi_\theta}(z \mid \theta)}{p(z)}\, dz \tag{2}
$$

は **閉形式で評価可能**。

→ 図: 潜在空間における2つのエンコーダの重なり（[assets/latent_overlap.png](assets/latent_overlap.png)）

---

### Slide 3: 潜在空間ベイズ推論 — オフライン/オンライン（要旨 §2-1 後半）

**提案フレームワークでは:**

- **オフライン**: 事前から $(\theta, x)$ サンプルを生成 → Multimodal Variational Autoencoder (MVAE) を学習 → 近似尤度 $\hat{L}$ を構築
- **オンライン**: Sequential Monte Carlo (SMC) が $\hat{L}$ のみを呼び事後分布から粒子をサンプリング

→ 推論ループで **シミュレータを呼ばない**

→ 図: パイプライン全体図（[assets/pipeline.png](assets/pipeline.png)）

---

### Slide 4: フレームワーク実装 — Protocol で抽象化（要旨 §2-2）

**本研究の貢献**: 文献 1) の具体実装を、各構成要素が後から差し替え可能なライブラリとして再整備。

具体手法（MVAE, SMC, Metropolis-Hastings など）を共通の概念的役割に分離:

| 概念的役割     | Protocol               | 現実装                     |
| -------------- | ---------------------- | -------------------------- |
| 尤度モデル     | `LikelihoodProtocol`   | MVAE 潜在尤度              |
| サンプラー     | `class SMC`            | 焼きなまし型 SMC           |
| MCMC カーネル  | `KernelProtocol`       | RW-Metropolis              |
| 提案分布       | `ProposalProtocol`     | Ching & Chen (2007)        |
| 事前分布       | `PriorProtocol`        | HierarchicalPrior          |

Python の `Protocol` でインターフェースのみを宣言:

```python
class LikelihoodProtocol(Protocol):
    def __call__(self, theta: Theta) -> LP: ...

class KernelProtocol(Protocol):
    def __call__(
        self, particles: Particles, q: float,
        prior: PriorProtocol, likelihood: LikelihoodProtocol,
    ) -> tuple[Pop, LP, Mask]: ...
```

→ 他の尤度モデル（NF・拡散モデル）、サンプラー（HMC・Nested Sampling）への差し替えが簡単
→ 設計思想: 「**どの手法がどの問題に最適か**」を共通ベンチマーク上で **横断比較** できる構成

---

### Slide 5: 検証 — 4自由度せん断建物（要旨 §3 前半）

**ベンチマーク設定** (Yaoyama 2026 と同じ):

- 4 自由度せん断建物の **層剛性同定問題**
- 観測は屋上 FRF（1024 点）のみ
- **真値以外に 3 つの等価解** が存在することが既知

**学習設定**:
- 訓練データ: 事前分布から生成した $10^5$ サンプル
- MVAE: 潜在次元 8、早期終了付き

→ 図: 4DOFせん断建物の質点ばねダンパ模式図（任意・テキストで代用可）

---

### Slide 6: 検証 — 結果（要旨 §3 中盤）

**SMC 実行**: 粒子数 2000、GPU 上で実行

**結果**: 真値・等価解の 4 モードがすべて再現
- 多峰性に本フレームワークが対応できることを確認

→ 図: 事後分布散布図 [../../posterior_plot.png](../../posterior_plot.png)

---

### Slide 7: 課題と今後（要旨 §3 後半）

**今後の方針**:

- **尤度モデル拡張を最優先課題** とする
- Normalizing Flow・拡散モデル等を **同一ベンチマーク上で横断比較**
- 複雑系へ適用を広げ、**汎用ライブラリへ発展** させる

→ 図: なし（テキスト + 締め）

---

## 図表チェックリスト

| 図                       | スライド | 出典                                                |
|--------------------------|----------|-----------------------------------------------------|
| 潜在空間のエンコーダ重なり | Slide 2  | [assets/latent_overlap.png](assets/latent_overlap.png) |
| パイプライン             | Slide 3  | [assets/pipeline.png](assets/pipeline.png)          |
| Protocol コード抜粋      | Slide 4  | スライド内コードブロック                            |
| 事後分布散布図           | Slide 6  | [../../posterior_plot.png](../../posterior_plot.png) |

予備:
- [assets/vae_mvae_compare.png](assets/vae_mvae_compare.png) — VAE/MVAE 比較。Q&A 行きとして温存

---

## Q&A 想定の重点項目

[04_qa.md](04_qa.md) で扱うトピック:

- 5分内で扱いきれない既存アプローチ（古典 MCMC / NF / 拡散）の俯瞰
- なぜ不確実性を評価する必要があるのか
- 等価解（多峰性）をどう扱うか
- 工学一般への適用性
- ライブラリ設計の細部（Protocol を選んだ理由など）

---

## ビルド方針

- スライド: Marp markdown (`02_slides.md`)
- ビルド: `npx @marp-team/marp-cli@latest 02_slides.md --pdf`
- 図生成: `make figures`（make_figures.py で JA/EN 両言語）
- 英語版 (`02_slides_en.md` 等) は v3 では未更新（v2 構造のまま）
