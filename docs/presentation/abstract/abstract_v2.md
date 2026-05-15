# 工学逆問題における不確実性定量化のための潜在空間ベイズ推論フレームワーク構築

A Latent-Space Bayesian Inference Framework for Uncertainty Quantification in Engineering Inverse Problems

指導教員　村上健太 准教授
03-250919　笠井優作

---

## 1. はじめに

工学では観測データからモデルパラメータを推定する逆問題が広く現れ、診断やリスク評価には推定の不確実性定量化が不可欠である。ベイズ推論は事後分布として不確実性を直接与えるが、高次元応答時系列と高コストなシミュレータ呼び出しが実用上の壁となる。本研究では、これらを回避する潜在空間ベイズ推論を起点に、各構成要素を差し替え可能とした推論フレームワークを構築する。

## 2. フレームワーク概要

### 2-1. 潜在空間ベイズ推論

文献 1) に従い、潜在空間ベイズ推論の枠組みを概説する。ベイズの定理より、事後分布は尤度と事前分布の積に比例する。

$$
p(\theta \mid x_\mathrm{obs}) \;\propto\; L(\theta;\, x_\mathrm{obs})\, p(\theta) \tag{1}
$$

潜在空間ベイズ推論では低次元潜在変数 $z$ を介して尤度を近似する。観測側エンコーダ $q_{\phi_x}$、パラメータ側エンコーダ $q_{\phi_\theta}$、潜在事前 $p(z) = \mathcal{N}(0, I)$ を全てガウスとして構成すれば、

$$
\hat{L}(\theta;\, x_\mathrm{obs}) \;=\; \int \frac{q_{\phi_x}(z \mid x_\mathrm{obs})\, q_{\phi_\theta}(z \mid \theta)}{p(z)}\, dz \tag{2}
$$

は閉形式で評価可能となる。事前から $(\theta, x)$ サンプルを生成し Multimodal Variational Autoencoder (MVAE) を学習して近似尤度 $\hat{L}$ をオフラインで構築し、オンラインでは Sequential Monte Carlo (SMC) が $\hat{L}$ のみを呼び事後分布から粒子をサンプリングする。

### 2-2. フレームワーク実装

本研究では、文献 1) の具体実装を、各構成要素が後から差し替え可能なライブラリとして再整備した。文献 1) で採用されている具体手法（MVAE、SMC、ランダムウォーク Metropolis-Hastings、適応共分散による提案分布など）を、共通の概念的役割に分離し、Python の Protocol によりインターフェースのみを宣言する形に再整理した。図 1 に代表的な Protocol 定義の抜粋を示す。

```python
class LikelihoodProtocol(Protocol):
    def __call__(self, theta: Theta) -> LP: ...

class KernelProtocol(Protocol):
    def __call__(
        self, particles: Particles, q: float,
        prior: PriorProtocol, likelihood: LikelihoodProtocol,
    ) -> tuple[Pop, LP, Mask]: ...
```

**図 1**　代表的な Protocol 定義（抜粋）。

具体実装は Protocol を介して上位ロジックと結合しているため、他の尤度モデル（Normalizing Flow、拡散モデルなど）やサンプラー（HMC、Nested Sampling など）への差し替えが、上位ロジックを書き換えずに可能となる。設計思想は、「どの手法がどの問題に最適か」を共通ベンチマーク上で横断比較できる構成にすることにある。

## 3. 検証と課題

文献 1) と同じ 4 自由度せん断建物の層剛性同定問題（屋根周波数応答関数のみ観測）で動作確認した。観測の制約により、真値以外に 3 つの等価解が存在することが既知である 1)。

MVAE は、事前分布から生成した $10^5$ 個のサンプルを訓練データとし、潜在次元 8・早期終了付きで学習した。粒子数 2000 の SMC を GPU 上で実行し、約 1.5 秒で完了した。図 2 に得られた事後分布の散布図を示す。

![図 2 — 4 自由度せん断建物の事後分布散布図](../../../posterior_plot.png)

**図 2**　事後分布散布図（4 自由度せん断建物）。★ は真値、△ は既知の等価解 3 点。

真値および既知の 3 つの等価解に対応する 4 つの集中モードが再現されており、観測の制約に由来する多峰性を本フレームワークが正しく追跡できることを確認した。

今後は尤度モデル拡張を最優先課題とし、Normalizing Flow や拡散モデル等を同一ベンチマーク上で横断比較する。さらに複雑系へ適用を広げ、汎用ライブラリへ発展させる。

## 参考文献

1) Yaoyama, T. et al., Finite element model updating of building structures under seismic excitation: A parallelized latent space-based Bayesian framework, arXiv:2604.22305 (preprint, 2026).
