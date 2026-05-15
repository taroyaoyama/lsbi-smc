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

本研究では、文献 1) の具体実装を、各構成要素が後から差し替え可能なライブラリとして再整備した。文献 1) で採用されている具体手法 — 尤度モデルとしての MVAE、サンプラーとしての SMC、MCMC カーネルとしてのランダムウォーク Metropolis-Hastings、適応共分散による提案分布など — を、共通の概念的役割（尤度モデル・サンプラー・MCMC カーネル・提案分布・事前分布）に分離し、Python の Protocol によりインターフェースのみを宣言する形に再整理した。図 1 に代表的な Protocol 定義の抜粋を示す。

```python
class LikelihoodProtocol(Protocol):
    def __call__(self, theta: Theta) -> LP: ...

class KernelProtocol(Protocol):
    def __call__(
        self, particles: Particles, q: float,
        prior: PriorProtocol, likelihood: LikelihoodProtocol,
    ) -> tuple[Pop, LP, Mask]: ...
```

**図 1**　代表的な Protocol 定義（抜粋）。各モジュールが満たすべきインターフェースのみを宣言する。

具体実装は Protocol を介して上位ロジックと結合しているため、MVAE 以外の尤度モデル（Normalizing Flow、拡散モデル、直接尤度など）や、SMC 以外のサンプラー（Hamiltonian Monte Carlo、Nested Sampling など）、別の MCMC カーネル（MALA など）への差し替えが、上位ロジックを書き換えずに可能となる。設計思想は、「どの手法がどの問題に最適か」を共通ベンチマーク上で横断比較できる構成にすることにある。

## 3. 検証と課題

文献 1) で扱われている 4 自由度せん断建物の屋根周波数応答関数からの層剛性同定問題で動作確認した。本ベンチマークは、屋根応答のみを観測するという制約から、真値以外に 3 つの等価解が存在することが既知である 1)。

MVAE の学習は、事前分布から生成した $10^5$ 個のシミュレーション結果を訓練データとし、潜在次元 8 のもとで早期終了付きで実施した。学習済みモデルを用いて、粒子数 2000 の SMC を GPU 上で実行したところ、約 1.5 秒で事後サンプリングが完了した。

![図 2 — 4 自由度せん断建物の事後分布散布図](../../../posterior_plot.png)

**図 2**　4 自由度せん断建物の事後分布散布図。真値 $\theta = (1, 1, 1, 1)$ と既知の等価解 3 点を、各層剛性 $\theta_i$ の周辺・同時分布上に重ねて表示。

図 2 より、真値および既知の 3 つの等価解にそれぞれ対応する 4 つの集中モードが事後分布として再現されていることが確認できる。これにより、観測情報の不足に由来する多峰性を持つ事後分布を、本フレームワークが正しく追跡できることを示した。

今後は、尤度モデル部分の拡張比較を最優先課題とし、Normalizing Flow や拡散モデル等の代替アプローチを、精度・推論時間・多峰性への頑健さの観点から同一ベンチマーク上で横断比較する。さらに、より高次元・非線形・多モダリティ観測を含む複雑系へと適用範囲を広げ、工学逆問題ごとに最適な手法構成を選択可能なライブラリへと発展させる。

## 参考文献

1) Yaoyama, T. et al., Finite element model updating of building structures under seismic excitation: A parallelized latent space-based Bayesian framework, arXiv:2604.22305 (preprint, 2026).
