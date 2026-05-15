# 工学逆問題における不確実性定量化のための潜在空間ベイズ推論フレームワーク構築

A Latent-Space Bayesian Inference Framework for Uncertainty Quantification in Engineering Inverse Problems

指導教員　[教員氏名 教授/准教授/講師]
03-XXXXXX　[学生氏名]

---

## 1. はじめに

工学では観測データからモデルパラメータを推定する逆問題が広く現れ、診断やリスク評価には推定の不確実性定量化が不可欠である。ベイズ推論は事後分布として不確実性を直接与えるが、高次元応答時系列と高コストなシミュレータ呼び出しが実用上の壁となる。本研究では、これらを回避する潜在空間ベイズ推論を起点に、各構成要素を差し替え可能とした推論フレームワークを構築する。

## 2. 方法

ベイズの定理より、事後分布は尤度と事前分布の積に比例する。

$$
p(\theta \mid x_\mathrm{obs}) \;\propto\; L(\theta;\, x_\mathrm{obs})\, p(\theta) \tag{1}
$$

潜在空間ベイズ推論では低次元潜在変数 $z$ を介して尤度を近似する。観測側エンコーダ $q_{\phi_x}$、パラメータ側エンコーダ $q_{\phi_\theta}$、潜在事前 $p(z) = \mathcal{N}(0, I)$ を全てガウスとして構成すれば、

$$
\hat{L}(\theta;\, x_\mathrm{obs}) \;=\; \int \frac{q_{\phi_x}(z \mid x_\mathrm{obs})\, q_{\phi_\theta}(z \mid \theta)}{p(z)}\, dz \tag{2}
$$

は閉形式で評価可能となる。提案フレームワークでは事前から $(\theta, x)$ サンプルを生成し Multimodal Variational Autoencoder を学習して近似尤度 $\hat{L}$ をオフラインで構築、オンラインでは Sequential Monte Carlo が $\hat{L}$ のみを呼び事後分布から粒子をサンプリングする。サンプラー、MCMC カーネル、提案分布、尤度モデル、事前分布の各構成要素は Python の Protocol で抽象化し、後続で代替手法へ差し替え可能とした。

## 3. 結果と今後の展望

4 自由度せん断建物の屋根 FRF からの層剛性同定問題 1) で動作確認した。事後分布のメインモードは真値 $\theta = (1, 1, 1, 1)$ 近傍に集中し、識別不能性に由来する等価解も別モードとして再現された。事後平均は $(1.16, 1.02, 0.89, 1.01)$ となり、屋根に近い層の剛性は強く拘束される一方、下層は等価解の影響でやや広い分布を示した。GPU 上で粒子数 2000 の SMC は約 1.5 秒で完了し、尤度評価約 38 万回に対しシミュレータ呼び出しは 0 回である。多峰性のある不確実性定量化付き逆推定を、高速かつ並列に実施可能であることを示した。

本研究の貢献は新規手法の提案ではなく、既存の潜在空間ベイズ推論を、各構成要素を差し替え可能な比較研究の土台として再整備した点にある。これにより「どの手法がどの問題に最適か」という、これまで横断的に検証されてこなかった問いに取り組む基盤が整ったと位置づけられる。今後は尤度モデル部分の拡張比較を最優先課題とし、Normalizing Flow や拡散モデル等の代替を同一ベンチマーク上で比較する。さらに複雑系へ適用範囲を広げ、工学逆問題ごとに最適な手法構成を選択可能なライブラリへ発展させる。

## 参考文献

1) Yaoyama, T. et al., Finite element model updating of building structures under seismic excitation: A parallelized latent space-based Bayesian framework, arXiv:2604.22305 (preprint, 2026).
2) Itoi, T. et al., Bayesian structural model updating with multimodal variational autoencoder, *Comput. Methods Appl. Mech. Eng.* **429**, 117148 (2024).
3) Ching, J., Chen, Y.-C., Transitional Markov chain Monte Carlo method for Bayesian model updating, model class selection, and model averaging, *J. Eng. Mech.* **133**(7), 816-832 (2007).
