# 先行研究まとめ

工学逆問題に対するベイズ的アプローチおよび、近年の機械学習援用ベイズ推論手法について、本研究との関係を整理。プレゼンの「研究背景」「既存アプローチ」「位置づけ」「限界」の各セクションでここを参照する。

---

## 1. 古典的ベイズFE モデル更新 (Bayesian FE model updating)

工学逆問題に対する確率論的なフレームワーク。

| 文献 | 内容 |
|---|---|
| **Beck & Katafygiotis (1998)** [^1] | ベイズFE updating の原典。事前分布と尤度から事後分布を更新する基本枠組み |
| **Katafygiotis & Beck (1998)** [^2] | 観測の制約による「モデル同定不能性」を提起。等価解の存在を示す |
| **Simoen, De Roeck, Lombaert (2015)** [^3] | レビュー: モデル更新における不確実性の扱い |
| **Huang et al. (2019)** [^4] | レビュー: 構造同定・損傷検出におけるベイズ推論 |
| **Kiran, Das, Bansal (2025)** [^5] | レビュー: 構造系のベイズFE updating の最新動向 |

**特徴**: 厳密な確率論的枠組み。事後分布が不確実性を表現するため、ロバストな損傷検出・将来応答予測・リスク情報に基づく維持管理に活用可能。

**課題**: MCMC ベースの推論は、各イテレーションで高コストな FE シミュレータ呼び出しが必要。何万〜何十万回の評価で実用困難。

---

## 2. 機械学習を援用した近似ベイズ推論

シミュレータの繰り返し呼び出しを回避するため、ニューラルネットワークで尤度関数または事後分布を **償却的 (amortized)** に近似するトレンド。「LBI (Likelihood-Based Inference)」「SBI (Simulation-Based Inference)」「LFI (Likelihood-Free Inference)」などの呼称。

### 2.1 Normalizing Flow ベース

| 文献 | 内容 |
|---|---|
| **Zeng, Wang, Tartakovsky, Barajas-Solano (2025a)** [^6] | NF を用いた high-dim 逆問題の amortized LFI、noisy/incomplete data 対応 |
| **Zeng, Xue, Chen (2025b)** [^7] | NF + ML ベースのリアルタイム確率論的モデル更新と損傷検出 |
| **Wang, Bi, Zhao, Dinh, Mottershead (2025)** [^8] | 深層生成モデルによるデータ駆動型確率論的モデル更新 |

### 2.2 拡散モデルベース

| 文献 | 内容 |
|---|---|
| **Wang & Bi (2026)** [^9] | 条件付き拡散モデルを用いた確率論的モデル更新 |

### 2.3 潜在空間ベイズ推論 (LSBI) — **本研究のベース**

| 文献 | 内容 |
|---|---|
| **Itoi, Amishiki, Lee, Yaoyama (2024)** [^10] | MVAE を用いた構造ベイズモデル更新 (LSBI-MVAE の先駆) |
| **Lee, Yaoyama, Matsumoto, Hida, Itoi (2024)** [^11] | 非線形ヒステリシスモデルへの LSBI 適用、単一観測ベース尤度 |
| **Lee, Yaoyama, Kitahara, Itoi (2025)** [^12] | LSBI による確率論的モデル更新の一般化 |
| **Yaoyama, Lee, Matsubara, Kodera, Ugata, Itoi (2026)** [^13] | **本研究で実装する論文**: GPU並列 SMC + MVAE 潜在尤度を組み合わせた LSBI-SMC |

**LSBI の発想**: 高次元観測 $x \in \mathbb{R}^{n_x}$ をそのまま尤度モデル化するのではなく、低次元潜在変数 $z \in \mathbb{R}^{n_z}$ ($n_z \ll n_x$) を介在させ、潜在空間で尤度を解析的に評価する。

---

## 3. 各アプローチの比較

| アプローチ | 強み | 弱み |
|---|---|---|
| 古典 MCMC + 解析尤度 | 厳密 | シミュレータコスト高、高次元観測で尤度定義困難 |
| Normalizing Flow | 表現力高、可逆変換による密度評価 | 学習が重く、ハイパラ依存 |
| 拡散モデル | 多モード分布に強い | 推論が遅い、勾配ベースサンプリング必要 |
| **LSBI (本研究)** | 潜在空間で**閉形式尤度**、サンプリング高速、GPU 並列化容易 | 潜在表現の品質に依存、表現力は NF/Diffusion より制限的 |

---

## 4. 本研究で参照する SMC・MCMC 関連

| 文献 | 内容 |
|---|---|
| **Ching & Chen (2007)** [^14] | Transitional MCMC: SMC の遷移カーネル共分散の適応則 (TMCMC) |
| **Betz, Papaioannou, Straub (2016)** [^15] | SMC の構造信頼性問題への応用 |
| **Carrera & Papaioannou (2024)** [^16] | SMC の最近の改良 |

---

## 5. 先行研究 (LSBI 関連) の限界と本研究の動機

- **既存の LSBI-SMC 実装 (Yaoyama 2026 等)** は、特定の手法構成 (MVAE + SMC + RW-MH + Ching&Chen) に固定されており、**他の選択肢との比較**が容易でない
- 工学逆問題の性質 (次元、観測数、多峰性、非線形性) は問題ごとに大きく異なるため、**どの組合せがベスト**かは事前には不明 — 本来は問題に応じて選ぶべき
- LSBI と NF・拡散モデルの **公平比較**もほとんど行われていない

→ これらを**比較可能にする土台**として、各構成要素を差し替え可能なライブラリを構築することが本研究の動機。

---

## 6. 文献リスト

[^1]: Beck, J. L., Katafygiotis, L. S., "Updating models and their uncertainties. I: Bayesian statistical framework", *J. Eng. Mech.* 124 (1998) 455–461.

[^2]: Katafygiotis, L. S., Beck, J. L., "Updating models and their uncertainties. II: Model identifiability", *J. Eng. Mech.* 124 (1998) 463–467.

[^3]: Simoen, E., De Roeck, G., Lombaert, G., "Dealing with uncertainty in model updating for damage assessment: A review", *Mech. Syst. Signal Process.* 56–57 (2015) 123–149.

[^4]: Huang, Y., Shao, C., Wu, B., Beck, J. L., Li, H., "State-of-the-art review on Bayesian inference in structural system identification and damage assessment", *Adv. Struct. Eng.* 22 (2019) 1329–1351.

[^5]: Kiran, R. P., Das, A., Bansal, S., "A state-of-the-art review of Bayesian finite element model updating techniques for structural systems", *Probab. Eng. Mech.* 80 (2025) 103761.

[^6]: Zeng, J., Wang, Y., Tartakovsky, A. M., Barajas-Solano, D. A., "Solving high-dimensional inverse problems using amortized likelihood-free inference with noisy and incomplete data", *Comput. Methods Appl. Mech. Eng.* 443 (2025a) 118064.

[^7]: Zeng, J., Xue, K., Chen, H., "Real-time probabilistic model updating and damage detection using machine learning-based likelihood-free inference", *Mech. Syst. Signal Process.* 230 (2025b) 112612.

[^8]: Wang, T., Bi, S., Zhao, Y., Dinh, L., Mottershead, J., "Data-driven stochastic model updating and damage detection with deep generative model", *Mech. Syst. Signal Process.* 232 (2025) 112743.

[^9]: Wang, T., Bi, S., "Stochastic model updating using conditional diffusion-based probabilistic generative models", *Mech. Syst. Signal Process.* 246 (2026) 113891.

[^10]: Itoi, T., Amishiki, K., Lee, S., Yaoyama, T., "Bayesian structural model updating with multimodal variational autoencoder", *Comput. Methods Appl. Mech. Eng.* 429 (2024) 117148.

[^11]: Lee, S., Yaoyama, T., Matsumoto, Y., Hida, T., Itoi, T., "Latent space-based likelihood estimation using a single observation for Bayesian updating of a nonlinear hysteretic model", *ASCE ASME J. Risk Uncertain. Eng. Syst. A Civ. Eng.* 10 (2024) 04024072.

[^12]: Lee, S., Yaoyama, T., Kitahara, M., Itoi, T., "Latent space-based stochastic model updating", *Mech. Syst. Signal Process.* 235 (2025) 112841.

[^13]: Yaoyama, T., Lee, S., Matsubara, M., Kodera, K., Ugata, T., Itoi, T., "Finite element model updating of building structures under seismic excitation: A parallelized latent space-based Bayesian framework", *Nucl. Eng. Des.* (2026). **← 本研究の対象論文**

[^14]: Ching, J., Chen, Y.-C., "Transitional Markov chain Monte Carlo method for Bayesian model updating, model class selection, and model averaging", *J. Eng. Mech.* 133 (2007) 816–832.

[^15]: Betz, W., Papaioannou, I., Straub, D., "Transitional Markov chain Monte Carlo: Observations and improvements", *J. Eng. Mech.* 142 (2016) 04016016.

[^16]: Carrera, B., Papaioannou, I., (2024). Recent improvements to SMC samplers in structural problems.
