# 線形代数とNumPy/SciPyの基礎

このドキュメントは、LSBI-SMC のコードを読むのに必要な **線形代数の基礎** と、それを実装するための **NumPy/SciPy の使い方** を、初学者でもわかるように説明する。

構造工学・機械学習のどちらにも線形代数は不可欠で、本プロジェクトでは特に以下が頻出する：

- **ベクトル・行列の演算**（運動方程式、ニューラルネットの線形変換）
- **固有値問題**（建物の振動モード解析）
- **フーリエ変換**（FRF の導出、周波数領域での応答計算）
- **行列分解**（数値的安定性、効率的なサンプリング）

概念の流れ：

```
ベクトル・行列・テンソル → 線形変換 → 固有値問題 → フーリエ変換 → 行列分解
        ↓                    ↓               ↓               ↓               ↓
NumPy配列 → ブロードキャスト → scipy.linalg → （周波数領域） → scipy.stats
```

フーリエ変換は厳密には線形代数の話ではないが、「関数を直交基底（正弦波）で展開する」という発想は固有値分解の連続版にあたるので、本ドキュメントでまとめて扱う。

---

## 1. ベクトル・行列・テンソル

### 1.1 数学的な定義

| 名前 | 次元 | 例 | 形状（NumPy） |
|------|------|-----|---------------|
| **スカラー** | 0 | $a = 3.14$ | `()` |
| **ベクトル** | 1 | $\mathbf{v} = (v_1, v_2, v_3)$ | `(3,)` |
| **行列** | 2 | $A_{ij}$（行×列） | `(m, n)` |
| **テンソル** | 3以上 | $T_{ijk}$ | `(d_1, d_2, d_3, ...)` |

本プロジェクトでは4階テンソルが頻出する。例えば学習データの形状 `(100000, 4, 1, 1024)` は「100,000 サンプル × 4階分 × 高さ1 × 周波数1024点」。

### 1.2 NumPy での表現

```python
import numpy as np

a = np.array(3.14)                        # スカラー、shape = ()
v = np.array([1.0, 2.0, 3.0])              # ベクトル、shape = (3,)
A = np.array([[1, 2], [3, 4]])             # 行列、shape = (2, 2)
T = np.zeros((100, 4, 1, 1024))            # 4階テンソル
```

`shape` 属性で形状を、`ndim` で次元数を、`dtype` で要素の型を確認できる：

```python
print(A.shape)   # (2, 2)
print(A.ndim)    # 2
print(A.dtype)   # int64（または float64）
```

---

## 2. 行列演算の基本

### 2.1 行列・ベクトル積

行列 $A \in \mathbb{R}^{m \times n}$ とベクトル $\mathbf{x} \in \mathbb{R}^n$ の積は：

$$(A\mathbf{x})_i = \sum_{j=1}^n A_{ij} x_j$$

NumPy では `@` 演算子（または `np.matmul`）：

```python
A = np.array([[1, 2], [3, 4]])
x = np.array([5, 6])
y = A @ x          # = [1*5 + 2*6, 3*5 + 4*6] = [17, 39]
```

> **注意：** `*` は要素ごとの積（アダマール積）であり、行列積ではない。これを混同するのが初学者の典型的バグ。

### 2.2 転置

行と列を入れ替える操作：$(A^T)_{ij} = A_{ji}$

```python
A.T              # 転置
A.transpose()    # 同じ
```

### 2.3 逆行列・連立方程式

逆行列 $A^{-1}$ は $AA^{-1} = I$ を満たす行列。連立方程式 $A\mathbf{x} = \mathbf{b}$ の解は $\mathbf{x} = A^{-1}\mathbf{b}$ だが、**実装では `np.linalg.solve` を使う**：

```python
# 悪い例：逆行列を計算してから掛ける（数値的に不安定・遅い）
x = np.linalg.inv(A) @ b

# 良い例：直接解く（LU分解を内部で使い、より安定）
x = np.linalg.solve(A, b)
```

本プロジェクトでは [frfshearm.py:66](../src/lsbi_smc/example_shear4dof/frfshearm.py#L66) でレイリー減衰係数を求める際に使用：

```python
a0, a1 = np.linalg.solve(a_coeff, b)
```

### 2.4 対角行列・単位行列

```python
np.diag([1, 2, 3])           # 対角成分 (1,2,3) の対角行列
np.eye(4)                     # 4×4 の単位行列
```

[frfshearm.py:28](../src/lsbi_smc/example_shear4dof/frfshearm.py#L28) の `m_mat = np.diag(ms)` は質量を対角成分に並べた質量行列の作成。

---

## 3. ブロードキャスティング

### 3.1 形状が違う配列の演算

NumPy（や PyTorch）の最重要機能の一つ。形状が異なる配列でも、**特定のルールに従って自動的に拡張**して演算してくれる。

**ルール：** 末尾の次元から比較し、各次元が（i）等しい、または（ii）どちらかが1、なら演算可能。サイズ1の次元は他方に合わせて複製される。

### 3.2 具体例

```python
A = np.array([[1, 2, 3],
              [4, 5, 6]])    # shape = (2, 3)

# ベクトル + 行列
v = np.array([10, 20, 30])    # shape = (3,)
A + v                          # 各行に v を加算
# [[11, 22, 33], [14, 25, 36]]

# 列ベクトル + 行ベクトル
col = np.array([[1], [2]])    # shape = (2, 1)
row = np.array([10, 20, 30])   # shape = (3,)
col + row                      # shape = (2, 3) に自動拡張
# [[11, 21, 31], [12, 22, 32]]
```

### 3.3 `[:, None]` と `[None, :]` の意味

新しい軸を追加するイディオム。**ブロードキャスティングを意図通りに使うために必須**：

```python
v = np.array([1, 2, 3])         # shape = (3,)
v[:, None]                       # shape = (3, 1)  列ベクトル化
v[None, :]                       # shape = (1, 3)  行ベクトル化

# 外積を計算
v[:, None] * v[None, :]          # shape = (3, 3)
```

### 3.4 本プロジェクトでの使用例

[frfshearm.py:79-82](../src/lsbi_smc/example_shear4dof/frfshearm.py#L79-L82) でモード応答を一括計算する箇所：

```python
omega_nat_col = omega_nat[:, None]   # (n_dof, 1)  ← モード方向
zeta_r_col = zeta_r[:, None]         # (n_dof, 1)
denom = (
    -(omeg[None, :] ** 2) +          # (1, n_freq) ← 周波数方向
    2j * zeta_r_col * omega_nat_col * omeg[None, :] +
    omega_nat_col**2
)
# 結果は shape (n_dof, n_freq) になる：各モード×各周波数の応答
```

「モード次元と周波数次元を別々の軸に分けて、ブロードキャスティングで全組み合わせを一気に計算する」という典型的な高速化パターン。明示的な for ループより数十倍速い。

---

## 4. 固有値問題

### 4.1 通常の固有値問題

行列 $A$ に対して：

$$A\mathbf{v} = \lambda \mathbf{v}$$

を満たす **固有値** $\lambda$ と **固有ベクトル** $\mathbf{v}$ を求める問題。

**直感：** $\mathbf{v}$ は「$A$ で変換しても向きが変わらない特別な方向」、$\lambda$ は「その方向で何倍に伸びるか」。

### 4.2 一般化固有値問題

2つの行列 $A, B$ に対して：

$$A\mathbf{v} = \lambda B\mathbf{v}$$

通常の問題（$B = I$）の拡張。**構造工学では、運動方程式 $M\ddot{\mathbf{u}} + K\mathbf{u} = 0$ から自然に出現する**：

調和振動 $\mathbf{u} = \boldsymbol{\phi} \sin(\omega t)$ を仮定すると：

$$K\boldsymbol{\phi} = \omega^2 M\boldsymbol{\phi}$$

ここで $A = K$（剛性行列）, $B = M$（質量行列）, $\lambda = \omega^2$（固有角振動数の2乗）。

### 4.3 `scipy.linalg.eigh` の使い方

`eigh` は **対称（エルミート）行列ペアに最適化された** 固有値ソルバー。`np.linalg.eig` よりも高速かつ数値的に安定：

```python
from scipy.linalg import eigh

# 通常の固有値問題：A v = λ v
lam, vec = eigh(A)

# 一般化固有値問題：A v = λ B v
lam, vec = eigh(A, B)
```

戻り値：
- `lam`: 固有値（昇順にソートされる）、shape `(n,)`
- `vec`: 固有ベクトルを **列方向** に並べた行列、shape `(n, n)`。`vec[:, i]` が `lam[i]` に対応する固有ベクトル

[frfshearm.py:39](../src/lsbi_smc/example_shear4dof/frfshearm.py#L39) での使用：

```python
lam, phi = eigh(k_mat, m_mat)        # 一般化固有値問題
omega_nat = np.sqrt(lam)              # 固有角振動数 [rad/s]
# phi[:, i] = i 次のモード形状ベクトル
```

### 4.4 なぜ一般化固有値問題は標準形に変換できるか

質量行列 $M$ が正定値対称なら、Cholesky 分解 $M = LL^T$ を使って：

$$L^{-1}KL^{-T}(L^T\boldsymbol{\phi}) = \lambda(L^T\boldsymbol{\phi})$$

という標準的な対称固有値問題に変換できる。`scipy.linalg.eigh(K, M)` は内部的にこの変換を行ってから解いている。

### 4.5 固有ベクトルの直交性（モード分解の核心）

対称行列 $A$, $B$ の一般化固有ベクトル $\Phi = [\phi_1, \phi_2, \ldots, \phi_n]$ には次の性質がある（**M-直交性、K-直交性**）：

$$\Phi^T M \Phi = I, \qquad \Phi^T K \Phi = \mathrm{diag}(\omega_i^2)$$

つまり「$\Phi$ で座標変換すると、$M$ と $K$ が同時に対角化される」。これがモーダル解析の数学的根拠で、**レイリー減衰の説明**（[04_structural_engineering.md §3.5](04_structural_engineering.md)）でも本質的な役割を果たす。

---

## 5. フーリエ変換の基礎

線形代数で「ベクトルを直交基底で展開する」のと同じ発想で、時間の関数 $f(t)$ を **正弦波という連続的な「基底」で展開** する操作がフーリエ変換。本プロジェクトでは運動方程式を周波数領域で解いて FRF を導出するために使う（[04_structural_engineering.md §4.2.1](04_structural_engineering.md)）。

### 5.1 定義と直感

ある時間波形 $f(t)$ に対して

$$\hat{f}(\omega) = \int_{-\infty}^{\infty} f(t)\,e^{-j\omega t}\,dt$$

- 入力：時間の関数 $f(t)$
- 出力：周波数の関数 $\hat{f}(\omega)$（一般には複素数）
- 振幅 $|\hat{f}(\omega)|$ は「周波数 $\omega$ で揺れている強さ」

逆変換（周波数 → 時間）も同様に積分で定義され、両者は1対1対応する。**情報を失わずに「視点」を変えているだけ**。

直感的には「波形 $f(t)$ を $e^{j\omega t}$（角振動数 $\omega$ の純粋な正弦波）の重ね合わせとして書き表すと、それぞれの $\omega$ 成分がどれだけ入っているか」を測る操作。線形代数の「ベクトルを直交基底に分解する内積」の連続版と思えばよい。

### 5.2 鍵となる性質：微分が掛け算になる

$f(t)$ を $e^{j\omega t}$ の重ね合わせと見ると、時間微分 $d/dt$ は $e^{j\omega t}$ に対して $j\omega$ を掛けるだけの操作になる：

$$\frac{d}{dt}\,e^{j\omega t} = j\omega\,e^{j\omega t}$$

これが各成分について成り立つので、線形性により

$$\boxed{\;\mathcal{F}[\dot{f}] = j\omega\,\hat{f}, \qquad \mathcal{F}[\ddot{f}] = (j\omega)^2\,\hat{f} = -\omega^2\,\hat{f}\;}$$

これがフーリエ変換を使う最大の理由：**微分方程式（解くのが面倒）が代数方程式（割り算で解ける）に変わる**。

> **固有値分解との対応：** $e^{j\omega t}$ は微分演算子 $d/dt$ の固有関数で、固有値が $j\omega$。フーリエ変換は「$d/dt$ という線形演算子を固有関数基底で対角化する操作」と見なせる。§4 の固有値問題が**離散の世界**（行列を固有ベクトル基底で対角化）だったのに対し、フーリエ変換は**連続の世界**での同じ操作。だから「座標変換で微分作用素 $\frac{d^2}{dt^2} + 2\zeta\omega_r \frac{d}{dt} + \omega_r^2$ が $-\omega^2 + 2j\zeta\omega_r\omega + \omega_r^2$ という単なる複素数の掛け算になる」のは、「行列が固有基底で対角化される」のと同じ現象。

### 5.3 線形定数係数 ODE への適用パターン

本プロジェクトで使う典型例：2階線形定数係数の常微分方程式

$$\ddot{q}(t) + 2\zeta\omega_n\dot{q}(t) + \omega_n^2\,q(t) = f(t)$$

の両辺をフーリエ変換すると、5.2 の規則で各項が

| 時間領域 | 周波数領域 |
|----|----|
| $q(t)$ | $\hat{q}(\omega)$ |
| $\dot{q}(t)$ | $j\omega\,\hat{q}(\omega)$ |
| $\ddot{q}(t)$ | $-\omega^2\,\hat{q}(\omega)$ |
| $f(t)$ | $\hat{f}(\omega)$ |

に置き換わり、$\hat{q}$ について整理するだけで

$$\hat{q}(\omega) = \frac{\hat{f}(\omega)}{\omega_n^2 - \omega^2 + 2j\zeta\omega_n\omega}$$

と**割り算一発で陽に解ける**。分母 $\omega_n^2 - \omega^2 + 2j\zeta\omega_n\omega$ は「1自由度系の伝達関数」と呼ばれ、$\omega = \omega_n$ で実部が $0$ になって絶対値が極小になる。これが**共振**の数式的正体。

このパターンは [04_structural_engineering.md §4.2.1](04_structural_engineering.md) の FRF 導出で、各モードに対してそのまま適用される。

---

## 6. 行列分解

### 6.1 Cholesky分解

正定値対称行列 $A$ を下三角行列 $L$ で分解：

$$A = LL^T$$

連立方程式やサンプリングを高速化するために使われる。NumPy では：

```python
L = np.linalg.cholesky(A)             # A = L @ L.T
```

### 6.2 多変量正規分布のサンプリング

平均 $\boldsymbol{\mu}$、共分散 $\Sigma$ の多変量正規分布からサンプリングするには：

1. 標準正規 $\boldsymbol{\epsilon} \sim \mathcal{N}(0, I)$ から引く
2. $\Sigma = LL^T$ と Cholesky 分解
3. $\mathbf{x} = \boldsymbol{\mu} + L\boldsymbol{\epsilon}$ とすれば $\mathbf{x} \sim \mathcal{N}(\boldsymbol{\mu}, \Sigma)$

これは VAE の **再パラメータ化トリック**（[03_ml_prerequisites.md §2.3](03_ml_prerequisites.md)）の基礎。1次元なら：

```python
z = mu + sigma * epsilon         # epsilon ~ N(0, 1)
```

本プロジェクトの SMC でも、提案分布の共分散行列に Cholesky 分解（PyTorch の `MultivariateNormal`）を使っている。

---

## 7. NumPy の便利な機能

### 7.1 配列生成

```python
np.zeros((3, 4))                  # 全要素0、shape (3, 4)
np.ones(5)                        # 全要素1
np.full(10, 3.14)                  # 全要素 3.14
np.arange(0, 10, 0.5)              # 等差数列 [0, 0.5, 1.0, ..., 9.5]
np.linspace(0, 1, 11)              # 区間 [0, 1] を 11等分
```

### 7.2 形状操作

```python
A = np.zeros((6,))
A.reshape(2, 3)                    # 形状を (2, 3) に変える
A[None, :]                          # 軸を追加 → (1, 6)
np.expand_dims(A, axis=0)          # 同じ
np.squeeze(B)                       # サイズ1の軸を削除
A[..., :10]                         # `...` は省略記号（残りの軸全て）
```

### 7.3 集約関数（reduction）

```python
A.sum()                            # 全要素の和
A.sum(axis=0)                      # 軸0方向に和をとる（行を潰す）
A.mean(axis=-1)                    # 最後の軸方向に平均
A.max(), A.min(), A.std()
```

`axis` パラメータの理解は重要。「指定した軸を `潰す`」と覚える：

```python
A = np.zeros((3, 4, 5))
A.sum(axis=0).shape                # (4, 5)  ← 軸0が消える
A.sum(axis=1).shape                # (3, 5)  ← 軸1が消える
A.sum(axis=(0, 2)).shape           # (4,)    ← 軸0と軸2が消える
```

### 7.4 インデックス・スライシング

```python
A[0]                               # 0行目
A[:, 1]                            # 1列目
A[0:2, 1:3]                        # 部分行列
A[[-1]]                            # 最後の要素（リストで渡すと軸を保つ）
A[A > 0]                           # 条件にマッチする要素のみ
```

[train.py](../src/lsbi_smc/example_shear4dof/train.py) での `y[:, [-1], :, :]` は「全サンプル × 最後の階（屋根）のみを軸を保ったまま抽出」。`[-1]` だと軸が潰れて `(N, 1024)` になるが、`[[-1]]` だと `(N, 1, 1, 1024)` を保てる。

### 7.5 入出力

```python
np.savez("data.npz", x=x_array, y=y_array)   # 複数配列を保存
data = np.load("data.npz")
x = data["x"]
```

[create_dataset.py](../src/lsbi_smc/example_shear4dof/create_dataset.py) で学習データを `train_data.npz` として保存している。

---

## 8. SciPy の主要モジュール

本プロジェクトで使う SciPy の機能を概観する。

### 8.1 `scipy.linalg`

NumPy にもある `np.linalg` の上位互換的な線形代数ライブラリ。`eigh`, `solve`, `cholesky` など、より多機能・高性能。

```python
from scipy.linalg import eigh, cholesky, solve
```

本プロジェクトでは `eigh` を [frfshearm.py:39](../src/lsbi_smc/example_shear4dof/frfshearm.py#L39) で使用。

### 8.2 `scipy.stats`

確率分布の操作・サンプリングを担う。

```python
from scipy.stats import norm

norm.pdf(x, loc=0, scale=1)        # 正規分布の確率密度関数
norm.logpdf(x, loc=0, scale=1)     # その対数（実装ではこちらが多用される）
norm.cdf(x)                         # 累積分布関数
norm.ppf(q)                         # その逆関数（quantile function）
norm.rvs(size=10)                   # サンプリング
```

[create_dataset.py:](../src/lsbi_smc/example_shear4dof/create_dataset.py) でノイズ生成：

```python
y_sim_n = y_sim + noise_level * norm.rvs(size=y_sim.shape)
```

### 8.3 `scipy.stats.qmc`（準モンテカルロ）

ラテン超方格法（Latin Hypercube Sampling）など、空間を均一にカバーするサンプリング手法：

```python
from scipy.stats import qmc

sampler = qmc.LatinHypercube(d=4)
x = sampler.random(n=100000)        # shape (100000, 4) を [0,1]^4 でサンプル
```

[create_dataset.py](../src/lsbi_smc/example_shear4dof/create_dataset.py) でパラメータサンプリングに使用。一様乱数より少ないサンプルで空間を均等にカバーできる。

### 8.4 `scipy.io`

MATLAB 形式（`.mat`）のファイル入出力：

```python
import scipy.io as sio
sio.savemat("posterior.mat", {"theta": theta_array})
data = sio.loadmat("posterior.mat")
```

[inference.py](../src/lsbi_smc/example_shear4dof/inference.py) で SMC の推論結果を保存している。

---

## 9. NumPy ⇄ PyTorch の対応

PyTorch のテンソルは NumPy 配列とほぼ同じ API を持つ。本プロジェクトでは、シミュレーション側（NumPy）と学習・推論側（PyTorch）でデータを変換する。

| 操作 | NumPy | PyTorch |
|------|-------|---------|
| 配列生成 | `np.zeros((3,4))` | `torch.zeros(3, 4)` |
| 行列積 | `A @ B` | `A @ B` |
| 形状変更 | `A.reshape(...)` | `A.reshape(...)` または `A.view(...)` |
| 軸追加 | `A[:, None]` | `A[:, None]`（同じ）または `A.unsqueeze(1)` |
| 集約 | `A.sum(axis=0)` | `A.sum(dim=0)` |
| 相互変換 | `tensor.numpy()` | `torch.from_numpy(arr)` |

PyTorch 特有の概念：

- **`requires_grad`**：勾配計算を追跡するか
- **`.to(device)`**：CPU ⇄ GPU 間でテンソルを移動
- **`.detach()`**：計算グラフから切り離す（勾配を流さない）

これらは [03_ml_prerequisites.md](03_ml_prerequisites.md) と [05_mvae.md](05_mvae.md) で詳述する。

---

## 10. 数値計算の落とし穴

### 10.1 浮動小数点の桁落ち

「ほぼ等しい2つの値の引き算」で精度が大きく落ちる：

```python
1e16 + 1.0 - 1e16    # → 0.0（1.0 が消える）
```

対策：
- 引き算の順序を工夫する
- 対数スケールで計算する（後述）
- `numpy.float64` を使う（デフォルト）

### 10.2 オーバーフロー・アンダーフロー

確率密度の積は容易に 0 や ∞ になる：

```python
1e-300 * 1e-300       # → 0.0（アンダーフロー）
```

**対策：対数スケールで扱う。** 確率の積を対数の和に変える：

$$\log(p_1 \cdot p_2 \cdots p_n) = \sum_i \log p_i$$

本プロジェクトでも、尤度・事前分布は **すべて log で実装** されている（`logpdf`, `log_prob` など）。

### 10.3 `logsumexp` トリック

「対数の和」を「指数の和の対数」に戻すと、再びオーバーフローの危険がある：

$$\log\!\left(\sum_i e^{a_i}\right)$$

直接計算すると $e^{a_i}$ で爆発する。代わりに最大値を引いて：

$$\log\!\left(\sum_i e^{a_i}\right) = a_{\max} + \log\!\left(\sum_i e^{a_i - a_{\max}}\right)$$

NumPy なら `scipy.special.logsumexp`、PyTorch なら `torch.logsumexp` を使えばこれを安全に計算してくれる。SMC の重み正規化でも頻出のテクニック。

### 10.4 数値安定化のイディオム

本プロジェクトのコードで頻出するパターン：

```python
torch.clamp(vr, min=eps)                        # 分散を下からクリップ（log/除算前）
torch.nan_to_num(z, neginf=-1e30, posinf=1e30)  # NaN/Inf を有限値に
cov * b**2 + eps * I                             # 共分散の正則化（特異点回避）
```

これらは [CLAUDE.md](../CLAUDE.md) の「Numerical Stability Patterns」にもまとめられている。

---

## まとめ：本プロジェクトでの線形代数の使われ方

```
NumPy/SciPy（CPU側）                        PyTorch（GPU側）
─────────────────────                       ────────────────
frfshearm2:                                  MVAE forward:
  ・eigh(K, M)            一般化固有値問題    ・@ 演算子          線形変換
  ・@ 演算子              モード重ね合わせ    ・[:, None] etc.   ブロードキャスト
  ・broadcast (omega)     全周波数を一括計算  ・logsumexp        数値安定性

create_dataset:                              SMC（CPU/GPU両方）:
  ・qmc.LatinHypercube    パラメータサンプル  ・cholesky          多変量正規サンプリング
  ・norm.rvs              ノイズ生成          ・log_prob          対数尤度計算
```

各概念の物理的な意味は [04_structural_engineering.md](04_structural_engineering.md)、確率論的な意味は [02_probability.md](02_probability.md) で詳述する。
