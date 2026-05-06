# frfshearm.py — N自由度せん断建物の周波数応答関数（FRF）計算

**ファイルパス**: `src/lsbi_smc/example_shear4dof/frfshearm.py`  
**関連学術ドキュメント**: [03_structural_engineering.md](../03_structural_engineering.md)

---

## 概要

このモジュールは、**N自由度せん断建物モデル**の**周波数応答関数（FRF）** を物理学的に正確に計算する関数群を提供します。構造工学の固有値解析と複素数の周波数応答の計算をNumPy/SciPyで実装しています。

パイプライン上の位置：
```
frfshearm.py (物理シミュレーター)
    ↓
simulator.py (並列ラッパー)
    ↓
create_dataset.py (データセット生成)
```

---

## `frfshearm2` 関数

```python
def frfshearm2(
    ks: npt.ArrayLike,
    ms: npt.NDArray[np.float64] | float | int,
    zeta: npt.NDArray[np.float64] | float | int,
    damping: Literal["rayleigh", "stiffness"] = "rayleigh",
    omega_target: npt.NDArray[np.float64] | None = None,
    dlf: float = 0.005,
    fmax: float = 5.12,
) -> npt.NDArray[np.float64]:
```

### 入力パラメータ

| パラメータ | 型 | 説明 |
|-----------|-----|------|
| `ks` | array-like, shape `(n_dof,)` | 各階の層剛性 \[N/m\] |
| `ms` | float または array | 各階の質量 \[kg\]。スカラーの場合は全階共通値 |
| `zeta` | float または array | 減衰比。スカラーの場合は全モード共通 |
| `damping` | `"rayleigh"` または `"stiffness"` | 減衰モデルの種類 |
| `omega_target` | array, shape `(2,)` または None | レイリー減衰の目標角振動数 \[rad/s\] |
| `dlf` | float | 周波数分解能 \[Hz\] |
| `fmax` | float | 最大周波数 \[Hz\] |

### 戻り値

```
shape: (n_dof, n_freq)
```

- `n_freq = int(fmax / dlf)` 個の周波数点における各階の **対数FRF絶対値**
- 具体的には `log(|H_ij(ω)|)` を返す（絶対値の自然対数）

ベンチマーク設定（`create_dataset.py`）では `dlf=0.02, fmax=20.48` → `n_freq=1024`

---

## 内部アルゴリズムの詳細

### Step 1: 質量行列・剛性行列の構築

```python
# 質量行列: 対角行列
m_mat = np.diag(ms)

# 剛性行列: せん断ばねの接続による三重対角型
k_mat = np.zeros((n_dof, n_dof))
k_mat[0, 0] += ks[0]                     # 1階のみ: k_1
for i in range(1, n_dof):
    k_mat[i-1:i+1, i-1:i+1] += [[ ks[i], -ks[i]],
                                  [-ks[i],  ks[i]]]  # i階: k_i を両端に追加
```

4自由度の場合、剛性行列は次の形になる：

$$K = \begin{pmatrix}
k_1+k_2 & -k_2 & 0 & 0 \\
-k_2 & k_2+k_3 & -k_3 & 0 \\
0 & -k_3 & k_3+k_4 & -k_4 \\
0 & 0 & -k_4 & k_4
\end{pmatrix}$$

### Step 2: 固有値解析

```python
lam, phi = eigh(k_mat, m_mat)   # 一般化固有値問題 K φ = λ M φ
omega_nat = np.sqrt(lam)         # 固有角振動数 ω_r = sqrt(λ_r)
```

`scipy.linalg.eigh` を使用（対称行列用の固有値解析。実数固有値が保証される）。

- `lam`: 固有値 `shape (n_dof,)` → $\omega_r^2$
- `phi`: 振動モード行列 `shape (n_dof, n_dof)` → 各列が固有ベクトル（モードシェープ）

### Step 3: レイリー減衰係数の計算

レイリー減衰: $C = a_0 M + a_1 K$

```
減衰比: ζ_r = a_0 / (2ω_r) + a_1 ω_r / 2
```

目標周波数 `omega_target = [ω_1, ω_2]` での減衰比を `zeta` にするための係数 `a_0, a_1` を連立方程式で解く：

$$\begin{pmatrix} 1/(2\omega_1) & \omega_1/2 \\ 1/(2\omega_2) & \omega_2/2 \end{pmatrix} \begin{pmatrix} a_0 \\ a_1 \end{pmatrix} = \begin{pmatrix} \zeta_1 \\ \zeta_2 \end{pmatrix}$$

```python
a_coeff = np.array([[1/(2*omega1), omega1/2],
                    [1/(2*omega2), omega2/2]])
a0, a1 = np.linalg.solve(a_coeff, [zeta1, zeta2])
```

ベンチマーク設定: `omega_target = [1.0, 20.0] * 2π` rad/s（1 Hz 〜 20 Hz）

### Step 4: FRFの計算

床応答FRFの計算（モード重ね合わせ法）：

$$H_{ij}(\omega) = \sum_{r=1}^{n} \frac{\phi_{ir} \cdot g_r}{-\omega^2 + 2j\zeta_r\omega_r\omega + \omega_r^2}$$

ここで $g_r = \phi_r^T M \mathbf{r}$（有効モード質量）、$\mathbf{r}$ は単位ベクトル（基礎加速度入力）。

```python
r = np.ones(n_dof)                           # 基礎入力ベクトル
m_r = m_mat @ r                              # 質量ベクトル
g = phi.T @ m_r                              # 有効モード質量係数

omeg = np.arange(dlf, fmax + dlf, dlf) * 2π  # 角周波数の配列

# 分母: 複素数の計算 (モード x 周波数)
denom = (-omeg**2 + 2j * zeta_r * omega_nat * omeg + omega_nat**2)
v = (1.0 / denom) * g[:, None]               # モード応答

u_mat = phi @ v                               # 物理座標へ変換
h_frf = r[:, None] + omeg**2 * u_mat         # 基礎入力からの相対変位FRF

return np.log(np.abs(h_frf))                 # 対数絶対値を返す
```

**なぜ対数をとるのか**: 生のFRF値のスケールは周波数領域で数桁変動する。対数変換でスケールを均一化し、ニューラルネットワークへの入力に適した形にする。

---

## `eigen` 関数

```python
def eigen(ks, ms) -> tuple[omega_nat, phi]:
```

固有値解析のみを行うユーティリティ関数。`frfshearm2` の内部ロジックを再利用。固有振動数とモードシェープを返す（デバッグ・可視化用途）。

---

## ベンチマーク設定まとめ

`create_dataset.py` および `inference.py` での呼び出し設定：

| 設定項目 | 値 | 意味 |
|---------|-----|------|
| `ks` | `x * 1000` | 正規化パラメータを `[kN/m]` スケールに変換 |
| `ms` | `1.0` | 全階共通質量 1 kg |
| `zeta` | `0.02` | 全モード共通減衰比 2% |
| `dlf` | `0.020` | 周波数分解能 0.02 Hz |
| `fmax` | `20.48` | 最大周波数 20.48 Hz |
| `damping` | `"rayleigh"` | レイリー減衰 |
| `omega_target` | `[1.0, 20.0] * 2π` | 目標減衰周波数 1〜20 Hz |
| 出力点数 | `1024` | `= 20.48 / 0.02` |

---

## 数値的な注意点

1. `eigh` は対称行列専用 → $K$ と $M$ の対称性が保証されている設計
2. 分母 `denom` は複素数 → Python/NumPy の複素数演算で自動処理
3. 戻り値は `float64` → 精度の高い物理量の計算
4. 出力の対数変換で共鳴峰付近の数値的挙動が安定
