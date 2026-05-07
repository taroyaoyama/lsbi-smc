# simulator.py — シミュレーター並列ラッパー

**ファイルパス**: `src/lsbi_smc/simulator/simulator.py`  
**関連学術ドキュメント**: [04_structural_engineering.md](../04_structural_engineering.md)

---

## 概要

`Simulator` クラスは、任意の物理シミュレーター関数を受け取り、以下の機能を追加するラッパーです。

1. **パラメータのスケール変換**：正規化パラメータ \[0, 1\] を物理パラメータ \[LLIM, ULIM\] に変換
2. **スレッドプールによる並列実行**：複数のパラメータセットを同時にシミュレーション
3. **出力テンソルの整形**：`(n, n_floors, 1, n_freq)` の形状に統一

---

## クラス定義

```python
class Simulator:
    fun: Callable[[NDArray[float32]], NDArray[float64]]
    chunksize: int
    llim: float
    ulim: float
    workers: int
```

### コンストラクタ

```python
def __init__(
    self,
    fun: Callable,         # シミュレーター関数 (1サンプル分を処理)
    lims: list[float],     # [下限, 上限] の物理パラメータ範囲
    workers: int | None,   # 並列ワーカー数 (None = CPUコア数)
    chunksize: int = 8,    # スレッドプールのチャンクサイズ
)
```

**デフォルト `workers`**: `os.cpu_count()` を使用。GPU環境では CPUコア数で並列化する。

---

## `__call__` メソッド（主要処理）

```python
def __call__(self, theta: NDArray[float32]) -> NDArray[float32]:
```

### 処理フロー

```
入力: theta  shape (n, ndof),  値域 [0, 1]
  ↓
  1. スケール変換:  theta_phys = theta * (ulim - llim) + llim
  ↓
  2. 各行を1サンプルとして ThreadPool.map で並列実行
     [fun(theta_phys[0]), fun(theta_phys[1]), ..., fun(theta_phys[n-1])]
  ↓
  3. np.stack で (n, n_floors, n_freq) に統合
  ↓
  4. [:, :, None, :] でチャンネル次元を追加 → (n, n_floors, 1, n_freq)
  ↓
  5. float32 にキャスト
出力: sims  shape (n, n_floors, 1, n_freq)
```

### 実装コード

```python
def __call__(self, theta):
    # [0, 1] → [LLIM, ULIM] へのスケール変換
    theta = theta * (self.ulim - self.llim) + self.llim

    # スレッドプールで並列実行
    with ThreadPool(processes=self.workers) as pool:
        sims_list = pool.map(self.fun, list(theta), self.chunksize)

    # 結果を1つのテンソルに統合
    sims = np.stack(sims_list, axis=0)                      # (n, n_floors, n_freq)
    sims = sims[:, :, None, :].astype(np.float32, copy=False)  # (n, n_floors, 1, n_freq)
    return sims
```

---

## 並列化の仕組み

`multiprocessing.dummy.Pool`（= **スレッドプール**）を使用。

プロセスプールではなくスレッドプールを使う理由：
- FRF計算は NumPy/SciPy の数値演算（GILを解放する）
- プロセス起動のオーバーヘッドなしに並列化可能
- データのコピーが不要（メモリ効率が高い）

```python
from multiprocessing.dummy import Pool as ThreadPool  # スレッドベースPool
```

---

## スケール変換の詳細

| 段階 | 値域 | 説明 |
|------|------|------|
| 入力 `theta` | \[0, 1\] | ラテンハイパーキューブなどのサンプラーが生成する正規化値 |
| `theta_phys` | \[0.33, 3.00\] | 実際の剛性比（= 1000 倍で N/m 単位の剛性） |
| `fun` の入力 `x` | \[330, 3000\] | `x * 1000` で N/m スケールの剛性 |

ベンチマーク設定: `LLIM=0.33, ULIM=3.00`

---

## 出力形状の意味

```
(n, n_floors, 1, n_freq)
 │   │        │   │
 │   │        │   └── 周波数点数 (1024)
 │   │        └────── チャンネル次元 (畳み込みニューラルネットの慣例)
 │   └─────────────── 観測点 (4: 各階の加速度センサー)
 └─────────────────── サンプル数
```

学習・推論では屋根階（最終チャンネル `ch = [-1]`）のみを使用するため、実際には `(n, 1, 1, 1024)` にスライスされる。

---

## 使用例

```python
from lsbi_smc.example_shear4dof.frfshearm import frfshearm2
from lsbi_smc.simulator.simulator import Simulator

def fun(x):
    return frfshearm2(x * 1000, ms=1.0, zeta=0.02,
                      dlf=0.020, fmax=20.48, damping='rayleigh',
                      omega_target=np.array([1.0, 20.0]) * 2 * np.pi)

simulator = Simulator(fun, lims=[0.33, 3.00])

# 正規化パラメータを渡す
theta = np.array([[0.5, 0.5, 0.5, 0.5],   # → [1.665, 1.665, 1.665, 1.665]
                  [0.3, 0.7, 0.2, 0.8]])   # → [1.131, 2.199, 0.963, 2.331]

y = simulator(theta)  # shape: (2, 4, 1, 1024)
```

---

## 設計上のポイント

- `fun` は1行（1サンプル）を受け取り `(n_floors, n_freq)` を返す設計にする（Pool.map との相性のため）
- `workers=2` を `inference.py` で指定しているのは推論時の計算量が少ないため（合成観測の生成のみ）
- `create_dataset.py` では `workers` を省略（= `os.cpu_count()`）で最大並列化
