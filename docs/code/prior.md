# prior.py — 階層的事前分布

**ファイルパス**: `src/lsbi_smc/smc/prior.py`  
**関連学術ドキュメント**: [01_math_prerequisites.md](../01_math_prerequisites.md), [06_smc.md](../06_smc.md)

---

## 概要

`HierarchicalPrior` クラスは、複数の確率変数ノード（`variables.py` 参照）をまとめて管理し、SMC が必要とする以下の操作を提供します：

- `sample(n)`: 事前分布から n サンプルを生成
- `lp(theta)`: 対数事前確率を計算
- `check_support(theta)`: パラメータがサポート内かどうかを確認

---

## `HierarchicalPrior` クラス

```python
class HierarchicalPrior:
    variables: list[DistVar]   # ソート済みの確率変数リスト
    names: list[str]           # 各次元の名前 (例: ['k01[0]', 'k02[0]', ...])
    index: list[str]           # 各次元が属する変数名 (例: ['k01', 'k01', 'k02', ...])
    dim: int                   # パラメータの総次元数
    n_current: int | None      # 現在の粒子数
```

### コンストラクタ

```python
def __init__(self, variables: Sequence[DistVar]):
    # 1. depth でトポロジカルソート
    depths = np.array([var.depth for var in variables])
    sorted_idx = np.argsort(depths, kind='stable')
    self.variables = [variables[i] for i in sorted_idx]

    # 2. 名前インデックスの構築
    self.names = []   # ['k01[0]', 'k02[0]', 'k03[0]', 'k04[0]']
    self.index = []   # ['k01', 'k02', 'k03', 'k04']
    for variable in self.variables:
        self.names += [variable.name + '[' + str(i) + ']'
                       for i in range(variable.dim)]
        self.index += [variable.name] * variable.dim
```

### `sample` メソッド

```python
def sample(self, n: int = 1) -> Tensor:
    for variable in self.variables:
        variable.sample(n)          # 各変数が内部状態 _values を更新
    self.n_current = n
    return self.collect()           # 全変数の値を連結して返す
```

`collect()` は各変数の `values()` を横に連結する：
```python
def collect(self) -> Tensor:
    theta = [variable.values() for variable in self.variables]
    return torch.cat(theta, dim=-1)  # shape: (n, total_dim)
```

### `lp` メソッド（対数事前確率）

```python
def lp(self, values: Tensor | None = None) -> Tensor:
    if values is None:
        values = self.collect()
    else:
        self.assign(values)    # 各変数に値を割り当て
    lp = torch.zeros((self.n_current, 1), ...)
    for variable in self.variables:
        lp += variable.lp(values[:, np.array(self.index) == variable.name])
    return lp[:, 0]   # shape: (n,)
```

各変数の対数確率を独立に計算して合計（独立な事前分布を仮定）。

### `assign` メソッド（内部用）

```python
def assign(self, theta: Tensor) -> None:
    n = len(theta)
    self.n_current = n
    for variable in self.variables:
        variable._values = theta[:, np.array(self.index) == variable.name]
```

`theta` テンソルの列を `index` に基づいて各変数の `_values` に割り当てる。`lp()` が `values` を受け取るときに呼ばれる。

### `check_support` メソッド

```python
def check_support(self, values: Tensor) -> Tensor:
    support_checks = []
    for variable in self.variables:
        check_result = variable.check_support(
            values[:, np.array(self.index) == variable.name]
        )
        support_checks.append(check_result)
    support_checks = torch.cat(support_checks, dim=-1)
    return torch.prod(support_checks, dim=1).bool()  # 全変数がサポート内なら True
```

全変数のサポートチェックの論理積（AND）。`RWMetropolisKernel` が提案粒子を棄却するために使用。

---

## `to_dict` メソッド（ユーティリティ）

```python
def to_dict(self, values=None) -> dict[str, Tensor]:
    # {'k01': Tensor(n, 1), 'k02': Tensor(n, 1), ...} の辞書を返す
```

変数名をキーとするテンソル辞書。デバッグや可視化に便利。

---

## 使用例

```python
from lsbi_smc.smc.variables import Constant, Normal
from lsbi_smc.smc.prior import HierarchicalPrior

# 4つの独立な N(0, 1) 変数
k_labels = [f'k{i:02}' for i in range(1, 5)]
variables = [Normal(l, Constant(0.0), Constant(1.0)) for l in k_labels]
prior = HierarchicalPrior(variables)

# サンプリング
samples = prior.sample(2000)   # shape: (2000, 4)

# 対数確率
lp = prior.lp(samples)         # shape: (2000,)

# サポートチェック
within = prior.check_support(samples)  # shape: (2000,), dtype bool
```

---

## インデックス構造の詳細

各変数が複数次元を持つ場合（例：2次元の `Normal`）：

```python
variables = [
    Normal('k',  Constant(np.array([0.0, 0.0])), Constant(np.array([1.0, 1.0])))
]
# → names = ['k[0]', 'k[1]']
# → index = ['k', 'k']
# → dim = 2
```

`np.array(self.index) == variable.name` でブールインデックスを作って、`theta` の対応する列を選択する仕組み。
