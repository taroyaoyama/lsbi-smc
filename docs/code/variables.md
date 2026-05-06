# variables.py — 確率変数クラス群

**ファイルパス**: `src/lsbi_smc/smc/variables.py`  
**関連学術ドキュメント**: [01_math_prerequisites.md](../01_math_prerequisites.md), [06_smc.md](../06_smc.md)

---

## 概要

SMC の事前分布を構成するための**確率変数クラス群**を定義します。各クラスはDAG（有向非巡回グラフ）のノードとして機能し、`HierarchicalPrior` が組み合わせて複雑な階層事前分布を表現できます。

すべての確率変数クラスは `DistVar` プロトコルに従い、共通インタフェースを持ちます：
- `sample(n)`: n 個のサンプルを生成し、内部状態（`_values`）に保存
- `lp(values)`: 対数確率密度を計算
- `check_support(values)`: サポート内かどうかを確認

---

## `DistVar` プロトコル

```python
class DistVar(Protocol):
    depth: int          # DAG の深さ（親ノードの最大深さ + 1）
    dim: int            # このノードの次元数
    name: str           # 変数名（HierarchicalPrior でのインデックスに使用）
    _values: Tensor | None

    def values(self, n=None) -> Tensor: ...
    def sample(self, n, detach=False) -> Tensor | None: ...
    def lp(self, values=None) -> Tensor: ...
    def check_support(self, values) -> Tensor: ...
```

`depth` 属性により `HierarchicalPrior` が変数をトポロジカル順にソートし、依存関係の正しい順序でサンプリングを実行する。

---

## `ConstantVector` / `Constant` クラス

```python
class ConstantVector:
    def __init__(self, value: ArrayLike | Tensor, device=None, dtype=torch.float32)
    def values(self, n=1, device=None) -> Tensor  # (n, dim) に tile して返す

class Constant(ConstantVector):
    def __init__(self, value: float | int)   # スカラーの定数
```

確率分布ではなく定数値を表すノード。`depth=0`（ルートノード）として他の変数のパラメータ（平均・分散など）に使用。

**使用例**:
```python
Constant(0.0)   # 平均 0
Constant(1.0)   # 標準偏差 1
```

---

## `Uniform` クラス

$$\theta \sim \mathcal{U}(a, b)$$

```python
class Uniform:
    def __init__(self, name: str, lower: ConstantVector, upper: ConstantVector)
```

| メソッド | 実装 |
|---------|------|
| `sample(n)` | `dist.Uniform(lower, upper).sample()` |
| `lp(values)` | `dist.Uniform(lower, upper).log_prob(values)` |
| `check_support(values)` | `(values >= lower) & (values <= upper)` |

`depth = max(lower.depth, upper.depth) + 1` で親ノードからの深さを計算。

**使用例** （物理空間での一様事前分布）:
```python
Uniform('k01', Constant(0.33), Constant(3.00))
```

---

## `Normal` クラス

$$\theta \sim \mathcal{N}(\mu, \sigma^2)$$

```python
class Normal:
    def __init__(self, name: str, mu: ConstantVector, sg: ConstantVector)
    # sg は標準偏差（σ）であり分散（σ²）ではない
```

| メソッド | 実装 |
|---------|------|
| `sample(n)` | `dist.Normal(mu, sg).sample()` |
| `lp(values)` | `dist.Normal(mu, sg).log_prob(values)` |
| `check_support(values)` | 常に `True`（全実数がサポート） |

`inference.py` での使用（SMC の潜在空間での事前分布）:
```python
Normal('k01', Constant(0.0), Constant(1.0))  # N(0, 1)
```

---

## `HalfNormal` クラス（`Normal` のサブクラス）

$$\theta \sim \mathcal{HN}(\sigma), \quad \theta \geq 0$$

```python
class HalfNormal(Normal):
    def __init__(self, name: str, sg: ConstantVector)
    # 内部で Normal(name, Constant(0.0), sg) を呼ぶ
```

| メソッド | 実装 |
|---------|------|
| `sample(n)` | `dist.HalfNormal(sg).sample()` |
| `lp(values)` | `dist.HalfNormal(sg).log_prob(values)` |
| `check_support(values)` | `values >= 0` |

正の値のみを取るパラメータ（例：標準偏差、スケール）の事前分布として使用。

---

## `Laplace` クラス

$$\theta \sim \text{Laplace}(\mu, b), \quad b > 0$$

```python
class Laplace:
    def __init__(self, name: str, mu: ConstantVector, b: ConstantVector)
    # mu: 位置パラメータ（ロケーション）
    # b:  スケールパラメータ（> 0）
```

正規分布より裾が重い分布。スパース性（多くの値がゼロ付近に集中）を誘導する事前分布として有用。

---

## `Exponential` クラス

$$\theta \sim \text{Exp}(\lambda), \quad \theta \geq 0, \quad \lambda > 0$$

```python
class Exponential:
    def __init__(self, name: str, rate: ConstantVector)
    # rate: λ（レートパラメータ = 1/平均）
```

| メソッド | 実装 |
|---------|------|
| `check_support(values)` | `values >= 0` |

正の実数のみを取るパラメータの事前分布。

---

## 各クラスの比較

| クラス | サポート | 特徴 |
|--------|---------|------|
| `Uniform` | `[a, b]` | 有界の一様分布 |
| `Normal` | $\mathbb{R}$ | 標準的な正規分布 |
| `HalfNormal` | $[0, \infty)$ | 正の値のみ |
| `Laplace` | $\mathbb{R}$ | 正規より裾が重い |
| `Exponential` | $[0, \infty)$ | 指数関数的な減少 |

---

## DAG（有向非巡回グラフ）の仕組み

```
Constant(0.0)  depth=0    Constant(1.0)  depth=0
      └─────────────────────────┘
                    ↓
         Normal('k01', ...)     depth=1
```

`HierarchicalPrior` はこの `depth` でソートしてサンプリングの順序を決定する。深い（依存している）変数は後でサンプリングされる。

```python
# HierarchicalPrior.__init__ での処理
depths = np.array([var.depth for var in variables])
sorted_idx = np.argsort(depths, kind='stable')
self.variables = [variables[i] for i in sorted_idx]
```

**注意**: 現在のベンチマーク実装では `Constant` と `Normal` のみ使用。`Uniform` や他のクラスは拡張用。
