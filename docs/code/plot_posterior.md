# plot_posterior.py — 事後分布の可視化

**ファイルパス**: `src/lsbi_smc/example_shear4dof/plot_posterior.py`  
**関連学術ドキュメント**: [06_smc.md](../06_smc.md)

---

## 概要

`posterior.mat` に保存された事後分布サンプルを読み込み、**コーナープロット**（Corner Plot / ペアプロット）を生成するスクリプトです。

- 対角成分：各パラメータの周辺事後分布（ヒストグラム）
- 非対角成分：パラメータペアの同時分布（散布図 + KDE 等高線）
- 真値、事後平均、等価解も重ね合わせて表示

**実行方法**（プロジェクトルートから）：
```bash
uv run python src/lsbi_smc/example_shear4dof/plot_posterior.py
```

**前提**: `posterior.mat` が存在すること（`inference.py` で生成）  
**出力**: `posterior_plot.png`

---

## グローバル定数

```python
MAT_PATH = 'posterior.mat'
OUT_PATH = 'posterior_plot.png'
PARAM_NAMES = ['k₁', 'k₂', 'k₃', 'k₄']
LLIM, ULIM = 0.33, 3.00

# 真値（真の剛性パラメータ）
TRUE_PARAMS = np.array([1.0, 1.0, 1.0, 1.0])

# 等価解（真値と同じ屋根FRFを持つ異なるパラメータ組み合わせ）
EQUIV_SOLUTIONS = {
    'Equiv A': np.array([1.722, 0.636, 1.301, 0.701]),
    'Equiv B': np.array([1.999, 1.000, 0.500, 1.000]),
    'Equiv C': np.array([2.640, 0.647, 0.813, 0.720]),
}
```

**等価解について**: 異なる剛性パラメータの組み合わせでも、同一の屋根 FRF を生成できる場合がある（観測の不完全性から生じる識別不能問題）。事後分布が多峰性を示す原因であり、SMC がこれを捉えられるかを検証できる。

---

## 関数の詳細

### `load_pop`

```python
def load_pop() -> np.ndarray:
    data = sio.loadmat(MAT_PATH)
    return data['pop']   # shape: (2000, 4)
```

MATLAB 形式のファイルを `scipy.io.loadmat` で読み込む。

### `print_summary`

```python
def print_summary(pop: np.ndarray) -> None:
```

事後分布の統計サマリーを標準出力に表示：

```
Param     Mean      Std     2.5%      50%    97.5%
----------------------------------------------
k₁       0.9523   0.3178   0.4456   0.9953   1.4673
k₂       1.0152   0.3673   0.5209   1.0216   1.6119
...
```

### `plot_corner`

```python
def plot_corner(pop: np.ndarray) -> Figure:
```

4×4 のサブプロット配列を生成（下三角のみ使用）：

```
[ヒスト(k₁)]    [空白]       [空白]       [空白]
[散布図(k₁,k₂)] [ヒスト(k₂)] [空白]       [空白]
[散布図(k₁,k₃)] [散布図(k₂,k₃)] [ヒスト(k₃)] [空白]
[散布図(k₁,k₄)] [散布図(k₂,k₄)] [散布図(k₃,k₄)] [ヒスト(k₄)]
```

#### 対角成分（ヒストグラム）

```python
ax.hist(pop[:, col], bins=40, density=True, range=(LLIM, ULIM), ...)

# 真値（黒実線）
ax.axvline(TRUE_PARAMS[col], color='black', linewidth=2.0)

# 事後平均（赤破線）
ax.axvline(pop[:, col].mean(), color='#DD4444', linestyle='--')

# 等価解（色付き点線）
for ev, ec in zip(EQUIV_SOLUTIONS.values(), EQUIV_COLORS):
    ax.axvline(ev[col], color=ec, linestyle=':')
```

#### 非対角成分（散布図 + KDE）

```python
# 散布図（薄い青）
ax.scatter(pop[:, col], pop[:, row], alpha=0.15, s=2, ...)

# KDE 等高線（scipy.stats.gaussian_kde を使用）
kde = gaussian_kde(np.vstack([pop[:, col], pop[:, row]]))
xg, yg = np.mgrid[LLIM:ULIM:60j, LLIM:ULIM:60j]
z = kde(np.vstack([xg.ravel(), yg.ravel()])).reshape(xg.shape)
ax.contour(xg, yg, z, levels=4, ...)

# 真値（黒星マーカー）
ax.plot(TRUE_PARAMS[col], TRUE_PARAMS[row], 'k*', markersize=10)

# 等価解（三角マーカー）
for ev, ec in zip(EQUIV_SOLUTIONS.values(), EQUIV_COLORS):
    ax.plot(ev[col], ev[row], '^', color=ec, markersize=7, ...)
```

KDE の計算に失敗した場合（特異な共分散など）は `except Exception: pass` で無視し、散布図のみ表示。

---

## 出力例の読み方

正常に推論が機能した場合：
- 対角ヒストグラムは真値（黒線）付近に山が現れる
- 等価解（色付き線）付近にも副峰が見られる場合がある → 多峰性
- 非対角散布図では等価解の位置（三角マーカー）付近にも点が集まる

---

## 使用されるレイアウト技術

```python
gs = gridspec.GridSpec(n_params, n_params, figure=fig, hspace=0.08, wspace=0.08)
ax = fig.add_subplot(gs[row, col])
```

`GridSpec` を使った細かい余白調整。`col > row` の場合はサブプロット生成をスキップ（上三角は空白）。

---

## 注意点

- `gaussian_kde` は粒子数が少ない（例: 数百以下）と不安定になる場合がある
- ヒストグラムの `range=(LLIM, ULIM)` で表示範囲を物理パラメータの定義域に固定
- `dpi=150` での保存（高解像度PNG）
