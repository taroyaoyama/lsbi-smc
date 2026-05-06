# create_dataset.py — 学習データセットの生成

**ファイルパス**: `src/lsbi_smc/example_shear4dof/create_dataset.py`  
**関連学術ドキュメント**: [03_structural_engineering.md](../03_structural_engineering.md), [00_overview.md](../00_overview.md)

---

## 概要

MVAE の学習に必要な **(θ, FRF) ペアのデータセット** を大量生成するスクリプトです。ラテンハイパーキューブサンプリング（LHS）で網羅的にパラメータ空間を探索し、物理シミュレーターで FRF を計算します。

**実行方法**（プロジェクトルートから）：
```bash
uv run python src/lsbi_smc/example_shear4dof/create_dataset.py
```

**出力**: `train_data.npz`（プロジェクトルート）

---

## 全体の処理フロー

```
1. 物理定数の設定 (LLIM, ULIM)
      ↓
2. シミュレーター関数の定義 (fun)
      ↓
3. Simulator ラッパーの作成
      ↓
4. LHS で 100,000 サンプル生成
      ↓
5. シミュレーター並列実行 → FRF 計算
      ↓
6. ガウスノイズ付加
      ↓
7. .npz ファイルに保存
```

---

## コードの詳細

### 定数定義

```python
LLIM, ULIM = 0.33, 3.00
```

各階剛性の正規化された下限・上限。物理的には：
- `LLIM = 0.33` → `330 N/m`（最小剛性）
- `ULIM = 3.00` → `3000 N/m`（最大剛性）

「真値」 θ = 1.0 は `(1.0 - 0.33) / (3.00 - 0.33) = 0.2509` に正規化される。

### シミュレーター関数 `fun`

```python
def fun(x: NDArray[float32]) -> NDArray[float64]:
    return frfshearm2(
        x * 1000,               # 正規化値 → 実剛性 [N/m]
        ms=1.0,                 # 各階質量: 1 kg
        zeta=0.02,              # 減衰比: 2%
        dlf=0.020,              # 周波数分解能: 0.02 Hz
        fmax=20.48,             # 最大周波数: 20.48 Hz → 1024 点
        damping='rayleigh',
        omega_target=np.array([1.0, 20.0]) * 2 * np.pi,  # 目標減衰周波数
    )
```

この関数は `Simulator` ラッパーに渡される。`Simulator.__call__` 内で `fun` を呼ぶ前にスケール変換が適用されるため、`fun` は `[LLIM, ULIM]` の値を受け取る（`x * 1000` でさらに N/m スケールに変換）。

### ラテンハイパーキューブサンプリング（LHS）

```python
ndof = 4
n_sim = 100000
sampler = qmc.LatinHypercube(d=ndof)
x_sim = sampler.random(n_sim)   # shape: (100000, 4), 値域 [0, 1]
```

**なぜ LHS か**: ランダムサンプリングより均一にパラメータ空間をカバーする。各次元を `n_sim` 等分した格子上で各区間から1点ずつサンプリングするため、偏りが少ない。

参考: [02_ml_prerequisites.md](../02_ml_prerequisites.md)（サンプリング手法の説明）

### FRF 計算

```python
y_sim = simulator(x_sim)   # shape: (100000, 4, 1, 1024)
```

`Simulator.__call__` が `x_sim` を `[LLIM, ULIM]` にスケール変換してから並列で `fun` を呼ぶ。

計算結果の形状：
- `100000`: サンプル数
- `4`: 4自由度（各階）
- `1`: チャンネル次元（畳み込み用）
- `1024`: 周波数点数（0.02 〜 20.48 Hz）

### ノイズ付加

```python
noise_level = 0.20
y_sim_n = y_sim + noise_level * norm.rvs(size=y_sim.shape)
```

標準正規分布ノイズを振幅 0.20 でスケールして加算。

**なぜノイズを付加するのか**:
- 実測データには常にセンサーノイズが含まれる
- ノイズ付きデータを学習に使うことで、MVAE が実測時にも機能するようになる
- `train.py` では `y_sim`（クリーン）を `enc_x` の入力に、`y_sim_n`（ノイズあり）をデコーダの目標に使う

詳細は [04_mvae.md](../04_mvae.md) を参照。

### 保存

```python
np.savez('train_data.npz',
         llim=LLIM, ulim=ULIM,
         x_sim=x_sim,        # shape: (100000, 4)
         y_sim=y_sim,         # shape: (100000, 4, 1, 1024)
         y_sim_n=y_sim_n)    # shape: (100000, 4, 1, 1024)
```

`train.py` と `inference.py` がこのファイルを読み込む。

---

## データ形状サマリー

| 変数 | 形状 | 値域 | 説明 |
|------|------|------|------|
| `x_sim` | `(100000, 4)` | `[0, 1]` | 正規化パラメータ |
| `y_sim` | `(100000, 4, 1, 1024)` | 実数 | ノイズなし FRF（対数絶対値） |
| `y_sim_n` | `(100000, 4, 1, 1024)` | 実数 | ノイズあり FRF |

---

## 実行時間の目安

100,000 サンプルの FRF 計算は数分かかる（CPUコア数に依存）。  
Docker + GPU 環境では `make build && make up` で GPU を使った高速化が可能（FRF 計算自体は CPU だが後続の MVAE 学習が GPU 加速される）。
