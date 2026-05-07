# train.py — MVAE の学習スクリプト

**ファイルパス**: `src/lsbi_smc/example_shear4dof/train.py`  
**関連学術ドキュメント**: [05_mvae.md](../05_mvae.md), [03_ml_prerequisites.md](../03_ml_prerequisites.md)

---

## 概要

`train_data.npz` から学習データを読み込み、MVAE を最大 1000 エポック学習するスクリプトです。早期終了（patience=20）で過学習を防ぎ、最良のモデルを `mvae_best.pth` に保存します。

**実行方法**（プロジェクトルートから）：
```bash
uv run python src/lsbi_smc/example_shear4dof/train.py
```

**前提**: `train_data.npz` が存在すること（`create_dataset.py` で生成）  
**出力**: `mvae_best.pth`（プロジェクトルート）

---

## 全体の処理フロー

```
1. データ読み込み (train_data.npz)
      ↓
2. チャンネル選択 (屋根のみ ch=[-1])
      ↓
3. 標準化 (FRFデータ)
      ↓
4. 学習/検証分割 (9:1)
      ↓
5. DataLoader 作成
      ↓
6. MVAE モデル初期化
      ↓
7. 学習ループ (最大1000エポック)
   ├── 訓練フェーズ (全バッチ処理)
   ├── 検証フェーズ (損失評価)
   ├── 早期終了判定 (patience=20)
   └── 最良モデル保存
```

---

## コードの詳細

### データ読み込みと前処理

```python
npl = np.load('train_data.npz')
x_sim   = npl['x_sim'].astype(np.float32)     # (100000, 4)
y_sim   = npl['y_sim'].astype(np.float32)     # (100000, 4, 1, 1024)
y_sim_n = npl['y_sim_n'].astype(np.float32)  # (100000, 4, 1, 1024)
del npl  # メモリ節約
```

### チャンネル選択（屋根のみ）

```python
ch = [-1]
y_sim_tensor   = torch.from_numpy(y_sim[:, ch, :, :])    # (100000, 1, 1, 1024)
y_sim_n_tensor = torch.from_numpy(y_sim_n[:, ch, :, :])  # (100000, 1, 1, 1024)
```

インデックス `-1` で最上階（屋根）を選択。論文のベンチマーク設定では屋根のみが観測可能という想定。

**なぜ屋根だけか**: 実構造物では屋根にセンサーを設置するケースが多い。全階のデータを使うより、現実的な設定でアルゴリズムの有効性を示すため。

### データの標準化

```python
y_mn, y_sd = y_sim_tensor.mean(), y_sim_tensor.std()
y_sim_tensor   = (y_sim_tensor   - y_mn) / y_sd
y_sim_n_tensor = (y_sim_n_tensor - y_mn) / y_sd
```

**重要**: `y_mn` と `y_sd` の値は `inference.py` でハードコードされている。

```python
# inference.py:28
y_mn, y_sd = -2.1384575366973877, 2.809697389602661
```

これは `train.py` を実行して得られた統計量の値。学習データのスケールで観測データを正規化することで、学習時と推論時のスケールを一致させる。

### データセット・DataLoader の構築

```python
dataset = TensorDataset(x_sim_tensor, y_sim_tensor, y_sim_n_tensor)
# ← 各アイテムが (θ, y_clean, y_noisy) のタプル

n_train = int(0.9 * n_total)    # 90,000 件
n_valid = n_total - n_train     # 10,000 件

train_dataset, valid_dataset = random_split(
    dataset, [n_train, n_valid],
    generator=torch.Generator().manual_seed(42)  # 再現性のため固定シード
)

train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True,
                          num_workers=8, pin_memory=True)
valid_loader = DataLoader(valid_dataset, batch_size=512, shuffle=False,
                          num_workers=4, pin_memory=True)
```

- `batch_size=512`: 1回の勾配更新で512サンプルを処理
- `pin_memory=True`: CPU→GPU 転送を高速化（CUDA 使用時）
- `num_workers=8`: データ読み込みを並列化

### MVAE モデルの初期化

```python
ndof, z_dim = 4, 8
model = MVAE(z_dim=8, ch=1, size=1024, nlabel=4, depth=1).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
```

| ハイパーパラメータ | 値 | 意味 |
|-----------------|-----|------|
| `z_dim` | 8 | 潜在空間の次元数 |
| `ch` | 1 | 入力チャンネル数（屋根のみ） |
| `size` | 1024 | FRF の周波数点数 |
| `nlabel` | 4 | 構造パラメータの次元数 |
| `lr` | 1e-3 | Adam の学習率 |

### 学習ループ

```python
for epoch in range(epochs):
    # ---- 訓練フェーズ ----
    model.train()
    for x, y, yn in train_loader:
        x, y, yn = x.to(device), y.to(device), yn.to(device)
        optimizer.zero_grad()
        trl, kll, rcl = model.loss(yn, y, x, alp1=5.0)
        #                        ↑     ↑  ↑
        #              xo=ノイズあり  xi=クリーン  w=パラメータ
        trl.backward()
        optimizer.step()
```

**引数の対応**:
- `xo=yn`（ノイズあり FRF）→ デコーダの再構成目標
- `xi=y`（クリーン FRF）→ `enc_x` の入力
- `w=x`（構造パラメータ）→ `enc_w` の入力

`alp1=5.0`: 潜在空間の整合性を強調（論文の設定）。2つのエンコーダの潜在表現を強制的に揃える。

### 検証と早期終了

```python
model.eval()
with torch.no_grad():
    for x, y, yn in valid_loader:
        vll, _, _ = model.loss(yn, y, x, alp1=5.0)
        vl_loss += vll.item() * y.size(0)

if vl_loss < best_vl_loss:
    best_vl_loss = vl_loss
    epochs_no_improve = 0
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, 'mvae_best.pth')
else:
    epochs_no_improve += 1

if epochs_no_improve >= patience:  # patience=20
    break
```

検証損失が20エポック改善しなければ学習を打ち切る。最良の検証損失を記録したモデルのみ保存。

### 保存形式

`mvae_best.pth` には以下が含まれる：

| キー | 内容 |
|------|------|
| `epoch` | 最良エポック番号 |
| `model_state_dict` | モデルの重みパラメータ |
| `optimizer_state_dict` | オプティマイザの状態（再学習用） |

`inference.py` での読み込み：

```python
model.load_state_dict(
    torch.load('mvae_best.pth', map_location=device)['model_state_dict']
)
```

---

## 損失の構成と意味

```
Total Loss = kl1 + kl2 + 5.0*(kl_x1x2 + kl_x2x1) + 10.0*rec_xx + 10.0*rec_wx
```

| 損失項 | 係数 | 役割 |
|--------|------|------|
| `kl1` | 1.0 | `enc_x` の潜在空間を標準正規に近づける |
| `kl2` | 1.0 | `enc_w` の潜在空間を標準正規に近づける |
| `kl_x1x2 + kl_x2x1` | 5.0 | 2つのエンコーダの潜在表現を一致させる（**最重要**） |
| `rec_xx` | 10.0 | FRF → 潜在 → FRF の再構成精度 |
| `rec_wx` | 10.0 | θ → 潜在 → FRF の予測精度 |

`alp1=5.0` の高い係数が重要: これにより `enc_x(FRF)` と `enc_w(θ)` の潜在表現が同じ場所にマッピングされる。推論時に `enc_w(θ)` と観測の `enc_x(FRF_obs)` が比較可能になる。

---

## エポック進行の出力例

```
Epoch 000: Train loss =         1234.5678 | Valid loss =         1256.3421
Epoch 001: Train loss =          876.2345 | Valid loss =          891.1234
...
Epoch 045: Train loss =          123.4567 | Valid loss =          125.6789
```

---

## 注意点

1. **標準化定数の記録**: 学習後、`y_mn` と `y_sd` の値を `inference.py` にコピーする必要がある（現在はハードコード）
2. **GPU推奨**: 10万サンプルの学習はCPUでも可能だが、GPUで10〜20倍高速
3. **再現性**: `random_split` のシード固定（seed=42）で訓練/検証分割が再現可能
