# プレゼン資料

5分プレゼン（構造工学系・ML弱め向け）の発表資料一式。

## ファイル

| ファイル | 内容 |
|---|---|
| [01_outline.md](01_outline.md) | 構成案・配分・図表チェックリスト |
| [02_slides.md](02_slides.md) | Marp 形式のスライドソース |
| [03_script.md](03_script.md) | 発表スクリプト（読み上げ用） |
| [04_qa.md](04_qa.md) | 想定問答集 |
| [assets/make_figures.py](assets/make_figures.py) | スライド用図を matplotlib で生成 |
| `assets/*.png` | 生成物（gitignore）、`make figures` で再生成 |

## 図の生成

スライドに使う pipeline / VAE比較 / 潜在尤度の3図は Python で生成する:

```bash
make figures        # = uv run python docs/presentation/assets/make_figures.py
```

`japanize-matplotlib` が必要（pyproject.toml に追加済み）。

## スライドのビルド

Marp CLI が必要です。

```bash
# Node 環境がある場合（推奨）
npx @marp-team/marp-cli@latest 02_slides.md --pdf
npx @marp-team/marp-cli@latest 02_slides.md --pptx
npx @marp-team/marp-cli@latest 02_slides.md --html

# Docker を使う場合
docker run --rm -v $PWD:/home/marp/app marpteam/marp-cli 02_slides.md --pdf
```

成果物（`slides.pdf`, `slides.pptx`, `slides.html`）は **コミット対象外**。
ソースの `02_slides.md` のみコミットする。

### Google Slides に持っていく場合

1. `--pptx` で PPTX を書き出し
2. Google Drive にアップロード
3. 右クリック → 「Google スライドで開く」で変換
4. 数式は **画像化される**ので、変換後に式の見え方を必ず確認する

## 図表の作成方針

`assets/` 配下にプレゼン用に作り直した図を配置する（論文用の高密度図はそのまま使わない）。

| ファイル名（予定）       | 用途                                | 作成元                                            |
| ------------------------ | ----------------------------------- | ------------------------------------------------- |
| `building_fem.png`       | Slide 1: 建物 + FEM 模式図 + 観測   | draw.io / 既存写真                                |
| `vae_mvae_compare.png`   | Slide 4: VAE vs MVAE 構造比較       | draw.io（左右2枚並べる）                          |
| `particles_parallel.png` | Slide 6: 並列粒子のイメージ + 焼きなまし | matplotlib 3パネル散布図 + 並列の概念図          |
| `shear4dof_frf.png`      | Slide 7: 4DOFモデル + 観測 FRF 例   | draw.io（質点ばねダンパ） + matplotlib            |

スライドでは図のないスライドもあります:
- Slide 3（全体像）: ASCII アートのパイプライン図で代用（必要なら `pipeline.png` を追加）
- Slide 5（潜在尤度）: 式変形のみ（必要なら `latent_compare.png` を追加）
- Slide 9（取り組んだこと）: 表のみ（必要なら `module_blocks.png` を追加）

`../../posterior_plot.png`（既存）は Slide 8 でそのまま参照する。
プレゼン用に凡例・フォントの整形が必要ならコピーを `assets/` に置く。

### 図の品質方針

- 線は太め、フォントサイズ大（最低 18pt 相当）
- 文字は最小限、矢印・色分けで意味を伝える
- 余白を確保
- 色は colorblind-safe（matplotlib `tab10` 系 or Okabe-Ito）

## ローカル確認（Marp の VSCode 拡張）

VSCode で `Marp for VS Code` 拡張を入れると、`02_slides.md` のプレビューがエディタ右側で見られる。
スライド執筆時は VSCode + Marp 拡張、配布時は CLI で書き出す、の運用が楽。

## .marprc.yml について

このディレクトリの `.marprc.yml` は、**`<div style="...">` などの inline style 属性を許可**する設定。
Marp はセキュリティのためデフォルトで `style`/`class` 属性をサニタイズで剥がしてしまい、
スライド内の 2 段組（flex レイアウト）が機能しなくなる。`.marprc.yml` で whitelist することで
flex を含むレイアウトが正しく描画される。

VSCode の Marp 拡張も同ディレクトリの `.marprc.yml` を自動で読むので、特別な設定は不要。
