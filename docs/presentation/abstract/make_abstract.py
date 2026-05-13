"""Generate abstract.docx from the template `sample.docx`.

Run from project root:
    uv run --group presentation python docs/presentation/abstract/make_abstract.py

Strategy:
- Open the sample as a template (keeps page setup, columns, styles).
- Replace text in the title block paragraphs (paragraphs 0-4) and strip
  instruction textboxes.
- Remove existing body paragraphs after the title-block section break,
  then insert new body paragraphs using the template's registered styles.
"""

import shutil
from pathlib import Path

from docx import Document
from docx.oxml import OxmlElement
from docx.oxml.ns import qn

HERE = Path(__file__).parent
SOURCE = HERE / "sample.docx"
OUTPUT = HERE / "abstract.docx"

W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"


# ---- Content ----

TITLE_JA = "工学逆問題のための潜在空間ベイズ推論フレームワーク構築"
TITLE_EN = (
    "A Latent-Space Bayesian Inference Framework "
    "for Engineering Inverse Problems"
)
ADVISOR = "指導教員　[教員氏名 教授/准教授/講師]"
AUTHOR = "03-XXXXXX　[学生氏名]"

# Body sections: (style_name, text). Empty text = blank spacer paragraph.
BODY: list[tuple[str, str]] = [
    ("Heading 1", "１．はじめに"),
    (
        "Body Text Indent",
        "工学では、観測データからモデルパラメータを推定する逆問題が広く現れる"
        "（構造物の剛性同定、材料定数推定など）。ベイズ的アプローチは、"
        "点推定だけでなく事後分布を通じて推定の不確実性を定量化できる利点を持つ。"
        "一方で、(i) 観測が高次元な応答時系列となる、(ii) 評価のたびに"
        "高コストなシミュレータ呼び出しが必要、(iii) 観測の制約により"
        "事後分布が多峰となる、という3つの障害がある。"
        "本研究はこれらを回避する潜在空間ベイズ推論をフレームワーク "
        "として整備した。"
    ),
    ("Normal", ""),
    ("Heading 1", "２．提案フレームワーク"),
    (
        "Body Text Indent",
        "ベイズの定理 (1) より、尤度 L が評価できれば事後分布 p(θ|x_obs) は"
        "推定可能である。"
    ),
    ("数式", "p(θ|x_obs) ∝ L(θ; x_obs) p(θ)\t(1)"),
    (
        "Body Text Indent",
        "提案手法はオフラインとオンラインの 2 段構成からなる。"
        "オフラインでは事前分布から (θ, x) サンプルを生成し、"
        "Multimodal Variational Autoencoder (MVAE) を学習して、"
        "低次元の潜在空間における近似尤度 L̂ を構築する。"
        "これにより、推論時の尤度評価から重いシミュレータ呼び出しを排除する。"
        "オンラインでは Sequential Monte Carlo (SMC) サンプラーが "
        "L̂ のみを用いて事後分布から粒子をサンプリングする。"
        "SMC は焼きなまし型の段階更新により多峰性に頑健であり、"
        "粒子間が独立であるため GPU 並列化が容易である。"
    ),
    (
        "Body Text Indent",
        "実装は Python パッケージとして整備し、サンプラー・MCMC カーネル・"
        "提案分布・尤度モデル・事前分布の各構成要素を Protocol によって"
        "抽象化した。これにより、今後の比較研究で各要素を差し替えながら"
        "検証することが可能となる。"
    ),
    ("Normal", ""),
    ("Heading 1", "３．検証例"),
    (
        "Body Text Indent",
        "文献 1) に倣い、4 自由度せん断建物（屋上 FRF のみ観測）の"
        "層剛性推定で動作確認を行った。真値 θ = (1, 1, 1, 1) に対して、"
        "事後分布のメインモードは真値近傍に集中し、観測上の同定不能性に由来する"
        "等価解も別モードとして再現された。実行時間は SMC で約 0.8 秒、"
        "NUTS で 1782 秒となり、GPU 並列化により約 2200 倍の高速化が"
        "確認された 1)。"
    ),
    ("Normal", ""),
    ("Heading 1", "４．まとめ"),
    (
        "Body Text Indent",
        "工学一般の逆問題を対象とした、不確実性定量化付きの潜在空間ベイズ推論"
        "フレームワークを構築した。今後は代替尤度モデルやサンプラーの実装と"
        "比較、より複雑なシステムへの適用拡張を進める。"
    ),
    ("Normal", ""),
    ("Heading 1", "参考文献"),
    (
        "文献リスト",
        "Yaoyama, T. et al., Finite element model updating of building "
        "structures under seismic excitation: A parallelized latent "
        "space-based Bayesian framework, Nucl. Eng. Des. (2026).",
    ),
    (
        "文献リスト",
        "Itoi, T. et al., Bayesian structural model updating with multimodal "
        "variational autoencoder, Comput. Methods Appl. Mech. Eng., 429, "
        "117148 (2024).",
    ),
    (
        "文献リスト",
        "Ching, J., Chen, Y.-C., Transitional Markov chain Monte Carlo method "
        "for Bayesian model updating, model class selection, and model "
        "averaging, J. Eng. Mech., 133(7), 816-832 (2007).",
    ),
]


# ---- Helpers ----


def make_run(text: str) -> OxmlElement:
    r = OxmlElement("w:r")
    # Preserve the leading/trailing whitespace and tab characters
    parts = text.split("\t")
    for i, part in enumerate(parts):
        if i > 0:
            r.append(OxmlElement("w:tab"))
        if part:
            t = OxmlElement("w:t")
            t.set(qn("xml:space"), "preserve")
            t.text = part
            r.append(t)
    return r


def replace_text_in_titleblock(p, new_text: str) -> None:
    """Replace text inside a title-block paragraph, stripping textboxes."""
    pPr = p.find(W + "pPr")
    # Remove all children
    for child in list(p):
        p.remove(child)
    if pPr is not None:
        p.append(pPr)
    p.append(make_run(new_text))


def get_style_id(doc, name: str) -> str:
    """Map style display name -> internal style_id."""
    return doc.styles[name].style_id


def make_paragraph(doc, style_name: str, text: str) -> OxmlElement:
    p = OxmlElement("w:p")
    pPr = OxmlElement("w:pPr")
    pStyle = OxmlElement("w:pStyle")
    pStyle.set(qn("w:val"), get_style_id(doc, style_name))
    pPr.append(pStyle)
    p.append(pPr)
    if text:
        p.append(make_run(text))
    return p


# ---- Main ----


def main() -> None:
    shutil.copy(SOURCE, OUTPUT)
    doc = Document(str(OUTPUT))
    body = doc.element.body

    # Title block: paragraphs 0-4 in the sample.
    #   0: Japanese title (style 題目, style_id=a4)
    #   1: English title  (style タイトル, ac)
    #   2: blank
    #   3: 教官名 (ab)
    #   4: 著者名 (ad)
    paragraphs = [p for p in body if p.tag.endswith("}p")]
    replace_text_in_titleblock(paragraphs[0], TITLE_JA)
    replace_text_in_titleblock(paragraphs[1], TITLE_EN)
    replace_text_in_titleblock(paragraphs[3], ADVISOR)
    replace_text_in_titleblock(paragraphs[4], AUTHOR)

    # Paragraph 6 holds the section break to switch to 2 columns; keep it.
    section_break_p = paragraphs[6]

    # Remove all existing body paragraphs after the section break,
    # but keep the final <w:sectPr> at body level.
    for p in paragraphs[7:]:
        body.remove(p)

    # Insert new body paragraphs after the section break.
    anchor = section_break_p
    for style_name, text in BODY:
        new_p = make_paragraph(doc, style_name, text)
        anchor.addnext(new_p)
        anchor = new_p

    doc.save(str(OUTPUT))
    print(f"Wrote {OUTPUT.relative_to(Path.cwd())}")


if __name__ == "__main__":
    main()
