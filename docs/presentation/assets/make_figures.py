"""Generate presentation figures for slides 3, 4, 5.

Run from project root:
    uv run python docs/presentation/assets/make_figures.py
"""

from pathlib import Path

import japanize_matplotlib  # noqa: F401  -- registers IPAex Gothic for plt
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ASSETS = Path(__file__).parent

plt.rcParams.update(
    {
        "font.size": 14,
        "axes.linewidth": 1.2,
        "savefig.dpi": 180,
        "savefig.bbox": "tight",
    }
)


# ---------- common helpers ----------


def _box(ax, xy, w, h, text, fc="#eef3f8", ec="#1a3a5c", fontsize=13, fontweight="normal"):
    box = FancyBboxPatch(
        (xy[0] - w / 2, xy[1] - h / 2),
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=0.08",
        linewidth=1.5,
        facecolor=fc,
        edgecolor=ec,
    )
    ax.add_patch(box)
    ax.text(
        xy[0],
        xy[1],
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        fontweight=fontweight,
        color="#1a3a5c",
    )


def _arrow(ax, p0, p1, color="#1a3a5c", lw=1.6, style="->"):
    arr = FancyArrowPatch(
        p0,
        p1,
        arrowstyle=style,
        mutation_scale=18,
        linewidth=lw,
        color=color,
        shrinkA=4,
        shrinkB=4,
    )
    ax.add_patch(arr)


# ---------- figure 1: pipeline (Slide 3) ----------


def make_pipeline():
    fig, ax = plt.subplots(figsize=(13, 6))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 6.5)
    ax.axis("off")

    # backgrounds
    offline_bg = FancyBboxPatch(
        (0.2, 3.3), 12.6, 2.8,
        boxstyle="round,pad=0.02,rounding_size=0.1",
        facecolor="#f5f9fc",
        edgecolor="#7a9bbe",
        linewidth=1.5,
        linestyle="--",
    )
    online_bg = FancyBboxPatch(
        (0.2, 0.2), 12.6, 2.5,
        boxstyle="round,pad=0.02,rounding_size=0.1",
        facecolor="#fbf3f0",
        edgecolor="#c0392b",
        linewidth=1.5,
        linestyle="--",
    )
    ax.add_patch(offline_bg)
    ax.add_patch(online_bg)
    ax.text(0.4, 5.85, "オフライン (一度だけ)", fontsize=13,
            color="#1a3a5c", fontweight="bold")
    ax.text(0.4, 2.45, "オンライン (推論)", fontsize=13,
            color="#c0392b", fontweight="bold")

    # offline boxes (single row, left-to-right)
    _box(ax, (1.5, 4.5), 1.8, 0.9, "事前 $p(\\theta)$")
    _box(ax, (3.9, 4.5), 1.8, 0.9, "FE 解析")
    _box(ax, (6.3, 4.5), 1.8, 0.9, "$(\\theta, x)$ ペア")
    _box(ax, (8.7, 4.5), 1.8, 0.9, "MVAE 学習", fc="#dfeaf4", fontweight="bold")
    _box(ax, (11.5, 4.5), 2.2, 1.1,
         "近似尤度\n$\\hat{L}(\\theta;\\, x)$",
         fc="#fff7d6", ec="#b9871a", fontsize=13, fontweight="bold")

    # offline arrows
    _arrow(ax, (2.4, 4.5), (3.0, 4.5))
    _arrow(ax, (4.8, 4.5), (5.4, 4.5))
    _arrow(ax, (7.2, 4.5), (7.8, 4.5))
    _arrow(ax, (9.6, 4.5), (10.4, 4.5))

    # ML annotation
    ax.text(8.7, 5.4, "↑ 機械学習はここ (尤度を作る)",
            ha="center", fontsize=12, color="#b9871a", fontweight="bold")

    # online boxes
    _box(ax, (1.5, 1.4), 1.8, 0.9, "観測 $x_{\\mathrm{obs}}$",
         fc="#fbe6e1", ec="#c0392b")
    _box(ax, (6.3, 1.4), 2.0, 0.9, "SMC サンプラー",
         fc="#fbe6e1", ec="#c0392b", fontweight="bold")
    _box(ax, (11.5, 1.4), 2.2, 0.9,
         "事後 $p(\\theta\\mid x_{\\mathrm{obs}})$",
         fc="#fbe6e1", ec="#c0392b", fontweight="bold")

    # online arrows
    _arrow(ax, (2.4, 1.4), (5.3, 1.4), color="#c0392b")
    _arrow(ax, (7.3, 1.4), (10.4, 1.4), color="#c0392b")
    # Lhat -> SMC (diagonal)
    _arrow(ax, (11.0, 3.9), (6.9, 1.9), color="#b9871a", style="->", lw=2.0)
    ax.text(9.4, 2.95, "$\\hat{L}$ のみを呼ぶ\n(FE 解析は不要)",
            ha="center", va="center", fontsize=11,
            color="#b9871a", fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", fc="white",
                      ec="#b9871a", lw=1))

    fig.savefig(ASSETS / "pipeline.png")
    plt.close(fig)


# ---------- figure 2: VAE vs MVAE (Slide 4) ----------


def make_vae_mvae_compare():
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.0))

    for ax, title in zip(axes, ["通常 VAE", "MVAE (本手法)"]):
        ax.set_xlim(0, 12)
        ax.set_ylim(0, 6)
        ax.axis("off")
        ax.text(6, 5.6, title, ha="center", fontsize=16,
                color="#1a3a5c", fontweight="bold")

    # ---- left: VAE ----
    ax = axes[0]
    _box(ax, (1.1, 3.0), 1.2, 1.0, "$x$", fc="#fbe6e1", ec="#c0392b")
    _box(ax, (3.4, 3.0), 1.9, 1.0, "Encoder\n$q_\\phi(z|x)$")
    _box(ax, (5.7, 3.0), 1.2, 1.0, "$z$", fc="#fff7d6", ec="#b9871a")
    _box(ax, (8.0, 3.0), 1.9, 1.0, "Decoder\n$p_\\eta(x|z)$")
    _box(ax, (10.5, 3.0), 1.3, 1.0, "$\\hat{x}$\n(復元)",
         fc="#fbe6e1", ec="#c0392b", fontsize=12)
    _arrow(ax, (1.7, 3.0), (2.45, 3.0))
    _arrow(ax, (4.35, 3.0), (5.1, 3.0))
    _arrow(ax, (6.3, 3.0), (7.05, 3.0))
    _arrow(ax, (8.95, 3.0), (9.85, 3.0))

    ax.text(6, 1.5,
            "損失 = 復元誤差($x,\\hat{x}$) $+\\; D_{KL}(q_\\phi \\| p(z))$",
            ha="center", fontsize=13)
    ax.text(6, 0.6, "入力: $x$ のみ ／ エンコーダ: 1個",
            ha="center", fontsize=12, color="#555")

    # ---- right: MVAE ----
    ax = axes[1]
    _box(ax, (1.0, 4.0), 1.1, 0.9, "$x$", fc="#fbe6e1", ec="#c0392b")
    _box(ax, (1.0, 2.0), 1.1, 0.9, "$\\theta$", fc="#e1f0e7", ec="#2c8a4f")
    _box(ax, (3.2, 4.0), 1.9, 0.9, "Encoder $q_{\\phi_x}$",
         fc="#fbe6e1", ec="#c0392b", fontsize=12)
    _box(ax, (3.2, 2.0), 1.9, 0.9, "Encoder $q_{\\phi_\\theta}$",
         fc="#e1f0e7", ec="#2c8a4f", fontsize=12)
    _box(ax, (6.0, 3.0), 1.3, 1.0, "共有 $z$", fc="#fff7d6",
         ec="#b9871a", fontweight="bold")
    _box(ax, (8.3, 3.0), 1.9, 1.0, "Decoder\n$p_\\eta(x|z)$")
    _box(ax, (10.8, 3.0), 1.3, 1.0, "$\\hat{x}$\n(復元)",
         fc="#fbe6e1", ec="#c0392b", fontsize=12)

    _arrow(ax, (1.55, 4.0), (2.25, 4.0))
    _arrow(ax, (1.55, 2.0), (2.25, 2.0))
    _arrow(ax, (4.15, 4.0), (5.4, 3.3))
    _arrow(ax, (4.15, 2.0), (5.4, 2.7))
    _arrow(ax, (6.65, 3.0), (7.4, 3.0))
    _arrow(ax, (9.25, 3.0), (10.15, 3.0))

    # KL bridge between encoders
    ax.annotate("", xy=(3.2, 3.45), xytext=(3.2, 2.55),
                arrowprops=dict(arrowstyle="<->", color="#b9871a", lw=2))
    ax.text(3.7, 3.0, "$D_{KL}$\nで揃える",
            ha="left", va="center", fontsize=11,
            color="#b9871a", fontweight="bold")

    ax.text(6, 1.0,
            "損失 = 復元 + 復元(クロス) $+\\; \\alpha\\, D_{KL}(q_{\\phi_\\theta}\\|q_{\\phi_x})$",
            ha="center", fontsize=12)
    ax.text(6, 0.3, "入力: $(\\theta, x)$ ペア ／ エンコーダ: 2個",
            ha="center", fontsize=12, color="#555")

    fig.savefig(ASSETS / "vae_mvae_compare.png")
    plt.close(fig)


# ---------- figure 3: latent overlap (Slide 5) ----------


def _gauss2d_contour(ax, mu, cov, color, label, fill_alpha=0.20, line_alpha=0.95):
    x = np.linspace(-3, 3, 200)
    y = np.linspace(-3, 3, 200)
    X, Y = np.meshgrid(x, y)
    pos = np.dstack([X, Y])
    inv = np.linalg.inv(cov)
    diff = pos - mu
    expo = -0.5 * np.einsum("...i,ij,...j->...", diff, inv, diff)
    norm = 1.0 / (2 * np.pi * np.sqrt(np.linalg.det(cov)))
    Z = norm * np.exp(expo)
    zmax = Z.max()
    levels = np.linspace(0.15, 0.95, 4) * zmax
    ax.contourf(X, Y, Z, levels=levels, colors=[color], alpha=fill_alpha)
    ax.contour(X, Y, Z, levels=levels, colors=color, linewidths=1.2, alpha=line_alpha)
    ax.scatter([mu[0]], [mu[1]], color=color, s=50, zorder=5,
               edgecolor="white", linewidth=1.2)
    ax.annotate(label, xy=(mu[0], mu[1]), xytext=(mu[0] + 0.4, mu[1] + 0.4),
                color=color, fontsize=12, fontweight="bold")


def make_latent_overlap():
    fig, ax = plt.subplots(figsize=(7, 6.5))

    mu_obs = np.array([1.0, 0.7])
    cov_obs = np.array([[0.35, 0.08], [0.08, 0.45]])
    mu_th = np.array([0.5, 0.2])
    cov_th = np.array([[0.45, -0.04], [-0.04, 0.30]])

    ax.set_title(
        "潜在空間でガウスが重なるほど\n$\\theta$ は観測と整合 → $\\hat{L}(\\theta;\\, x_{\\mathrm{obs}})$ が大",
        fontsize=14, color="#1a3a5c"
    )
    _gauss2d_contour(ax, mu_obs, cov_obs, "#c0392b", "観測 $q_{\\phi_x}$")
    _gauss2d_contour(ax, mu_th, cov_th, "#2c8a4f", "$\\theta$ 候補 $q_{\\phi_\\theta}$")

    # prior contour (dashed, gray)
    x = np.linspace(-3, 3, 200)
    y = np.linspace(-3, 3, 200)
    X, Y = np.meshgrid(x, y)
    R2 = X**2 + Y**2
    ax.contour(X, Y, R2, levels=[1, 4], colors="#888", linewidths=1.2,
               linestyles="dashed", alpha=0.8)
    ax.text(-2.4, 2.15, "$p(z)\\!=\\!\\mathcal{N}(0,I)$\n(破線, 潜在事前)",
            fontsize=11, color="#666")

    ax.set_xlim(-2.5, 2.5)
    ax.set_ylim(-2.5, 2.5)
    ax.set_aspect("equal")
    ax.set_xlabel("$z_1$", fontsize=13)
    ax.set_ylabel("$z_2$", fontsize=13)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(ASSETS / "latent_overlap.png")
    plt.close(fig)


if __name__ == "__main__":
    make_pipeline()
    make_vae_mvae_compare()
    make_latent_overlap()
    print("Generated:")
    for p in sorted(ASSETS.glob("*.png")):
        print(f"  - {p.relative_to(ASSETS.parent.parent.parent)}")
