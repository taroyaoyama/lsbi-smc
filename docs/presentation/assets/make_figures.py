"""Generate presentation figures (Japanese and English variants).

Run from project root:
    uv run python docs/presentation/assets/make_figures.py
"""

from dataclasses import dataclass
from pathlib import Path

import japanize_matplotlib  # noqa: F401  -- registers IPAex Gothic for plt
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle

ASSETS = Path(__file__).parent

plt.rcParams.update(
    {
        "font.size": 14,
        "axes.linewidth": 1.2,
        "savefig.dpi": 180,
        "savefig.bbox": "tight",
    }
)


# ---------- localized strings ----------


@dataclass
class L10n:
    suffix: str
    pipeline_offline: str
    pipeline_online: str
    pipeline_prior: str
    pipeline_fe: str
    pipeline_pair: str
    pipeline_train: str
    pipeline_lhat: str
    pipeline_ml_note: str
    pipeline_obs: str
    pipeline_smc: str
    pipeline_posterior: str
    pipeline_lhat_only: str
    vae_title: str
    vae_recon: str
    vae_loss: str
    vae_meta: str
    mvae_title: str
    mvae_shared_z: str
    mvae_kl_align: str
    mvae_loss: str
    mvae_meta: str
    latent_title: str
    latent_obs: str
    latent_theta: str
    latent_prior: str
    mvae_io_enc: str
    mvae_io_dec: str
    mvae_io_input_x: str
    mvae_io_input_th: str
    mvae_io_latent: str


JA = L10n(
    suffix="",
    pipeline_offline="オフライン (一度だけ)",
    pipeline_online="オンライン (推論)",
    pipeline_prior="事前 $p(\\theta)$",
    pipeline_fe="FE 解析",
    pipeline_pair="$(\\theta, x)$ ペア",
    pipeline_train="MVAE 学習",
    pipeline_lhat="近似尤度\n$\\hat{L}(\\theta;\\, x)$",
    pipeline_ml_note="↑ 機械学習はここ (尤度を作る)",
    pipeline_obs="観測 $x_{\\mathrm{obs}}$",
    pipeline_smc="SMC サンプラー",
    pipeline_posterior="事後 $p(\\theta\\mid x_{\\mathrm{obs}})$",
    pipeline_lhat_only="$\\hat{L}$ のみを呼ぶ\n(FE 解析は不要)",
    vae_title="通常 VAE",
    vae_recon="$\\hat{x}$\n(復元)",
    vae_loss="損失 = 復元誤差($x,\\hat{x}$) $+\\; D_{KL}(q_\\phi \\| p(z))$",
    vae_meta="入力: $x$ のみ ／ エンコーダ: 1個",
    mvae_title="MVAE (本手法)",
    mvae_shared_z="共有 $z$",
    mvae_kl_align="$D_{KL}$\nで揃える",
    mvae_loss="損失 = 復元 + 復元(クロス) $+\\; \\alpha\\, D_{KL}(q_{\\phi_\\theta}\\|q_{\\phi_x})$",
    mvae_meta="入力: $(\\theta, x)$ ペア ／ エンコーダ: 2個",
    latent_title=(
        "潜在空間でガウスが重なるほど\n"
        "$\\theta$ は観測と整合 → $\\hat{L}(\\theta;\\, x_{\\mathrm{obs}})$ が大"
    ),
    latent_obs="観測 $q_{\\phi_x}$",
    latent_theta="$\\theta$ 候補 $q_{\\phi_\\theta}$",
    latent_prior="$p(z)\\!=\\!\\mathcal{N}(0,I)$\n(破線, 潜在事前)",
    mvae_io_enc="Encoder",
    mvae_io_dec="Decoder",
    mvae_io_input_x="FRF",
    mvae_io_input_th="パラメータ",
    mvae_io_latent="潜在空間",
)

EN = L10n(
    suffix="_en",
    pipeline_offline="Offline (once)",
    pipeline_online="Online (inference)",
    pipeline_prior="Prior $p(\\theta)$",
    pipeline_fe="FE analysis",
    pipeline_pair="$(\\theta, x)$ pairs",
    pipeline_train="MVAE training",
    pipeline_lhat="Approx. likelihood\n$\\hat{L}(\\theta;\\, x)$",
    pipeline_ml_note="↑ ML lives here (builds the likelihood)",
    pipeline_obs="Observation $x_{\\mathrm{obs}}$",
    pipeline_smc="SMC sampler",
    pipeline_posterior="Posterior $p(\\theta\\mid x_{\\mathrm{obs}})$",
    pipeline_lhat_only="Call $\\hat{L}$ only\n(no FE call)",
    vae_title="Standard VAE",
    vae_recon="$\\hat{x}$\n(recon.)",
    vae_loss="Loss = recon.$(x,\\hat{x})$ $+\\; D_{KL}(q_\\phi \\| p(z))$",
    vae_meta="Input: $x$ only / 1 encoder",
    mvae_title="MVAE (this work)",
    mvae_shared_z="Shared $z$",
    mvae_kl_align="Align\nvia $D_{KL}$",
    mvae_loss="Loss = recon. + recon.(cross) $+\\; \\alpha\\, D_{KL}(q_{\\phi_\\theta}\\|q_{\\phi_x})$",
    mvae_meta="Input: $(\\theta, x)$ pair / 2 encoders",
    latent_title=(
        "Larger Gaussian overlap in latent space\n"
        "⇒ $\\theta$ more consistent with $x_{\\mathrm{obs}}$ ⇒ larger $\\hat{L}$"
    ),
    latent_obs="Obs. $q_{\\phi_x}$",
    latent_theta=r"$\theta$ cand. $q_{\phi_\theta}$",
    latent_prior="$p(z)\\!=\\!\\mathcal{N}(0,I)$\n(dashed, latent prior)",
    mvae_io_enc="Encoder",
    mvae_io_dec="Decoder",
    mvae_io_input_x="FRF",
    mvae_io_input_th="parameter",
    mvae_io_latent="Latent Space",
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


# ---------- figure 1: pipeline ----------


def make_pipeline(loc: L10n):
    fig, ax = plt.subplots(figsize=(13, 6))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 6.5)
    ax.axis("off")

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
    ax.text(0.4, 5.85, loc.pipeline_offline, fontsize=13,
            color="#1a3a5c", fontweight="bold")
    ax.text(0.4, 2.45, loc.pipeline_online, fontsize=13,
            color="#c0392b", fontweight="bold")

    _box(ax, (1.5, 4.5), 1.8, 0.9, loc.pipeline_prior)
    _box(ax, (3.9, 4.5), 1.8, 0.9, loc.pipeline_fe)
    _box(ax, (6.3, 4.5), 1.8, 0.9, loc.pipeline_pair)
    _box(ax, (8.7, 4.5), 1.8, 0.9, loc.pipeline_train, fc="#dfeaf4", fontweight="bold")
    _box(ax, (11.5, 4.5), 2.2, 1.1, loc.pipeline_lhat,
         fc="#fff7d6", ec="#b9871a", fontsize=13, fontweight="bold")

    _arrow(ax, (2.4, 4.5), (3.0, 4.5))
    _arrow(ax, (4.8, 4.5), (5.4, 4.5))
    _arrow(ax, (7.2, 4.5), (7.8, 4.5))
    _arrow(ax, (9.6, 4.5), (10.4, 4.5))

    ax.text(8.7, 5.4, loc.pipeline_ml_note,
            ha="center", fontsize=12, color="#b9871a", fontweight="bold")

    _box(ax, (1.5, 1.4), 1.8, 0.9, loc.pipeline_obs,
         fc="#fbe6e1", ec="#c0392b")
    _box(ax, (6.3, 1.4), 2.0, 0.9, loc.pipeline_smc,
         fc="#fbe6e1", ec="#c0392b", fontweight="bold")
    _box(ax, (11.5, 1.4), 2.2, 0.9, loc.pipeline_posterior,
         fc="#fbe6e1", ec="#c0392b", fontweight="bold")

    _arrow(ax, (2.4, 1.4), (5.3, 1.4), color="#c0392b")
    _arrow(ax, (7.3, 1.4), (10.4, 1.4), color="#c0392b")
    _arrow(ax, (11.0, 3.9), (6.9, 1.9), color="#b9871a", style="->", lw=2.0)
    ax.text(9.4, 2.95, loc.pipeline_lhat_only,
            ha="center", va="center", fontsize=11,
            color="#b9871a", fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", fc="white",
                      ec="#b9871a", lw=1))

    fig.savefig(ASSETS / f"pipeline{loc.suffix}.png")
    plt.close(fig)


# ---------- figure 2: VAE vs MVAE ----------


def make_vae_mvae_compare(loc: L10n):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.0))

    for ax, title in zip(axes, [loc.vae_title, loc.mvae_title]):
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
    _box(ax, (10.5, 3.0), 1.3, 1.0, loc.vae_recon,
         fc="#fbe6e1", ec="#c0392b", fontsize=12)
    _arrow(ax, (1.7, 3.0), (2.45, 3.0))
    _arrow(ax, (4.35, 3.0), (5.1, 3.0))
    _arrow(ax, (6.3, 3.0), (7.05, 3.0))
    _arrow(ax, (8.95, 3.0), (9.85, 3.0))

    ax.text(6, 1.5, loc.vae_loss, ha="center", fontsize=13)
    ax.text(6, 0.6, loc.vae_meta, ha="center", fontsize=12, color="#555")

    # ---- right: MVAE ----
    ax = axes[1]
    _box(ax, (1.0, 4.0), 1.1, 0.9, "$x$", fc="#fbe6e1", ec="#c0392b")
    _box(ax, (1.0, 2.0), 1.1, 0.9, "$\\theta$", fc="#e1f0e7", ec="#2c8a4f")
    _box(ax, (3.2, 4.0), 1.9, 0.9, "Encoder $q_{\\phi_x}$",
         fc="#fbe6e1", ec="#c0392b", fontsize=12)
    _box(ax, (3.2, 2.0), 1.9, 0.9, "Encoder $q_{\\phi_\\theta}$",
         fc="#e1f0e7", ec="#2c8a4f", fontsize=12)
    _box(ax, (6.0, 3.0), 1.3, 1.0, loc.mvae_shared_z, fc="#fff7d6",
         ec="#b9871a", fontweight="bold")
    _box(ax, (8.3, 3.0), 1.9, 1.0, "Decoder\n$p_\\eta(x|z)$")
    _box(ax, (10.8, 3.0), 1.3, 1.0, loc.vae_recon,
         fc="#fbe6e1", ec="#c0392b", fontsize=12)

    _arrow(ax, (1.55, 4.0), (2.25, 4.0))
    _arrow(ax, (1.55, 2.0), (2.25, 2.0))
    _arrow(ax, (4.15, 4.0), (5.4, 3.3))
    _arrow(ax, (4.15, 2.0), (5.4, 2.7))
    _arrow(ax, (6.65, 3.0), (7.4, 3.0))
    _arrow(ax, (9.25, 3.0), (10.15, 3.0))

    ax.annotate("", xy=(3.2, 3.45), xytext=(3.2, 2.55),
                arrowprops=dict(arrowstyle="<->", color="#b9871a", lw=2))
    ax.text(3.7, 3.0, loc.mvae_kl_align,
            ha="left", va="center", fontsize=11,
            color="#b9871a", fontweight="bold")

    ax.text(6, 1.0, loc.mvae_loss, ha="center", fontsize=12)
    ax.text(6, 0.3, loc.mvae_meta, ha="center", fontsize=12, color="#555")

    fig.savefig(ASSETS / f"vae_mvae_compare{loc.suffix}.png")
    plt.close(fig)


# ---------- figure 3: latent overlap ----------


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


def make_latent_overlap(loc: L10n):
    fig, ax = plt.subplots(figsize=(7, 6.5))

    mu_obs = np.array([1.0, 0.7])
    cov_obs = np.array([[0.35, 0.08], [0.08, 0.45]])
    mu_th = np.array([0.5, 0.2])
    cov_th = np.array([[0.45, -0.04], [-0.04, 0.30]])

    ax.set_title(loc.latent_title, fontsize=14, color="#1a3a5c")
    _gauss2d_contour(ax, mu_obs, cov_obs, "#c0392b", loc.latent_obs)
    _gauss2d_contour(ax, mu_th, cov_th, "#2c8a4f", loc.latent_theta)

    x = np.linspace(-3, 3, 200)
    y = np.linspace(-3, 3, 200)
    X, Y = np.meshgrid(x, y)
    R2 = X**2 + Y**2
    ax.contour(X, Y, R2, levels=[1, 4], colors="#888", linewidths=1.2,
               linestyles="dashed", alpha=0.8)
    ax.text(-2.4, 2.15, loc.latent_prior, fontsize=11, color="#666")

    ax.set_xlim(-2.5, 2.5)
    ax.set_ylim(-2.5, 2.5)
    ax.set_aspect("equal")
    ax.set_xlabel("$z_1$", fontsize=13)
    ax.set_ylabel("$z_2$", fontsize=13)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(ASSETS / f"latent_overlap{loc.suffix}.png")
    plt.close(fig)


# ---------- figure 4: MVAE I/O (paper Fig.1 style) ----------


def make_mvae_io(loc: L10n):
    fig, ax = plt.subplots(figsize=(13, 5.0))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 5.0)
    ax.axis("off")

    orange = "#e07a3c"
    green = "#2c8a4f"
    outline = "#1a3a5c"
    prior_gray = "#b3b3b3"

    # ---- input FRF (mini plot, top-left) ----
    rng = np.random.default_rng(7)
    f = np.linspace(0, 1, 400)
    frf = (
        np.exp(-((f - 0.16) / 0.022) ** 2) * 1.00
        + np.exp(-((f - 0.34) / 0.030) ** 2) * 0.72
        + np.exp(-((f - 0.56) / 0.045) ** 2) * 0.45
        + np.exp(-((f - 0.82) / 0.035) ** 2) * 0.55
        + 0.04
        + np.abs(rng.normal(0, 0.012, len(f)))
    )
    in_x = ax.inset_axes([0.025, 0.69, 0.090, 0.20])
    in_x.plot(f, frf, color="black", linewidth=0.7)
    in_x.set_xticks([])
    in_x.set_yticks([])
    for s in in_x.spines.values():
        s.set_visible(False)
    ax.text(0.91, 4.66, loc.mvae_io_input_x, fontsize=12,
            ha="center", color=outline)
    ax.text(1.92, 3.95, r"$\mathbf{x}$", fontsize=18,
            color=outline, fontweight="bold", va="center")

    # ---- input parameter theta (mini bar chart, bottom-left) ----
    in_th = ax.inset_axes([0.025, 0.18, 0.090, 0.18])
    bars = [0.62, 0.85, 1.05, 0.78]
    in_th.bar(range(4), bars, color="#777", edgecolor="black",
              linewidth=0.4, width=0.72)
    in_th.set_ylim(0, 1.3)
    in_th.set_xticks([])
    in_th.set_yticks([])
    for s in in_th.spines.values():
        s.set_visible(False)
    ax.text(0.91, 0.55, loc.mvae_io_input_th, fontsize=12,
            ha="center", color=outline)
    ax.text(1.92, 1.20, r"$\boldsymbol{\theta}$", fontsize=18,
            color=outline, fontweight="bold", va="center")

    # ---- encoder q_phi_x (trapezoid narrowing right) ----
    enc_x = Polygon(
        [(2.45, 4.70), (4.30, 4.25), (4.30, 3.30), (2.45, 2.85)],
        closed=True, facecolor="white", edgecolor=outline, linewidth=2.0,
    )
    ax.add_patch(enc_x)
    ax.text(3.38, 3.90, loc.mvae_io_enc, ha="center", va="center",
            fontsize=13, fontweight="bold", color=outline)
    ax.text(3.38, 3.50, r"$q_{\phi_x}(z\mid x)$", ha="center", va="center",
            fontsize=12, color=outline)

    # orange z output bar
    ax.add_patch(Rectangle((4.40, 3.35), 0.20, 0.85,
                           facecolor=orange, edgecolor=orange))
    ax.text(4.83, 3.78, r"$\mathbf{z}$", fontsize=13,
            color=orange, fontweight="bold", va="center")

    # ---- encoder q_phi_theta (trapezoid narrowing right) ----
    enc_th = Polygon(
        [(2.45, 1.95), (4.30, 1.50), (4.30, 0.55), (2.45, 0.10)],
        closed=True, facecolor="white", edgecolor=outline, linewidth=2.0,
    )
    ax.add_patch(enc_th)
    ax.text(3.38, 1.15, loc.mvae_io_enc, ha="center", va="center",
            fontsize=13, fontweight="bold", color=outline)
    ax.text(3.38, 0.75, r"$q_{\phi_\theta}(z\mid \theta)$",
            ha="center", va="center", fontsize=12, color=outline)

    # green z output bar
    ax.add_patch(Rectangle((4.40, 0.60), 0.20, 0.85,
                           facecolor=green, edgecolor=green))
    ax.text(4.83, 1.02, r"$\mathbf{z}$", fontsize=13,
            color=green, fontweight="bold", va="center")

    # ---- Latent Space box ----
    lx, ly, lw, lh = 5.45, 1.30, 3.60, 3.10
    ax.add_patch(Rectangle((lx, ly), lw, lh,
                           facecolor="white", edgecolor=outline, linewidth=1.5))
    ax.text(lx + lw / 2, ly + lh - 0.25, loc.mvae_io_latent,
            ha="center", fontsize=12.5, color=outline)

    # prior gray rings
    cx_p, cy_p = lx + lw / 2, ly + lh / 2 - 0.05
    for r in (1.15, 0.78, 0.45):
        ax.add_patch(Circle((cx_p, cy_p), r, facecolor="none",
                            edgecolor=prior_gray, linewidth=1.1))

    # orange gaussian
    cx_o, cy_o = cx_p - 0.55, cy_p + 0.45
    for r in (0.55, 0.32):
        ax.add_patch(Circle((cx_o, cy_o), r, facecolor="none",
                            edgecolor=orange, linewidth=1.6))
    ax.add_patch(Circle((cx_o, cy_o), 0.08,
                        facecolor=orange, edgecolor=orange))

    # green gaussian
    cx_g, cy_g = cx_p + 0.45, cy_p - 0.30
    for r in (0.55, 0.32):
        ax.add_patch(Circle((cx_g, cy_g), r, facecolor="none",
                            edgecolor=green, linewidth=1.6))
    ax.add_patch(Circle((cx_g, cy_g), 0.08,
                        facecolor=green, edgecolor=green))

    ax.text(cx_p, ly + 0.30, r"$q_{\mathcal{Z}}(z)$",
            ha="center", fontsize=12, color=outline)

    # dashed arrows: encoder outputs → latent clouds
    ax.annotate("", xy=(cx_o - 0.45, cy_o + 0.05), xytext=(4.78, 3.78),
                arrowprops=dict(arrowstyle="->", color=orange,
                                linewidth=1.3, linestyle=(0, (5, 3))))
    ax.annotate("", xy=(cx_g - 0.45, cy_g - 0.20), xytext=(4.78, 1.02),
                arrowprops=dict(arrowstyle="->", color=green,
                                linewidth=1.3, linestyle=(0, (5, 3))))

    # dashed arrows: latent clouds → sampled z (right side)
    z_bar_x = 9.55
    ax.annotate("", xy=(z_bar_x, 3.05), xytext=(cx_o + 0.4, cy_o + 0.0),
                arrowprops=dict(arrowstyle="->", color=orange,
                                linewidth=1.3, linestyle=(0, (5, 3))))
    ax.annotate("", xy=(z_bar_x, 2.65), xytext=(cx_g + 0.4, cy_g - 0.0),
                arrowprops=dict(arrowstyle="->", color=green,
                                linewidth=1.3, linestyle=(0, (5, 3))))

    # ---- sampled z bar (gray) ----
    ax.add_patch(Rectangle((z_bar_x, 2.40), 0.20, 0.85,
                           facecolor="#5a5a5a", edgecolor="#5a5a5a"))
    ax.text(z_bar_x + 0.40, 2.83, r"$\mathbf{z}$", fontsize=13,
            color="#333", fontweight="bold", va="center")

    # ---- decoder (trapezoid widening right) ----
    dec = Polygon(
        [(10.30, 3.40), (12.15, 3.85), (12.15, 1.85), (10.30, 2.30)],
        closed=True, facecolor="white", edgecolor=outline, linewidth=2.0,
    )
    ax.add_patch(dec)
    ax.text(11.22, 3.00, loc.mvae_io_dec, ha="center", va="center",
            fontsize=13, fontweight="bold", color=outline)
    ax.text(11.22, 2.62, r"$p_\eta(x\mid z)$", ha="center", va="center",
            fontsize=12, color=outline)

    # ---- reconstructed FRF (mini plot, far right) ----
    out_x = ax.inset_axes([0.952, 0.49, 0.040, 0.20])
    out_x.plot(f, frf * 0.95, color="black", linewidth=0.7)
    out_x.set_xticks([])
    out_x.set_yticks([])
    for s in out_x.spines.values():
        s.set_visible(False)
    ax.text(12.60, 2.85, r"$\mathbf{x}$", fontsize=18,
            color=outline, fontweight="bold", va="center")

    fig.savefig(ASSETS / f"mvae_io{loc.suffix}.png")
    plt.close(fig)


if __name__ == "__main__":
    for loc in (JA, EN):
        make_pipeline(loc)
        make_vae_mvae_compare(loc)
        make_latent_overlap(loc)
        make_mvae_io(loc)
    print("Generated:")
    for p in sorted(ASSETS.glob("*.png")):
        print(f"  - {p.relative_to(ASSETS.parent.parent.parent)}")
