"""Generate presentation figures (Japanese and English variants).

Run from project root:
    uv run python docs/presentation/assets/make_figures.py
"""

from dataclasses import dataclass
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
    mvae_io_title: str
    mvae_io_enc_x: str
    mvae_io_enc_th: str
    mvae_io_dec: str


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
    mvae_io_title="MVAE: 2 つのエンコーダ + 1 つのデコーダの入出力",
    mvae_io_enc_x="観測側エンコーダ $q_{\\phi_x}(z\\,|\\,x)$",
    mvae_io_enc_th="パラメータ側エンコーダ $q_{\\phi_\\theta}(z\\,|\\,\\theta)$",
    mvae_io_dec="デコーダ $p_\\eta(x\\,|\\,z)$",
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
    mvae_io_title="MVAE: I/O of the two encoders and one decoder",
    mvae_io_enc_x="Observation encoder $q_{\\phi_x}(z\\,|\\,x)$",
    mvae_io_enc_th="Parameter encoder $q_{\\phi_\\theta}(z\\,|\\,\\theta)$",
    mvae_io_dec="Decoder $p_\\eta(x\\,|\\,z)$",
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


# ---------- figure 4: MVAE I/O ----------


def make_mvae_io(loc: L10n):
    fig, ax = plt.subplots(figsize=(12, 4.3))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 4.3)
    ax.axis("off")

    ax.text(
        6.0, 3.95, loc.mvae_io_title,
        ha="center", fontsize=15, color="#1a3a5c", fontweight="bold",
    )

    rows = [
        # (y, input_label, input_color, net_label, net_fc, net_ec, output_label)
        (
            2.85,
            "$x$", ("#fbe6e1", "#c0392b"),
            loc.mvae_io_enc_x, "#fbe6e1", "#c0392b",
            "$(\\mu_x,\\, \\sigma_x^2)$",
        ),
        (
            1.75,
            "$\\theta$", ("#e1f0e7", "#2c8a4f"),
            loc.mvae_io_enc_th, "#e1f0e7", "#2c8a4f",
            "$(\\mu_\\theta,\\, \\sigma_\\theta^2)$",
        ),
        (
            0.6,
            "$z$", ("#fff7d6", "#b9871a"),
            loc.mvae_io_dec, "#eef3f8", "#1a3a5c",
            "$(\\hat\\mu_x,\\, \\hat\\sigma_x^2)$",
        ),
    ]

    for y, in_lbl, in_col, net_lbl, net_fc, net_ec, out_lbl in rows:
        _box(ax, (1.0, y), 1.2, 0.8, in_lbl, fc=in_col[0], ec=in_col[1], fontsize=15)
        _box(ax, (5.5, y), 4.2, 0.95, net_lbl, fc=net_fc, ec=net_ec, fontsize=13)
        _box(ax, (10.4, y), 2.4, 0.95, out_lbl,
             fc="#fff7d6", ec="#b9871a", fontsize=13)
        _arrow(ax, (1.65, y), (3.35, y))
        _arrow(ax, (7.65, y), (9.15, y))

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
