"""Visualize posterior samples from posterior.mat."""

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator

from lsbi_smc.shapes import PostSamples

MAT_PATH = "posterior.mat"
OUT_PATH = "posterior_plot.png"
PARAM_NAMES = ["k₁", "k₂", "k₃", "k₄"]
LLIM, ULIM = 0.33, 3.00

# True parameter values (physical space)
TRUE_PARAMS = np.array([1.0, 1.0, 1.0, 1.0])

# Equivalent solutions from docs (same roof FRF as true params)
EQUIV_SOLUTIONS = {
    "Equiv A": np.array([1.722, 0.636, 1.301, 0.701]),
    "Equiv B": np.array([1.999, 1.000, 0.500, 1.000]),
    "Equiv C": np.array([2.640, 0.647, 0.813, 0.720]),
}
EQUIV_COLORS = ["#E69F00", "#56B4E9", "#009E73"]


def load_pop() -> PostSamples:
    data = sio.loadmat(MAT_PATH)
    return data["pop"]  # (2000, 4)


def plot_corner(pop: PostSamples) -> Figure:
    n_params = pop.shape[1]
    fig = plt.figure(figsize=(12, 12))
    fig.suptitle(
        "Posterior distribution — 4-DOF shear building\n"
        "(★ = true params, markers = equivalent solutions with same roof FRF)",
        fontsize=12,
        y=0.98,
    )

    gs = gridspec.GridSpec(n_params, n_params, figure=fig, hspace=0.08, wspace=0.08)

    for row in range(n_params):
        for col in range(n_params):
            if col > row:
                continue

            ax = fig.add_subplot(gs[row, col])

            if row == col:
                # Diagonal: histogram with fixed range
                ax.hist(
                    pop[:, col],
                    bins=40,
                    color="#4C72B0",
                    edgecolor="white",
                    linewidth=0.3,
                    density=True,
                    range=(LLIM, ULIM),
                )
                ax.axvline(
                    TRUE_PARAMS[col],
                    color="black",
                    linewidth=2.0,
                    linestyle="-",
                    label=f"true={TRUE_PARAMS[col]:.2f}",
                )
                ax.axvline(
                    pop[:, col].mean(),
                    color="#DD4444",
                    linewidth=1.5,
                    linestyle="--",
                    label=f"post.mean={pop[:, col].mean():.3f}",
                )
                for (elbl, ev), ec in zip(EQUIV_SOLUTIONS.items(), EQUIV_COLORS, strict=True):
                    ax.axvline(
                        ev[col],
                        color=ec,
                        linewidth=1.2,
                        linestyle=":",
                        alpha=0.9,
                        label=f"{elbl}={ev[col]:.3f}",
                    )
                ax.set_xlim(LLIM, ULIM)
                if col == 0:
                    ax.legend(fontsize=5.5, loc="upper right", framealpha=0.8)
                ax.set_ylabel("density", fontsize=8)
            else:
                # Off-diagonal: scatter + KDE contours
                ax.scatter(
                    pop[:, col],
                    pop[:, row],
                    alpha=0.15,
                    s=2,
                    color="#4C72B0",
                    rasterized=True,
                )
                try:
                    from scipy.stats import gaussian_kde

                    xy = np.vstack([pop[:, col], pop[:, row]])
                    kde = gaussian_kde(xy)
                    xg, yg = np.mgrid[LLIM:ULIM:60j, LLIM:ULIM:60j]
                    z = kde(np.vstack([xg.ravel(), yg.ravel()])).reshape(xg.shape)
                    ax.contour(xg, yg, z, levels=4, colors="#DD4444", linewidths=0.8, alpha=0.8)
                except Exception:
                    pass
                # True params
                ax.plot(TRUE_PARAMS[col], TRUE_PARAMS[row], "k*", markersize=10, zorder=5)
                # Equivalent solutions
                for (_, ev), ec in zip(EQUIV_SOLUTIONS.items(), EQUIV_COLORS, strict=True):
                    ax.plot(
                        ev[col],
                        ev[row],
                        "^",
                        color=ec,
                        markersize=7,
                        zorder=5,
                        markeredgecolor="black",
                        markeredgewidth=0.5,
                    )
                ax.set_xlim(LLIM, ULIM)
                ax.set_ylim(LLIM, ULIM)

            if row == n_params - 1:
                ax.set_xlabel(PARAM_NAMES[col], fontsize=11)
            else:
                ax.set_xticklabels([])

            if col == 0 and row != 0:
                ax.set_ylabel(PARAM_NAMES[row], fontsize=11)
            elif row != col:
                ax.set_yticklabels([])

            ax.tick_params(labelsize=7)
            ax.xaxis.set_major_locator(MaxNLocator(4))
            ax.yaxis.set_major_locator(MaxNLocator(4))

    # Legend for scatter plots
    from matplotlib.lines import Line2D

    legend_elements = [
        Line2D(
            [0],
            [0],
            marker="*",
            color="black",
            linestyle="None",
            markersize=10,
            label="True params (1.0, 1.0, 1.0, 1.0)",
        ),
    ]
    for (elbl, ev), ec in zip(EQUIV_SOLUTIONS.items(), EQUIV_COLORS, strict=True):
        legend_elements.append(
            Line2D(
                [0],
                [0],
                marker="^",
                color=ec,
                linestyle="None",
                markersize=7,
                markeredgecolor="black",
                markeredgewidth=0.5,
                label=f"{elbl}: ({', '.join(str(round(v, 2)) for v in ev)})",
            )
        )
    fig.legend(
        handles=legend_elements,
        loc="upper right",
        bbox_to_anchor=(0.99, 0.97),
        fontsize=8,
        framealpha=0.9,
        title="Expected modes",
    )

    return fig


def print_summary(pop: PostSamples) -> None:
    print(f"{'Param':<6} {'Mean':>8} {'Std':>8} {'2.5%':>8} {'50%':>8} {'97.5%':>8}")
    print("-" * 46)
    for i, name in enumerate(PARAM_NAMES):
        col = pop[:, i]
        print(
            f"{name:<6} {col.mean():8.4f} {col.std():8.4f} "
            f"{np.percentile(col, 2.5):8.4f} {np.median(col):8.4f} {np.percentile(col, 97.5):8.4f}"
        )


if __name__ == "__main__":
    pop = load_pop()
    print_summary(pop)

    fig = plot_corner(pop)
    fig.savefig(OUT_PATH, dpi=150, bbox_inches="tight")
    print(f"\nSaved → {OUT_PATH}")
