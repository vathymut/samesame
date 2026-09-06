"""Conceptual intro figure comparing weighting families."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import typer
from scipy.stats import norm

from samesame.weights import domain_weights
from scripts.experiments import crump_trimming_mask, estimate_overlap_weights
from scripts.utils import MANUSCRIPT_DIR

app = typer.Typer()

MODE_TITLES = {"source": "Source-weighted", "target": "Target-weighted", "both": "Doubly weighted", "crump": "Crump-trimmed", "overlap": "Overlap-weighted"}
PANEL_COLORS = {
    "source": ("tab:blue", None),
    "target": (None, "tab:orange"),
    "both": ("#7570b3", "#7570b3"),
    "crump": ("tab:blue", "tab:orange"),
    "overlap": ("tab:blue", "tab:orange"),
}


def _background(ax, x, p_s, p_t) -> None:
    ax.plot(x, p_s, color="tab:blue", linestyle="--", alpha=0.4)
    ax.fill_between(x, p_s, alpha=0.06, color="tab:blue")
    ax.plot(x, p_t, color="tab:orange", linestyle="--", alpha=0.4)
    ax.fill_between(x, p_t, alpha=0.06, color="tab:orange")
    ax.set_xlabel("Feature $X$")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(alpha=0.25, linewidth=0.8)


def _panel(ax, x, p_s, p_t, mode: str, ds, dx) -> None:
    _background(ax, x, p_s, p_t)
    c_s, c_t = PANEL_COLORS[mode]
    if c_s:
        ax.plot(x, ds[0], color=c_s, linewidth=2.5)
    if c_t:
        ax.plot(x, ds[1], color=c_t, linewidth=2.5)


def _density(p_x, w, dx):
    return p_x * w / np.sum(p_x * w * dx)


def _riw_densities(p_s, p_t, p, mode: str, dx):
    w = domain_weights(source=p, target=p, reweight=mode, shrinkage=0.5)
    return _density(p_s, w.source, dx), _density(p_t, w.target, dx)


def _weight_functions(ax) -> None:
    pg = np.linspace(0.001, 0.999, 500)
    ax.plot(pg, pg * (1 - pg), color="#e6ab02", linewidth=2.5, label="Overlap")
    ax.plot(pg, (np.minimum(pg, 1 - pg) >= 0.1).astype(float) * 0.25, color="#e7298a", linewidth=2.5, label="Crump (x0.25)")
    ax.axvline(0.1, color="#e7298a", linestyle=":", alpha=0.5)
    ax.axvline(0.9, color="#e7298a", linestyle=":", alpha=0.5)
    lam = 0.5
    r = pg / (1 - pg)
    rs = r / ((1 - lam) + lam * r)
    rt = 1.0 / (lam + (1 - lam) * r)
    ax.plot(pg, rs, color="#1b9e77", linewidth=2.5, label="RIW source", linestyle="--")
    ax.plot(pg, rt, color="#d95f02", linewidth=2.5, label="RIW target", linestyle="--")
    ax.plot(pg, rs * rt, color="#7570b3", linewidth=2.5, label="RIW both (product)")
    ax.set_title("Weight as function of $p = P(\\text{target} \\mid x)$", fontsize=11)
    ax.set_xlabel("Domain probability $p$")
    ax.set_ylabel("Weight $w(p)$")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(alpha=0.25, linewidth=0.8)
    ax.legend(frameon=False, fontsize=8, ncol=1)


def generate_intro_figure(output_path: Path) -> None:
    x = np.linspace(-4, 4, 1000)
    dx = x[1] - x[0]
    p_s = norm.pdf(x, loc=-1, scale=1)
    p_t = norm.pdf(x, loc=1, scale=1)
    p = p_t / (p_s + p_t)
    fig, axes = plt.subplots(2, 3, figsize=(12, 7), sharey="row", constrained_layout=True)
    for col, mode in enumerate(("source", "target", "both")):
        _panel(axes[0, col], x, p_s, p_t, mode, _riw_densities(p_s, p_t, p, mode, dx), dx)
        axes[0, col].set_title(MODE_TITLES[mode], fontsize=12)
        if col == 0:
            axes[0, col].set_ylabel("Density")
    sm, tm = crump_trimming_mask(p, p)
    _panel(axes[1, 0], x, p_s, p_t, "crump", (_density(p_s, sm, dx), _density(p_t, tm, dx)), dx)
    axes[1, 0].fill_between(x, 0, p_s * (~sm), alpha=0.2, color="gray")
    axes[1, 0].fill_between(x, 0, p_t * (~tm), alpha=0.2, color="gray")
    axes[1, 0].set_ylabel("Density")
    ow_s, ow_t = estimate_overlap_weights(p, p)
    _panel(axes[1, 1], x, p_s, p_t, "overlap", (_density(p_s, ow_s, dx), _density(p_t, ow_t, dx)), dx)
    _weight_functions(axes[1, 2])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


@app.command()
def main(output: Path = typer.Option(MANUSCRIPT_DIR / "figures/intro_reweighting_modes.pdf")) -> None:
    generate_intro_figure(output)


if __name__ == "__main__":
    app()
