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


def generate_intro_figure(output_path: Path) -> None:
    x = np.linspace(-4, 4, 1000)
    dx = x[1] - x[0]
    p_s = norm.pdf(x, loc=-1, scale=1)
    p_t = norm.pdf(x, loc=1, scale=1)
    p = p_t / (p_s + p_t)
    fig, axes = plt.subplots(2, 3, figsize=(12, 7), sharey="row", constrained_layout=True)
    for col, mode in enumerate(["source", "target", "both"]):
        ax = axes[0, col]
        w = domain_weights(source=p, target=p, reweight=mode, shrinkage=0.5)
        ws = p_s * w.source / np.sum(p_s * w.source * dx)
        wt = p_t * w.target / np.sum(p_t * w.target * dx)
        ax.plot(x, p_s, color="tab:blue", linestyle="--", alpha=0.4)
        ax.fill_between(x, p_s, alpha=0.06, color="tab:blue")
        ax.plot(x, p_t, color="tab:orange", linestyle="--", alpha=0.4)
        ax.fill_between(x, p_t, alpha=0.06, color="tab:orange")
        if mode == "source":
            ax.plot(x, ws, color="tab:blue", linewidth=2.5)
        elif mode == "target":
            ax.plot(x, wt, color="tab:orange", linewidth=2.5)
        else:
            ax.plot(x, ws, color="#7570b3", linewidth=2.5)
            ax.plot(x, wt, color="#7570b3", linewidth=2.5)
        ax.set_title(MODE_TITLES[mode], fontsize=12)
        ax.set_xlabel("Feature $X$")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(alpha=0.25, linewidth=0.8)
        if col == 0:
            ax.set_ylabel("Density")
    ax = axes[1, 0]
    sm, tm = crump_trimming_mask(p, p)
    cs = p_s * sm / np.sum(p_s * sm * dx)
    ct = p_t * tm / np.sum(p_t * tm * dx)
    ax.plot(x, p_s, color="tab:blue", linestyle="--", alpha=0.4)
    ax.fill_between(x, p_s, alpha=0.06, color="tab:blue")
    ax.plot(x, p_t, color="tab:orange", linestyle="--", alpha=0.4)
    ax.fill_between(x, p_t, alpha=0.06, color="tab:orange")
    ax.plot(x, cs, color="tab:blue", linewidth=2.5)
    ax.plot(x, ct, color="tab:orange", linewidth=2.5)
    ax.fill_between(x, 0, p_s * (1 - sm), alpha=0.2, color="gray")
    ax.fill_between(x, 0, p_t * (1 - tm), alpha=0.2, color="gray")
    ax.set_title(MODE_TITLES["crump"], fontsize=12)
    ax.set_xlabel("Feature $X$")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(alpha=0.25, linewidth=0.8)
    ax.set_ylabel("Density")
    ax = axes[1, 1]
    ow_s, ow_t = estimate_overlap_weights(p, p)
    os = p_s * ow_s / np.sum(p_s * ow_s * dx)
    ot = p_t * ow_t / np.sum(p_t * ow_t * dx)
    ax.plot(x, p_s, color="tab:blue", linestyle="--", alpha=0.4)
    ax.fill_between(x, p_s, alpha=0.06, color="tab:blue")
    ax.plot(x, p_t, color="tab:orange", linestyle="--", alpha=0.4)
    ax.fill_between(x, p_t, alpha=0.06, color="tab:orange")
    ax.plot(x, os, color="tab:blue", linewidth=2.5)
    ax.plot(x, ot, color="tab:orange", linewidth=2.5)
    ax.set_title(MODE_TITLES["overlap"], fontsize=12)
    ax.set_xlabel("Feature $X$")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(alpha=0.25, linewidth=0.8)
    ax = axes[1, 2]
    pg = np.linspace(0.001, 0.999, 500)
    ax.plot(pg, pg * (1 - pg), color="#e6ab02", linewidth=2.5, label="Overlap")
    ax.plot(pg, (np.minimum(pg, 1 - pg) >= 0.1).astype(float) * 0.25, color="#e7298a", linewidth=2.5, label="Crump (x0.25)")
    ax.axvline(0.1, color="#e7298a", linestyle=":", alpha=0.5)
    ax.axvline(0.9, color="#e7298a", linestyle=":", alpha=0.5)
    r = pg / (1 - pg)
    lam = 0.5
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
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


@app.command()
def main(output: Path = typer.Option(MANUSCRIPT_DIR / "figures/intro_reweighting_modes.pdf")) -> None:
    generate_intro_figure(output)


if __name__ == "__main__":
    app()
