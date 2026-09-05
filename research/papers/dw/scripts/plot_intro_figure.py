"""Generate the conceptual introduction figure for all weighting approaches.

Panels:
  Row 1: RIW modes over feature space (source-weighted, target-weighted,
          doubly weighted) — existing conceptual display.
  Row 2: Crump-trimmed (binary keep/drop), Overlap-weighted (continuous p(1-p)),
          and a reference panel showing weight-vs-domain-probability curves
          for all three approaches on a common axis.

This expanded figure lets readers visually compare the three weighting
families before encountering Table 1 (weighting_benchmarks.tex).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import typer
from scipy.stats import norm

from samesame.weights import domain_weights
from scripts._repo import MANUSCRIPT_DIR
from scripts.weighting import crump_trimming_mask, estimate_overlap_weights

app = typer.Typer()

MODE_TITLES = {
    "source": "Source-weighted",
    "target": "Target-weighted",
    "both": "Doubly weighted",
    "crump": "Crump-trimmed",
    "overlap": "Overlap-weighted",
}


def generate_intro_figure(output_path: Path) -> None:
    x = np.linspace(-4, 4, 1000)
    dx = x[1] - x[0]
    p_source = norm.pdf(x, loc=-1, scale=1)
    p_target = norm.pdf(x, loc=1, scale=1)

    domain_probs = p_target / (p_source + p_target)

    fig, axes = plt.subplots(
        2, 3, figsize=(12, 7), sharey="row", constrained_layout=True
    )

    riw_modes = ["source", "target", "both"]
    for col, mode in enumerate(riw_modes):
        ax = axes[0, col]
        weights = domain_weights(
            source=domain_probs,
            target=domain_probs,
            reweight=mode,
            shrinkage=0.5,
        )

        w_source_pdf = p_source * weights.source
        w_source_pdf = w_source_pdf / np.sum(w_source_pdf * dx)
        w_target_pdf = p_target * weights.target
        w_target_pdf = w_target_pdf / np.sum(w_target_pdf * dx)

        ax.plot(x, p_source, color="tab:blue", linestyle="--", alpha=0.4)
        ax.fill_between(x, p_source, alpha=0.06, color="tab:blue")
        ax.plot(x, p_target, color="tab:orange", linestyle="--", alpha=0.4)
        ax.fill_between(x, p_target, alpha=0.06, color="tab:orange")

        if mode == "source":
            ax.plot(x, w_source_pdf, color="tab:blue", linewidth=2.5)
        elif mode == "target":
            ax.plot(x, w_target_pdf, color="tab:orange", linewidth=2.5)
        elif mode == "both":
            ax.plot(x, w_source_pdf, color="#7570b3", linewidth=2.5)
            ax.plot(x, w_target_pdf, color="#7570b3", linewidth=2.5)

        ax.set_title(MODE_TITLES[mode], fontsize=12)
        ax.set_xlabel("Feature $X$")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(alpha=0.25, linewidth=0.8)
        if col == 0:
            ax.set_ylabel("Density")

    ax = axes[1, 0]
    src_mask, tgt_mask = crump_trimming_mask(domain_probs, domain_probs)
    c_source_pdf = p_source * src_mask
    c_source_pdf = c_source_pdf / np.sum(c_source_pdf * dx)
    c_target_pdf = p_target * tgt_mask
    c_target_pdf = c_target_pdf / np.sum(c_target_pdf * dx)

    ax.plot(x, p_source, color="tab:blue", linestyle="--", alpha=0.4)
    ax.fill_between(x, p_source, alpha=0.06, color="tab:blue")
    ax.plot(x, p_target, color="tab:orange", linestyle="--", alpha=0.4)
    ax.fill_between(x, p_target, alpha=0.06, color="tab:orange")
    ax.plot(x, c_source_pdf, color="tab:blue", linewidth=2.5)
    ax.plot(x, c_target_pdf, color="tab:orange", linewidth=2.5)
    ax.fill_between(x, 0, p_source * (1 - src_mask), alpha=0.2, color="gray",
                    label="Discarded")
    ax.fill_between(x, 0, p_target * (1 - tgt_mask), alpha=0.2, color="gray")
    ax.set_title(MODE_TITLES["crump"], fontsize=12)
    ax.set_xlabel("Feature $X$")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(alpha=0.25, linewidth=0.8)
    ax.set_ylabel("Density")

    ax = axes[1, 1]
    ow_src, ow_tgt = estimate_overlap_weights(domain_probs, domain_probs)
    o_source_pdf = p_source * ow_src
    o_source_pdf = o_source_pdf / np.sum(o_source_pdf * dx)
    o_target_pdf = p_target * ow_tgt
    o_target_pdf = o_target_pdf / np.sum(o_target_pdf * dx)

    ax.plot(x, p_source, color="tab:blue", linestyle="--", alpha=0.4)
    ax.fill_between(x, p_source, alpha=0.06, color="tab:blue")
    ax.plot(x, p_target, color="tab:orange", linestyle="--", alpha=0.4)
    ax.fill_between(x, p_target, alpha=0.06, color="tab:orange")
    ax.plot(x, o_source_pdf, color="tab:blue", linewidth=2.5, label="Source (wtd)")
    ax.plot(x, o_target_pdf, color="tab:orange", linewidth=2.5, label="Target (wtd)")
    ax.set_title(MODE_TITLES["overlap"], fontsize=12)
    ax.set_xlabel("Feature $X$")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(alpha=0.25, linewidth=0.8)

    ax = axes[1, 2]
    p_grid = np.linspace(0.001, 0.999, 500)

    overlap_curve = p_grid * (1 - p_grid)
    ax.plot(p_grid, overlap_curve, color="#e6ab02", linewidth=2.5,
            label="Overlap")

    crump_curve = (np.minimum(p_grid, 1 - p_grid) >= 0.1).astype(float)
    ax.plot(p_grid, crump_curve * 0.25, color="#e7298a", linewidth=2.5,
            label="Crump (x0.25 for vis.)")
    ax.axvline(0.1, color="#e7298a", linestyle=":", alpha=0.5)
    ax.axvline(0.9, color="#e7298a", linestyle=":", alpha=0.5)

    r_grid = p_grid / (1 - p_grid)
    lam = 0.5
    riw_source_curve = r_grid / ((1 - lam) + lam * r_grid)
    riw_target_curve = 1.0 / (lam + (1 - lam) * r_grid)
    ax.plot(p_grid, riw_source_curve, color="#1b9e77", linewidth=2.5,
            label="RIW source", linestyle="--")
    ax.plot(p_grid, riw_target_curve, color="#d95f02", linewidth=2.5,
            label="RIW target", linestyle="--")
    ax.plot(p_grid, riw_source_curve * riw_target_curve, color="#7570b3",
            linewidth=2.5, label="RIW both (product)")

    ax.set_title("Weight as function of $p = P(\\text{target} \\mid x)$",
                 fontsize=11)
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
def main(
    output: Path = typer.Option(
        MANUSCRIPT_DIR / "figures/intro_reweighting_modes.pdf",
    ),
) -> None:
    generate_intro_figure(output)


if __name__ == "__main__":
    app()
