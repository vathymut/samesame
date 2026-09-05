#!/usr/bin/env python3
"""Generate Option A: Weight as a function of domain probability p."""
import matplotlib.pyplot as plt
import numpy as np

# --- Publication defaults ---
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold",
    "axes.labelsize": 10, "legend.fontsize": 8.5, "legend.frameon": False,
    "figure.dpi": 300, "savefig.dpi": 300, "savefig.bbox": "tight",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.15, "grid.linestyle": "-",
    "lines.linewidth": 2.2, "lines.markersize": 5,
})

# --- Color palette ---
COLORS = {
    "unweighted": "#B0BEC5",       # gray
    "crump": "#8C8C8C",            # darker gray
    "overlap": "#2A9D8F",          # teal
    "source": "#E9C46A",           # gold
    "target": "#F4A261",           # sandy orange
    "doubly": "#E76F51",           # coral (our method)
}

# Domain probability range
p = np.linspace(0.001, 0.999, 500)
prior_ratio = 1.0  # n_s / n_t
lambda_riw = 0.5   # stabilization parameter

# --- Weight functions ---
def w_unweighted(p):
    return np.ones_like(p)

def w_crump(p, threshold=0.1):
    """Crump trimming: binary mask"""
    mask = np.minimum(p, 1 - p) >= threshold
    return mask.astype(float)

def w_overlap(p):
    """Overlap weights: p(1-p)"""
    return p * (1 - p)

def w_riw_source(p, prior_ratio, lam):
    """Stabilized RIW for source: forward correction"""
    r = (p / (1 - p)) * prior_ratio
    return r / ((1 - lam) + lam * r)

def w_riw_target(p, prior_ratio, lam):
    """Stabilized RIW for target: inverse correction"""
    r_inv = ((1 - p) / p) / prior_ratio
    return r_inv / ((1 - lam) + lam * r_inv)

# --- Create figure ---
fig, ax = plt.subplots(figsize=(6.75, 3.2))

# Plot each weighting scheme
ax.plot(p, w_unweighted(p), label="Unweighted", color=COLORS["unweighted"],
        linestyle="--", linewidth=1.8, alpha=0.7)
ax.plot(p, w_crump(p), label="Crump (threshold=0.1)", color=COLORS["crump"],
        linestyle=":", linewidth=2.5)
ax.plot(p, w_overlap(p), label="Overlap", color=COLORS["overlap"],
        linewidth=2.2)
ax.plot(p, w_riw_source(p, prior_ratio, lambda_riw), label="RIW Source (forward)",
        color=COLORS["source"], linewidth=2.2)
ax.plot(p, w_riw_target(p, prior_ratio, lambda_riw), label="RIW Target (inverse)",
        color=COLORS["target"], linewidth=2.2)
# Doubly weighted is the product (normalized)
w_doubly = w_riw_source(p, prior_ratio, lambda_riw) * w_riw_target(p, prior_ratio, lambda_riw)
w_doubly = w_doubly / np.max(w_doubly)  # normalize for visualization
ax.plot(p, w_doubly, label="RIW Doubly (both)", color=COLORS["doubly"],
        linewidth=2.5, zorder=5)

# Annotations
ax.axvline(0.5, color="#CCCCCC", linestyle="--", linewidth=1, zorder=1)
ax.text(0.5, 1.05, "Balanced\noverlap", ha="center", va="bottom",
        fontsize=8, color="#666666")

# Key insight annotation
ax.annotate("Overlap: symmetric\n(same for both groups)",
            xy=(0.5, 0.25), xytext=(0.72, 0.15),
            arrowprops=dict(arrowstyle="->", color="#2A9D8F", lw=1.2),
            fontsize=7.5, color="#2A9D8F", ha="left",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#2A9D8F", linewidth=0.8))

ax.annotate("RIW: asymmetric\n(forward ≠ inverse)",
            xy=(0.15, w_riw_source(0.15, prior_ratio, lambda_riw)),
            xytext=(0.08, 0.7),
            arrowprops=dict(arrowstyle="->", color="#E9C46A", lw=1.2),
            fontsize=7.5, color="#E9C46A", ha="left",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#E9C46A", linewidth=0.8))

ax.set_xlabel("Domain Probability p (Pr[observation from target | x])")
ax.set_ylabel("Relative Weight")
ax.set_xlim(0, 1)
ax.set_ylim(-0.05, 1.15)
ax.legend(loc="upper left", ncol=2, frameon=True, fancybox=False,
          edgecolor="#DDDDDD", framealpha=0.95)
ax.set_title("Weighting Functions: How Each Scheme Assigns Weights")

fig.savefig("conceptual_option_a.pdf")
fig.savefig("conceptual_option_a.png", dpi=300)
print("Generated: conceptual_option_a.pdf and .png")
