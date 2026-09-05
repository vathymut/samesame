#!/usr/bin/env python3
"""Generate Option C: Hybrid (weight functions + concrete example)."""
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from sklearn.linear_model import LogisticRegression

# --- Publication defaults ---
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 9, "axes.titlesize": 10, "axes.titleweight": "bold",
    "axes.labelsize": 9, "legend.fontsize": 7.5, "legend.frameon": False,
    "figure.dpi": 300, "savefig.dpi": 300, "savefig.bbox": "tight",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.15, "grid.linestyle": "-",
    "lines.linewidth": 2.0,
})

# --- Color palette ---
COLORS = {
    "unweighted": "#B0BEC5",
    "crump": "#8C8C8C",
    "overlap": "#2A9D8F",
    "source": "#E9C46A",
    "target": "#F4A261",
    "doubly": "#E76F51",
}

# --- Generate synthetic data ---
np.random.seed(42)
n_source, n_target = 800, 1200
X_source = np.random.randn(n_source)
X_target = np.concatenate([
    np.random.randn(int(0.6 * n_target)) - 1.5,
    np.random.randn(int(0.4 * n_target)) + 0.5,
])

# Fit domain classifier
X_all = np.concatenate([X_source, X_target]).reshape(-1, 1)
y_all = np.concatenate([np.zeros(n_source), np.ones(n_target)])
clf = LogisticRegression(random_state=42, max_iter=1000)
clf.fit(X_all, y_all)
p_all = clf.predict_proba(X_all)[:, 1]
p_source = p_all[:n_source]
p_target = p_all[n_source:]

# --- Weight functions ---
def w_crump(p, threshold=0.1):
    mask = np.minimum(p, 1 - p) >= threshold
    return mask.astype(float)

def w_overlap(p):
    return p * (1 - p)

def w_riw_source(p, prior_ratio=1.0, lam=0.5):
    r = (p / (1 - p + 1e-10)) * prior_ratio
    return r / ((1 - lam) + lam * r)

def w_riw_target(p, prior_ratio=1.0, lam=0.5):
    r_inv = ((1 - p) / (p + 1e-10)) / prior_ratio
    return r_inv / ((1 - lam) + lam * r_inv)

prior_ratio = n_source / n_target
w_s_riw = w_riw_source(p_source, prior_ratio)
w_t_riw = w_riw_target(p_target, prior_ratio)

# Normalize weights to [0, 1] for use as alpha values
w_s_riw = w_s_riw / np.max(w_s_riw) if np.max(w_s_riw) > 0 else w_s_riw
w_t_riw = w_t_riw / np.max(w_t_riw) if np.max(w_t_riw) > 0 else w_t_riw

# --- Create hybrid figure: 1 row, 2 columns ---
fig = plt.figure(figsize=(6.75, 2.8))
gs = fig.add_gridspec(1, 2, width_ratios=[1.1, 1], wspace=0.28)

# LEFT PANEL: Weight functions w(p)
ax_left = fig.add_subplot(gs[0, 0])
p = np.linspace(0.001, 0.999, 500)

ax_left.plot(p, np.ones_like(p), label="Unweighted", color=COLORS["unweighted"],
             linestyle="--", linewidth=1.6, alpha=0.7)
ax_left.plot(p, w_crump(p), label="Crump", color=COLORS["crump"],
             linestyle=":", linewidth=2.2)
ax_left.plot(p, w_overlap(p), label="Overlap", color=COLORS["overlap"],
             linewidth=2.0)
ax_left.plot(p, w_riw_source(p, 1.0, 0.5), label="RIW Forward", color=COLORS["source"],
             linewidth=2.0)
ax_left.plot(p, w_riw_target(p, 1.0, 0.5), label="RIW Inverse", color=COLORS["target"],
             linewidth=2.0)
w_doubly = w_riw_source(p, 1.0, 0.5) * w_riw_target(p, 1.0, 0.5)
w_doubly = w_doubly / np.max(w_doubly)
ax_left.plot(p, w_doubly, label="RIW Doubly", color=COLORS["doubly"],
             linewidth=2.3, zorder=5)

# Annotation: key difference
ax_left.annotate("Overlap:\nsymmetric",
                 xy=(0.5, 0.25), xytext=(0.68, 0.12),
                 arrowprops=dict(arrowstyle="->", color=COLORS["overlap"], lw=1),
                 fontsize=7, color=COLORS["overlap"], ha="left",
                 bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
                          edgecolor=COLORS["overlap"], linewidth=0.6))
ax_left.annotate("RIW:\nasymmetric",
                 xy=(0.15, w_riw_source(0.15, 1.0, 0.5)), xytext=(0.08, 0.65),
                 arrowprops=dict(arrowstyle="->", color=COLORS["source"], lw=1),
                 fontsize=7, color=COLORS["source"], ha="left",
                 bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
                          edgecolor=COLORS["source"], linewidth=0.6))

ax_left.set_xlabel("Domain Probability p")
ax_left.set_ylabel("Relative Weight")
ax_left.set_xlim(0, 1)
ax_left.set_ylim(-0.05, 1.1)
ax_left.legend(loc="upper left", ncol=1, frameon=True, fancybox=False,
               edgecolor="#DDDDDD", framealpha=0.95, fontsize=7)
ax_left.set_title("(a) Weight Functions", loc="left", fontsize=10)

# RIGHT PANEL: Concrete example with densities
ax_right = fig.add_subplot(gs[0, 1])

x_range = np.linspace(-4, 3, 300)
kde_source = stats.gaussian_kde(X_source)
kde_target = stats.gaussian_kde(X_target)

# Plot densities
ax_right.fill_between(x_range, kde_source(x_range), alpha=0.3,
                       color=COLORS["source"], label="Source")
ax_right.fill_between(x_range, kde_target(x_range), alpha=0.3,
                       color=COLORS["target"], label="Target")
ax_right.plot(x_range, kde_source(x_range), color=COLORS["source"], linewidth=1.8)
ax_right.plot(x_range, kde_target(x_range), color=COLORS["target"], linewidth=1.8)

# Show RIW doubly weighted effect via scatter opacity
ax_right.scatter(X_source, np.zeros_like(X_source) + 0.42, s=2.5,
                c=COLORS["source"], alpha=w_s_riw * 0.8, rasterized=True)
ax_right.scatter(X_target, np.zeros_like(X_target) + 0.39, s=2.5,
                c=COLORS["target"], alpha=w_t_riw * 0.8, rasterized=True)

# Shade the common-support region
common_support_mask = (X_source > -2.5) & (X_source < 2)
cs_x = np.linspace(-2.5, 2, 100)
ax_right.fill_between(cs_x, 0, 0.5, alpha=0.08, color=COLORS["doubly"],
                       label="Common support\n(high effective weight)")

# Annotations for weak-overlap regions
ax_right.annotate("Weak overlap\n(downweighted)", xy=(-3, 0.25), xytext=(-3.2, 0.35),
                  fontsize=7, color="#666666", ha="center",
                  arrowprops=dict(arrowstyle="->", color="#999999", lw=0.8))
ax_right.annotate("", xy=(2.2, 0.15), xytext=(2.5, 0.28),
                  arrowprops=dict(arrowstyle="->", color="#999999", lw=0.8))

ax_right.set_xlabel("Feature x")
ax_right.set_ylabel("Density")
ax_right.set_ylim(0, 0.48)
ax_right.set_xlim(-4, 3)
ax_right.legend(loc="upper left", fontsize=6.5, frameon=True, fancybox=False,
                edgecolor="#DDDDDD", framealpha=0.95)
ax_right.set_title("(b) Example: Two-Sided Mismatch", loc="left", fontsize=10)
ax_right.grid(False)

plt.tight_layout()
fig.savefig("conceptual_option_c.pdf")
fig.savefig("conceptual_option_c.png", dpi=300)
print("Generated: conceptual_option_c.pdf and .png")
