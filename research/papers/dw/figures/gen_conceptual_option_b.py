#!/usr/bin/env python3
"""Generate Option B: Density-over-x with weight overlays."""
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from sklearn.linear_model import LogisticRegression

# --- Publication defaults ---
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold",
    "axes.labelsize": 10, "legend.fontsize": 8, "legend.frameon": False,
    "figure.dpi": 300, "savefig.dpi": 300, "savefig.bbox": "tight",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": False,
    "lines.linewidth": 2.2,
})

# --- Color palette ---
COLORS = {
    "source": "#2A9D8F",
    "target": "#E76F51",
    "crump": "#8C8C8C",
    "overlap": "#E9C46A",
    "riw": "#264653",
}

# --- Generate synthetic data (mismatched distributions) ---
np.random.seed(42)
n_source, n_target = 800, 1200

# Source: standard Gaussian
X_source = np.random.randn(n_source)
# Target: mixture of two Gaussians (shifted, creating weak overlap regions)
X_target = np.concatenate([
    np.random.randn(int(0.6 * n_target)) - 1.5,  # left cluster
    np.random.randn(int(0.4 * n_target)) + 0.5,  # right cluster
])

# Fit domain classifier to get domain probabilities
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

# Compute weights
prior_ratio = n_source / n_target
w_s_crump = w_crump(p_source)
w_t_crump = w_crump(p_target)
w_s_overlap = w_overlap(p_source)
w_t_overlap = w_overlap(p_target)
w_s_riw = w_riw_source(p_source, prior_ratio)
w_t_riw = w_riw_target(p_target, prior_ratio)

# Normalize weights to [0, 1] for use as alpha values
w_s_overlap = w_s_overlap / np.max(w_s_overlap) if np.max(w_s_overlap) > 0 else w_s_overlap
w_s_riw = w_s_riw / np.max(w_s_riw) if np.max(w_s_riw) > 0 else w_s_riw
w_t_riw = w_t_riw / np.max(w_t_riw) if np.max(w_t_riw) > 0 else w_t_riw

# --- Create multi-panel figure ---
fig, axes = plt.subplots(2, 3, figsize=(6.75, 4.5), sharex=True)

x_range = np.linspace(-4, 3, 300)

# Panel 1: Unweighted (densities only)
ax = axes[0, 0]
kde_source = stats.gaussian_kde(X_source)
kde_target = stats.gaussian_kde(X_target)
ax.fill_between(x_range, kde_source(x_range), alpha=0.4, color=COLORS["source"], label="Source")
ax.fill_between(x_range, kde_target(x_range), alpha=0.4, color=COLORS["target"], label="Target")
ax.plot(x_range, kde_source(x_range), color=COLORS["source"], linewidth=2)
ax.plot(x_range, kde_target(x_range), color=COLORS["target"], linewidth=2)
ax.set_ylabel("Density")
ax.set_title("Unweighted\n(full populations)")
ax.legend(loc="upper left", fontsize=7)
ax.set_ylim(0, 0.5)

# Panel 2: Crump trimming
ax = axes[0, 1]
ax.fill_between(x_range, kde_source(x_range), alpha=0.4, color=COLORS["source"])
ax.fill_between(x_range, kde_target(x_range), alpha=0.4, color=COLORS["target"])
# Shade the trimmed regions
for xi, wi in zip(X_source, w_s_crump):
    if wi == 0:
        ax.axvline(xi, color=COLORS["crump"], alpha=0.02, linewidth=0.5)
for xi, wi in zip(X_target, w_t_crump):
    if wi == 0:
        ax.axvline(xi, color=COLORS["crump"], alpha=0.02, linewidth=0.5)
ax.plot(x_range, kde_source(x_range), color=COLORS["source"], linewidth=2)
ax.plot(x_range, kde_target(x_range), color=COLORS["target"], linewidth=2)
ax.set_title("Crump Trimming\n(binary mask, min(p,1-p)≥0.1)")
ax.set_ylim(0, 0.5)

# Panel 3: Overlap weights
ax = axes[0, 2]
ax.fill_between(x_range, kde_source(x_range), alpha=0.4, color=COLORS["source"])
ax.fill_between(x_range, kde_target(x_range), alpha=0.4, color=COLORS["target"])
# Show weight attenuation by scatter opacity
ax.scatter(X_source, np.zeros_like(X_source) + 0.45, s=3, c=COLORS["overlap"],
           alpha=w_s_overlap, label="Overlap weight")
ax.plot(x_range, kde_source(x_range), color=COLORS["source"], linewidth=2)
ax.plot(x_range, kde_target(x_range), color=COLORS["target"], linewidth=2)
ax.set_title("Overlap Weights\n(symmetric, p(1-p))")
ax.set_ylim(0, 0.5)

# Panel 4: RIW Source-weighted
ax = axes[1, 0]
ax.fill_between(x_range, kde_source(x_range), alpha=0.4, color=COLORS["source"])
ax.fill_between(x_range, kde_target(x_range), alpha=0.4, color=COLORS["target"])
ax.scatter(X_source, np.zeros_like(X_source) + 0.45, s=3, c=COLORS["riw"],
           alpha=w_s_riw)
ax.plot(x_range, kde_source(x_range), color=COLORS["source"], linewidth=2)
ax.plot(x_range, kde_target(x_range), color=COLORS["target"], linewidth=2)
ax.set_xlabel("Feature x")
ax.set_ylabel("Density")
ax.set_title("RIW Source-weighted\n(forward correction)")
ax.set_ylim(0, 0.5)

# Panel 5: RIW Target-weighted
ax = axes[1, 1]
ax.fill_between(x_range, kde_source(x_range), alpha=0.4, color=COLORS["source"])
ax.fill_between(x_range, kde_target(x_range), alpha=0.4, color=COLORS["target"])
ax.scatter(X_target, np.zeros_like(X_target) + 0.45, s=3, c=COLORS["riw"],
           alpha=w_t_riw)
ax.plot(x_range, kde_source(x_range), color=COLORS["source"], linewidth=2)
ax.plot(x_range, kde_target(x_range), color=COLORS["target"], linewidth=2)
ax.set_xlabel("Feature x")
ax.set_title("RIW Target-weighted\n(inverse correction)")
ax.set_ylim(0, 0.5)

# Panel 6: RIW Doubly weighted
ax = axes[1, 2]
ax.fill_between(x_range, kde_source(x_range), alpha=0.4, color=COLORS["source"])
ax.fill_between(x_range, kde_target(x_range), alpha=0.4, color=COLORS["target"])
ax.scatter(X_source, np.zeros_like(X_source) + 0.45, s=3, c=COLORS["source"],
           alpha=w_s_riw * 0.7)
ax.scatter(X_target, np.zeros_like(X_target) + 0.42, s=3, c=COLORS["target"],
           alpha=w_t_riw * 0.7)
ax.plot(x_range, kde_source(x_range), color=COLORS["source"], linewidth=2)
ax.plot(x_range, kde_target(x_range), color=COLORS["target"], linewidth=2)
ax.set_xlabel("Feature x")
ax.set_title("RIW Doubly weighted\n(both corrections)")
ax.set_ylim(0, 0.5)

plt.tight_layout()
fig.savefig("conceptual_option_b.pdf")
fig.savefig("conceptual_option_b.png", dpi=300)
print("Generated: conceptual_option_b.pdf and .png")
