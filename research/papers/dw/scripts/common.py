"""Shared utilities for domain-weighting simulation experiments with samesame.

Mapping to samesame terms (see CONTEXT.md):
- source = reference population, target = population under evaluation
- S = severity score tested with test_harm(worse="higher")
- feature = domain feature; a feature-only domain classifier gives
  P(target|feature), which feeds domain_weights
  -> test_harm(..., weights=...).
- Decision rule: s-value scale with ROPE |deltas| <= 1 (D-SOS Kamulete 2022,
  S6; Benavoli et al. 2014 prior = one pseudo-experiment at delta = 0).
"""

from __future__ import annotations

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict

import samesame as ss

CLASSIFIERS = ("logreg", "rf_cal", "hgb_cal")
REWEIGHTS = ("source", "target", "both")
SHRINKAGES = (0.25, 0.5, 0.75)

PRIMARY = {"classifier": "hgb_cal", "reweight": "both", "shrinkage": 0.5}


def s_value(p: np.ndarray | float) -> np.ndarray | float:
    """s = -log2(p). Permutation p-values are > 0 by construction."""
    return -np.log2(np.asarray(p, dtype=float))


# ---------------------------------------------------------------------------
# Overlap DGP: 1-D feature with low-overlap regions (port of
# draw_overlap_dataset from the earlier manuscript suite). Shared N(0,1) in
# both groups, plus a source-only low-overlap region at -LOW_OVERLAP_LOC and
# a target-only low-overlap region at +LOW_OVERLAP_LOC; severity score =
# feature + noise. effect adds a common-support shift to shared target
# scores (0.0 = support-shift condition: no harmful change, so the
# unweighted test gives support-driven rejections as the off-support
# fraction grows while the doubly weighted test remains calibrated under
# the support-shift null).
# ---------------------------------------------------------------------------

LOW_OVERLAP_LOC = 3.0
LOW_OVERLAP_SD = 0.45
OVERLAP_SCORE_SD = 0.8


def gen_overlap(
    rng: np.random.Generator,
    n_source: int,
    n_target: int,
    off_support_fraction: float = 0.25,
    effect: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (source_score, target_score, source_feature, target_feature).

    off_support_fraction: fraction of each group in its off-support
    low-overlap region (0.0 = identical N(0,1) populations).
    effect: magnitude of the common-support shift added to shared target
    scores (0.0 = support-shift condition).
    """
    source_low_overlap = rng.random(n_source) < off_support_fraction
    target_low_overlap = rng.random(n_target) < off_support_fraction

    source_feature = rng.normal(0.0, 1.0, size=n_source)
    target_feature = rng.normal(0.0, 1.0, size=n_target)
    source_feature[source_low_overlap] = rng.normal(
        -LOW_OVERLAP_LOC, LOW_OVERLAP_SD, size=source_low_overlap.sum()
    )
    target_feature[target_low_overlap] = rng.normal(
        LOW_OVERLAP_LOC, LOW_OVERLAP_SD, size=target_low_overlap.sum()
    )

    source_score = source_feature + rng.normal(
        0.0, OVERLAP_SCORE_SD, size=n_source)
    target_score = target_feature + rng.normal(
        0.0, OVERLAP_SCORE_SD, size=n_target)
    target_score[~target_low_overlap] += effect
    return source_score, target_score, source_feature, target_feature

# ---------------------------------------------------------------------------
# Domain probabilities: C-only classifier, out-of-sample via CV
# ---------------------------------------------------------------------------

def _base_estimator(name: str, seed: int):
    if name == "logreg":
        return LogisticRegression(max_iter=1000)
    if name == "rf_cal":
        return CalibratedClassifierCV(
            RandomForestClassifier(n_estimators=200, random_state=seed),
            cv=10,
            method="sigmoid",
        )
    if name == "hgb_cal":
        return HistGradientBoostingClassifier(random_state=seed)
    raise ValueError(f"unknown classifier={name!r}")


def domain_probs(
    source_feature: np.ndarray, target_feature: np.ndarray, method: str, seed: int
) -> tuple[np.ndarray, np.ndarray, float]:
    """Out-of-sample P(target|feature) via cross_val_predict(cv=10).

    Returns (source_prob, target_prob, domain_auc).
    """
    from sklearn.metrics import roc_auc_score

    c = np.concatenate([source_feature, target_feature]).reshape(-1, 1)
    y = np.concatenate(
        [np.zeros(len(source_feature)), np.ones(len(target_feature))]
    ).astype(int)
    if method == "hgb_cal":
        est = CalibratedClassifierCV(
            _base_estimator("hgb_cal", seed), cv=10, method="sigmoid"
        )
    else:
        est = _base_estimator(method, seed)
    proba = cross_val_predict(est, c, y, cv=10, method="predict_proba")[:, 1]
    auc = float(roc_auc_score(y, proba))
    return proba[: len(source_feature)], proba[len(source_feature):], auc


def delta_s(p_weighted: float, p_unweighted: float) -> float:
    """Paired s-value difference: s_w - s_u (positive = weights more extreme)."""
    return float(s_value(p_weighted) - s_value(p_unweighted))


def rope_posterior(
    deltas: np.ndarray, rope: float = 1.0, n_draws: int = 20000, seed: int = 0
) -> dict:
    """Dirichlet-multinomial posterior over {better, equivalent, worse}.

    Counts wins/losses with ROPE |delta| <= 1; prior = one pseudo
    observation at delta = 0 (equivalent), matching D-SOS S6 description.
    'better' here = weighted s-value larger by > rope (more evidence of
    harm); sign flips meaning under the null where smaller s is desired —
    callers should interpret with the null/alt context.
    """
    rng = np.random.default_rng(seed)
    d = np.asarray(deltas, dtype=float)
    n_better = int(np.sum(d > rope))
    n_equiv = int(np.sum(np.abs(d) <= rope)) + 1  # +1 pseudo-experiment
    n_worse = int(np.sum(d < -rope))
    alpha = np.array([n_better, n_equiv, n_worse], dtype=float) + 1.0
    draws = rng.dirichlet(alpha, size=n_draws)
    post_mean = alpha / alpha.sum()
    # Report posterior means + P(weighted better) via sampling:
    p_weighted_better = float(np.mean(draws[:, 0] > draws[:, 2]))
    p_equiv = float(np.mean(
        (np.abs(draws[:, 0] - draws[:, 2]) < 0.05) |
        ((draws[:, 1] > draws[:, 0]) & (draws[:, 1] > draws[:, 2]))
    ))
    return {
        "n_better": n_better,
        "n_equiv": n_equiv,
        "n_worse": n_worse,
        "post_mean_better": float(post_mean[0]),
        "post_mean_equiv": float(post_mean[1]),
        "post_mean_worse": float(post_mean[2]),
        "p_weighted_better_vs_worse": p_weighted_better,
        "equiv_rate": float(np.sum(np.abs(d) <= rope) / max(len(d), 1)),
        "median_delta_s": float(np.median(d)) if len(d) else float("nan"),
    }
