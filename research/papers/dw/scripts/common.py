"""Shared utilities for Cobb et al. (2022) 4.1/4.2 replications with samesame.

Mapping to samesame terms (see CONTEXT.md):
- source = reference (original context), target = deployment (changed context)
- S = univariate outlier score tested with test_harm(worse="higher")
- C = domain context; a C-only domain classifier gives P(target|C),
  which feeds domain_weights -> test_harm(..., weights=...).
- Decision rule: s-value scale with ROPE |deltas| <= 1 (D-SOS Kamulete 2022,
  S6; Benavoli et al. 2014 prior = one pseudo-experiment at delta = 0).
"""

from __future__ import annotations

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.mixture import GaussianMixture
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
# 4.1 data generation: S|C ~ N(C,1), C0 ~ N(0,1)
# ---------------------------------------------------------------------------

K2_MUS = (-0.8, 0.8)
K2_SD = 0.2


def gen_41(
    rng: np.random.Generator,
    n_source: int,
    n_target: int,
    c1: str = "sigma_0.5",
    alt_eps: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (source_S, target_S, source_C, target_C) for experiment 4.1.

    c1: 'sigma_0.25' | 'sigma_0.5' | 'sigma_1.0' | 'k2'.
    alt_eps: mean shift added to target S (0.0 = null, 0.5 = alt per plan).
    For 'k2' the shift applies to one mixture mode only (the upper one).
    """
    source_c = rng.normal(0.0, 1.0, size=n_source)
    source_s = rng.normal(source_c, 1.0)

    if c1.startswith("sigma_"):
        sigma = float(c1.split("_")[1])
        target_c = rng.normal(0.0, sigma, size=n_target)
        target_s = rng.normal(target_c + alt_eps, 1.0)
    elif c1 == "k2":
        mus = np.array(K2_MUS)
        comp = rng.integers(0, 2, size=n_target)
        target_c = rng.normal(mus[comp], K2_SD)
        shift = np.where(comp == 1, alt_eps, 0.0)
        target_s = rng.normal(target_c + shift, 1.0)
    else:
        raise ValueError(f"unknown c1={c1!r}")
    return source_s, target_s, source_c, target_c


# ---------------------------------------------------------------------------
# 4.2 data generation: 2-D mixture, prevalence shift + mean shift
# ---------------------------------------------------------------------------

MU1 = np.array([-1.0, 0.0])
MU2 = np.array([1.0, 0.0])
SIGMA_42 = 0.5
SHIFT_42 = np.array([0.6, 0.0])


def gen_42(
    rng: np.random.Generator,
    n_source: int,
    n_target: int,
    pi1: float = 0.2,
    alt: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (source_S, target_S, source_C, target_C, source_X, target_X).

    S (harm score) = X[:, 0] so the +[0.6,0] mean shift is worse="higher".
    C = P(comp1 | X) from a 2-comp GMM fit on source X (paper fits on
    held-out ref; here per-run source fit for self-containment).
    """
    src_comp = rng.random(n_source) < 0.5
    source_x = np.where(
        src_comp[:, None],
        rng.normal(MU1, SIGMA_42, size=(n_source, 2)),
        rng.normal(MU2, SIGMA_42, size=(n_source, 2)),
    )
    tgt_comp = rng.random(n_target) < pi1
    mu2 = MU2 + (SHIFT_42 if alt else 0.0)
    target_x = np.where(
        tgt_comp[:, None],
        rng.normal(MU1, SIGMA_42, size=(n_target, 2)),
        rng.normal(mu2, SIGMA_42, size=(n_target, 2)),
    )
    gm = GaussianMixture(n_components=2, n_init=3, random_state=int(rng.integers(1e9)))
    gm.fit(source_x)
    # Align component 0 with MU1 (closest mean in x) for stable C definition.
    order = np.argsort(gm.means_[:, 0])
    probs_src = gm.predict_proba(source_x)[:, order[0]]
    probs_tgt = gm.predict_proba(target_x)[:, order[0]]
    return (
        source_x[:, 0],
        target_x[:, 0],
        probs_src,
        probs_tgt,
        source_x,
        target_x,
    )


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
    source_c: np.ndarray, target_c: np.ndarray, method: str, seed: int
) -> tuple[np.ndarray, np.ndarray, float]:
    """Out-of-sample P(target|C) via cross_val_predict(cv=10).

    Returns (source_prob, target_prob, domain_auc).
    """
    from sklearn.metrics import roc_auc_score

    c = np.concatenate([source_c, target_c]).reshape(-1, 1)
    y = np.concatenate([np.zeros(len(source_c)), np.ones(len(target_c))]).astype(int)
    if method == "hgb_cal":
        est = CalibratedClassifierCV(
            _base_estimator("hgb_cal", seed), cv=10, method="sigmoid"
        )
    else:
        est = _base_estimator(method, seed)
    proba = cross_val_predict(est, c, y, cv=10, method="predict_proba")[:, 1]
    auc = float(roc_auc_score(y, proba))
    return proba[: len(source_c)], proba[len(source_c):], auc


# ---------------------------------------------------------------------------
# One paired run: unweighted vs weighted test_harm
# ---------------------------------------------------------------------------

def paired_run(
    source_s: np.ndarray,
    target_s: np.ndarray,
    source_c: np.ndarray,
    target_c: np.ndarray,
    *,
    n_resamples: int,
    seed: int,
    classifiers: tuple[str, ...] = CLASSIFIERS,
    reweights: tuple[str, ...] = REWEIGHTS,
    shrinkages: tuple[float, ...] = SHRINKAGES,
) -> dict:
    """Run unweighted + full sweep of weighted test_harm on one dataset."""
    out: dict = {}
    r = ss.test_harm(
        source_s, target_s, worse="higher",
        n_resamples=n_resamples, rng=np.random.default_rng(seed),
    )
    out["unweighted"] = {"p": float(r.pvalue), "stat": float(r.statistic)}
    for clf in classifiers:
        sp, tp, auc = domain_probs(source_c, target_c, clf, seed)
        out[f"domain_auc/{clf}"] = auc
        for rw in reweights:
            for lam in shrinkages:
                w = ss.domain_weights(
                    source=sp, target=tp, reweight=rw, shrinkage=lam
                )
                ess = w.effective_sample_size()
                rr = ss.test_harm(
                    source_s, target_s, worse="higher", weights=w,
                    n_resamples=n_resamples,
                    rng=np.random.default_rng(seed),
                )
                out[f"{clf}/{rw}/{lam}"] = {
                    "p": float(rr.pvalue),
                    "stat": float(rr.statistic),
                    "ess_source": float(ess.source),
                    "ess_target": float(ess.target),
                }
    return out


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
