# Weight for common support

Start with an unweighted comparison. Reach for weighting only when poor feature overlap is a real concern. Weighting reframes the question around **common support** — the regions represented by both groups. It does not create information where the groups do not overlap, and it changes the population the test describes. It is not a default correction.

!!! note "Prerequisites"
    You have run an unweighted comparison and seen a shift, and you can estimate honest `P(target|x)` out of sample (for example with `cross_val_predict` or `oob_decision_function_`). Keep the domain probability separate from the interpretable score you test for harm. If you are new to `samesame`, start with [Get started](../tutorials/get-started.md) and [Monitor a credit model](../credit/monitor-credit.md) first.

## Why weight?

A handful of low-overlap observations can dominate an unweighted permutation test, even when the region you care about has changed. Weighting reduces their influence and focuses the comparison on the narrower question of whether the shift persists where source and target overlap.

> Training includes many 20-year-old students that production never sees, while production includes retirees that training never saw. An unweighted comparison is pulled by the extremes; a weighted comparison centers on the overlapping 30–60 range.

A domain classifier estimates `p̂(x) = P(target|x)`. After adjusting for the sample sizes, the odds correction `p̂/(1−p̂) · n_source/n_target` (Bickel et al., 2007) estimates the relative density and produces importance weights. This correction can be unstable when the groups separate well, because a few observations may receive very large weights. `samesame` stabilizes the result with a shrinkage parameter `λ` that pulls weights toward uniform. The default is `0.5` (Yamada et al., 2013):

--8<-- "snippets/shrinkage-table.txt"

**Guideline:** Use the domain probability to assess how observations differ between source and target and to build weights. Use a separate score with a clear sense of better and worse for the harm test. Domain probability describes group membership, not outcome quality.

--8<-- "snippets/reweight-table.txt"

## Synthetic check (no download)

```python
import numpy as np
from sklearn.datasets import make_classification
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import cross_val_predict
import samesame as ss

X, group = make_classification(n_samples=200, n_features=6, n_classes=2, random_state=12345)
domain_prob = cross_val_predict(
    HistGradientBoostingClassifier(random_state=12345),
    X, group, cv=10, method="predict_proba",
)[:, 1]  # pooled P(target|x) — for weighting only

rng = np.random.default_rng(12345)
risk_score = 0.9*X[:,0] - 0.6*X[:,1] + 0.4*X[:,2] + rng.normal(scale=0.4, size=len(group))
source_scores, target_scores = risk_score[group == 0], risk_score[group == 1]

source_prob, target_prob = domain_prob[group == 0], domain_prob[group == 1]
weights = ss.domain_weights(source=source_prob, target=target_prob, reweight="source", shrinkage=0.5)

rng = np.random.default_rng(12345)
unweighted = ss.test_harmful_shift(source=source_scores, target=target_scores, worse="higher", rng=rng)
rng = np.random.default_rng(12345)
weighted = ss.test_harmful_shift(source=source_scores, target=target_scores, worse="higher", weights=weights, rng=rng)
print(f"Unweighted p={unweighted.pvalue:.4f} Weighted p={weighted.pvalue:.4f}")  # → 1.0, 0.62
```

Neither test finds a shift in this synthetic draw; the snippet is meant to show the mechanics. The HELOC example below contrasts modes on real data.

Use the same `worse` value for the unweighted and weighted comparisons so that both test the same harmful direction. If a strong unweighted result weakens after weighting, the evidence was concentrated in low-overlap regions. If the result persists, the harmful shift is also present where source and target overlap.

## HELOC example

This example uses the same source and target split as [Monitor a credit model](../credit/monitor-credit.md): the lender that trained in calm conditions and was then evaluated in a more adverse mix. There, the risk alarm fired. Here we ask what remains when the comparison is reweighted: did comparable applicants fare worse, or did incomparable ones arrive?

```python
import samesame as ss

--8<-- "snippets/heloc-split.py:heloc-split"
--8<-- "snippets/heloc-split.py:heloc-domain"
--8<-- "snippets/heloc-split.py:heloc-risk-model"

source_prob = domain_prob[split.values == 0]
target_prob = domain_prob[split.values == 1]

unweighted = ss.test_harmful_shift(source=train_risk, target=deployment_risk, worse="higher", rng=np.random.default_rng(12345))
w_src = ss.domain_weights(source=source_prob, target=target_prob, reweight="source", shrinkage=0.5)
w_both = ss.domain_weights(source=source_prob, target=target_prob, reweight="both", shrinkage=0.5)
p_src = ss.test_harmful_shift(source=train_risk, target=deployment_risk, worse="higher", weights=w_src, rng=np.random.default_rng(12345))
p_both = ss.test_harmful_shift(source=train_risk, target=deployment_risk, worse="higher", weights=w_both, rng=np.random.default_rng(12345))
print(f"Unweighted p={unweighted.pvalue:.4f} Source p={p_src.pvalue:.4f} Both p={p_both.pvalue:.4f}")  # → all 0.0001 — persists on common support
```

The alarm survives reweighting (all p = 0.0001), and the effective sample sizes put that finding in context: the source side retains an effective 2,491 of 7,683 applications (ESS/n ≈ 0.32) and the target side 1,532 of 2,188 (≈ 0.70). The harmful shift is present where the two populations overlap, not only in low-overlap regions.

??? details "What a stricter protocol finds (research)"

    A companion research study on common-support weighting (under review, 2026) repeats this HELOC comparison under a stricter protocol: the split variable is removed from the features, the domain classifier is cross-validated, and the test runs on held-out subsamples (1,536 source and 2,776 target applications; 499 permutations; `shrinkage=0.5`). Under that protocol, the three standard modes — unweighted, source-weighted, and target-weighted — all reject at the permutation minimum (p = 0.002), but the doubly weighted test does not (p = 0.376; ESS 401 of 1,536 source, 803 of 2,776 target). Most of each population lies outside the common support, so the finding is driven by applicant profiles from low-overlap regions. The pattern suggests context change rather than harm on comparable applicants. The protocols differ and that is the point: compare modes under your own protocol before relying on any single p-value.

??? details "Diagnose weight concentration (ESS)"

    Weighting can concentrate the comparison on a small number of observations. Check the effective sample size (ESS) to judge whether the weighted result rests on enough information.

    ```python
    rng = np.random.default_rng(7)
    source_prob_demo = rng.beta(a=2, b=5, size=400)
    target_prob_demo = rng.beta(a=5, b=2, size=400)
    source_prob_demo[:8] = rng.uniform(0.97, 0.999, size=8)

    weights = ss.domain_weights(source=source_prob_demo, target=target_prob_demo, reweight="both", shrinkage=0.0)
    ess = weights.effective_sample_size()
    print(f"source ESS {ess.source:.1f} / 400")  # → 6.5 — concentrated
    ```

    Lower `shrinkage` gives a stronger correction with higher-variance weights; higher `shrinkage` gives more stable, more uniform weights. The default `shrinkage=0.5` balances correction strength and stability.

    ```python
    for lam in [0.0, 0.25, 0.5, 0.75, 1.0]:
        w = ss.domain_weights(source=source_prob_demo, target=target_prob_demo, reweight="both", shrinkage=lam)
        e = w.effective_sample_size()
        print(f"shrinkage={lam:<4}  source ESS={e.source:7.2f}  target ESS={e.target:7.2f}")
    ```

    Compare each ESS to its `n` through `ESS/n`. A low ratio — for example well below `0.5` — signals that the weighted result is fragile and driven by a few observations. This is a warning, not a hard validation rule, and there is no universal cutoff from Kish. The often-quoted `ESS < n/4` is a rough illustrative heuristic with no published empirical threshold (see Elvira et al., 2022). If `ESS/n` stays low even at `shrinkage=0.5`, the groups may lack enough common support for a reliable weighted comparison; consider keeping the comparison unweighted. See [Effective sample size](../../api/weighting.md#effective-sample-size) for details.

Full scripts: `examples/weighting/_code/` · Reference: [Importance weights](../../api/weighting.md).
