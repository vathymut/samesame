# Weight for common support

Start with an unweighted comparison. Use weighting only when poor feature
overlap is a real concern. Weighting reframes the comparison around common
support (the regions represented by both groups). It does not create
information where the groups do not overlap, and it changes the population
the test describes. It is not a default correction.

## Why weight?

A few low-overlap observations can dominate the permutation test, even if the
region you care about has genuinely changed. Weighting reduces their influence
and focuses the test on the narrower question of whether the shift is present
where source and target overlap.

> Training has many 20-year-old students production never sees; production has retirees never seen in training. Unweighted is swayed by extremes; weighted focuses on the 30–60 overlap.

A domain classifier estimates `p̂(x)=P(target|x)`. After accounting for the
source and target sample sizes, the odds correction
`p̂/(1-p̂)·n_source/n_target` (Bickel et al., 2007) estimates relative density
and produces importance weights. This correction can be unstable when the
groups separate well, because a few observations may receive huge weights.
`samesame` stabilizes the weights with shrinkage `λ`, which blends them toward
uniform weights (default `0.5`, Yamada et al., 2013):

--8<-- "snippets/shrinkage-table.txt"

**Rule:** Use the domain probability to measure how observations differ between
source and target, and to build weights. Use a separate score with a clear
good/bad interpretation for the harm test. A domain probability describes
distributional membership, not outcome quality.

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

Neither test finds a shift in this synthetic draw; the example shows the
plumbing. The HELOC case below shows the contrast between modes on real data.

Use the same `worse` value for both comparisons so that they test the same
harmful direction. If a strong unweighted result becomes weak after weighting,
the evidence was concentrated in low-overlap regions. If the result persists,
the harmful shift is also present in the common-support region.

## HELOC example

This example uses the same source and target split as [Monitor a credit
model](../credit/monitor-credit.md): the lender that trained on calm seas
and sailed into a storm. There, the risk alarm fired. Here we ask the
question it leaves open: did comparable applicants get worse, or did
incomparable ones simply arrive?

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

The alarm survives reweighting (all p = 0.0001), and the effective sample
sizes say how much water that carries: the source side of the comparison
rests on an effective 2,491 of 7,683 applications (ESS/n ≈ 0.32), the target
side on 1,532 of 2,188 (≈ 0.70). The harmful shift is present on common
support too, not a low-overlap artifact.

??? details "What a stricter protocol finds (research)"

    A companion research study on common-support weighting (under review,
    2026) re-runs this HELOC comparison under a stricter protocol: the split
    variable is dropped from the features, the domain classifier is
    cross-validated, and the test runs on held-out subsamples (1,536 source
    and 2,776 target applications; 499 permutations, `shrinkage=0.5`).
    Under that protocol, the three standard modes (unweighted,
    source-weighted, target-weighted) all reject at the permutation minimum
    (p = 0.002), but the doubly weighted test does not (p = 0.376; ESS 401 of
    1,536 source, 803 of 2,776 target). The verdict flips because most of
    each population lies outside the common support. The alarm was carried by
    applicant profiles from low-overlap regions: context change, not model
    harm on comparable applicants. The protocols differ, and that is the
    lesson: check the contrast between modes on *your* protocol before
    trusting any single p-value.

??? details "Diagnose weight concentration (ESS)"

    Weighting can concentrate the comparison on a few observations. Check the effective sample size (ESS) to see whether the weighted result is supported by enough information.

    ```python
    rng = np.random.default_rng(7)
    source_prob_demo = rng.beta(a=2, b=5, size=400)
    target_prob_demo = rng.beta(a=5, b=2, size=400)
    source_prob_demo[:8] = rng.uniform(0.97, 0.999, size=8)

    weights = ss.domain_weights(source=source_prob_demo, target=target_prob_demo, reweight="both", shrinkage=0.0)
    ess = weights.effective_sample_size()
    print(f"source ESS {ess.source:.1f} / 400")  # → 6.5 — concentrated
    ```

    Lower `shrinkage` gives a stronger correction but higher-variance weights;
    higher `shrinkage` gives more stable, more uniform weights. The default
    `shrinkage=0.5` balances correction strength and stability.

    ```python
    for lam in [0.0, 0.25, 0.5, 0.75, 1.0]:
        w = ss.domain_weights(source=source_prob_demo, target=target_prob_demo, reweight="both", shrinkage=lam)
        e = w.effective_sample_size()
        print(f"shrinkage={lam:<4}  source ESS={e.source:7.2f}  target ESS={e.target:7.2f}")
    ```

    Compare each ESS to its ``n`` via ``ESS/n``: a low ratio warns that a few
    observations dominate the weighted result. What counts as "low", the
    caveats, and why there is no universal cutoff: see [Effective sample
    size](../../api/weighting.md#effective-sample-size) in the reference.

Full scripts: `examples/weighting/_code/` · Reference: [Importance weights](../../api/weighting.md).
