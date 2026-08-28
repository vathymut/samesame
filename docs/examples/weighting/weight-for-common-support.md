# Weight for common support

Start unweighted, then compare. Weighting is for a known overlap problem, not
a default setting: it changes the target population of the test to the region
where source and target both have support.

## Why weight?

Even with a real change in the region you care about, points the other group
almost never visits can dominate a permutation test. The weighted result is
therefore not "the corrected answer" in all contexts; it answers a different,
common-support question.

> Training has many 20-year-old students production never sees; production has retirees never seen in training. Unweighted is swayed by extremes; weighted focuses on the 30–60 overlap.

A domain classifier gives `p̂(x)=P(target|x)`. Its odds estimate how much more
target-like an observation is than source-like. The plain correction
`p̂/(1-p̂)·n_source/n_target` is powerful but unstable: a few points get huge
weights when groups separate well. `samesame` stabilises this with shrinkage
λ blending toward uniform (default `0.5`):

--8<-- "snippets/shrinkage-table.txt"

**Rule:** domain probabilities are for *weighting* - do not reuse the same
`P(target|x)` as the harm score. A domain probability measures group
membership, not harm.

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

Use the same `worse` for both comparisons. If a strong unweighted result
becomes weak after weighting, the evidence was concentrated in low-overlap
regions. If it persists, the harmful shift is also present in common support.

## HELOC example

Same split as [Monitor a credit model](../credit/monitor-credit.md).

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

Use `source` when source has low-overlap cases and you want to represent the
target region. Use `target` for the reverse. Use `both` when both groups have
low-overlap regions and the intended comparison is their mutual support.

??? details "Diagnose weight concentration (ESS)"

    Weighting can pile mass on a few points. Check effective sample size before trusting:

    ```python
    rng = np.random.default_rng(7)
    source_prob_demo = rng.beta(a=2, b=5, size=400)
    target_prob_demo = rng.beta(a=5, b=2, size=400)
    source_prob_demo[:8] = rng.uniform(0.97, 0.999, size=8)

    weights = ss.domain_weights(source=source_prob_demo, target=target_prob_demo, reweight="both", shrinkage=0.0)
    ess = weights.effective_sample_size()
    print(f"source ESS {ess.source:.1f} / 400")  # → 6.5 — concentrated
    ```

    Sweep `shrinkage` — lower = stronger but higher variance:

    ```python
    for lam in [0.0, 0.25, 0.5, 0.75, 1.0]:
        w = ss.domain_weights(source=source_prob_demo, target=target_prob_demo, reweight="both", shrinkage=lam)
        e = w.effective_sample_size()
        print(f"shrinkage={lam:<4}  source ESS={e.source:7.2f}  target ESS={e.target:7.2f}")
    ```

    Worry when `ESS < n/4` for either group (rule of thumb). If ESS stays low even at `0.5`, groups barely overlap — skip weighting.

Full scripts: `examples/weighting/_code/` · Reference: [Importance weights](../../api/weighting.md).
