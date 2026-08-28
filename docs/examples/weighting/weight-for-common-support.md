# How to: Weight for common support

Down-weight low-overlap points so the test focuses on where source and target both have density. Includes the minimal synthetic check (no data download) and the HELOC production example.

!!! info "Prerequisites"
    - [Detect any shift](../tutorials/detect-distribution-shift.md) — honest scores and `P(target|x)`.
    - [Is it harmful?](../tutorials/check-shift-harm.md) — `worse` and direction.
    - [When weights help](../../explanation/importance-weights-rationale.md) — intuition.

## When to use which `reweight`

--8<-- "snippets/reweight-table.txt"

Start unweighted, then compare weighted. Weighting is for known overlap issues, not a default.

## Minimal synthetic check (no download, 30 seconds)

Proves the mechanics before you use your own data. Same pattern as the HELOC example below: domain `P(target|x)` is for weighting only — don't reuse it as the harm score.

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
)[:, 1]  # pooled P(target|x); prior ratio n_source/n_target inferred from sizes

rng = np.random.default_rng(12345)
risk_score = 0.9*X[:,0] - 0.6*X[:,1] + 0.4*X[:,2] + rng.normal(scale=0.4, size=len(group))
source_scores, target_scores = risk_score[group == 0], risk_score[group == 1]

source_prob, target_prob = domain_prob[group == 0], domain_prob[group == 1]
weights = ss.domain_weights(source=source_prob, target=target_prob, reweight="source", shrinkage=0.5)

rng = np.random.default_rng(12345)
unweighted = ss.test_harmful_shift(source=source_scores, target=target_scores, worse="higher", rng=rng)
rng = np.random.default_rng(12345)
weighted = ss.test_harmful_shift(source=source_scores, target=target_scores, worse="higher", weights=weights, rng=rng)
print(f"Unweighted p={unweighted.pvalue:.4f} Weighted p={weighted.pvalue:.4f}")  # → 1.0, 0.62 — no signal here
```

--8<-- "snippets/pvalue-guidance.txt"

- **Unweighted** — every point at full strength.
- **Weighted** — emphasizes overlap (here: source reweighted toward target).
- Strong unweighted but weak weighted → shift was in low-overlap regions.

--8<-- "snippets/clipping-note.txt"

Use the same `worse` for both calls.

## HELOC example (production-like)

Same split as [Monitor credit](../credit/monitor-credit.md). Domain `P(target|x)` is again for weighting only.

```python
import numpy as np
import samesame as ss

--8<-- "snippets/heloc-split.py:heloc-split"
--8<-- "snippets/heloc-split.py:heloc-domain"
--8<-- "snippets/heloc-split.py:heloc-risk-model"

source_prob = domain_prob[split.values == 0]
target_prob = domain_prob[split.values == 1]
```

--8<-- "snippets/honest-scores-ref.txt"

### Compare unweighted vs weighted

=== "Source-only (target unchanged)"

    ```python
    weights_src = ss.domain_weights(source=source_prob, target=target_prob, reweight="source", shrinkage=0.5)
    unweighted = ss.test_harmful_shift(source=train_risk, target=deployment_risk, worse="higher", rng=np.random.default_rng(12345))
    weighted_src = ss.test_harmful_shift(source=train_risk, target=deployment_risk, worse="higher", weights=weights_src, rng=np.random.default_rng(12345))
    print(f"Unweighted p={unweighted.pvalue:.4f}")      # → 0.0001
    print(f"Source-weighted p={weighted_src.pvalue:.4f}") # → 0.0001 — persists on common support
    ```

    Use when source contains cases production rarely sees.

=== "Both groups (mutual support)"

    ```python
    weights_both = ss.domain_weights(source=source_prob, target=target_prob, reweight="both", shrinkage=0.5)
    weighted_both = ss.test_harmful_shift(source=train_risk, target=deployment_risk, worse="higher", weights=weights_both, rng=np.random.default_rng(12345))
    print(f"Doubly-weighted p={weighted_both.pvalue:.4f}")  # → 0.0001
    ```

    Use when both sides have low-overlap regions. If the signal shrinks only here, target-side outliers were still influencing the result.

## Diagnose with effective sample size (ESS)

Weighting can pile mass on a few points. Check *before* trusting the result.

```python
rng = np.random.default_rng(7)
source_prob_demo = rng.beta(a=2, b=5, size=400)
target_prob_demo = rng.beta(a=5, b=2, size=400)
source_prob_demo[:8] = rng.uniform(0.97, 0.999, size=8)  # a few highly target-like points

weights = ss.domain_weights(source=source_prob_demo, target=target_prob_demo, reweight="both", shrinkage=0.0)
ess = weights.effective_sample_size()
print(f"source ESS {ess.source:.1f} / 400")  # → 6.5 / 400 — concentrated
print(f"target ESS {ess.target:.1f} / 400")  # → 203.6 / 400
```

Sweep `shrinkage` — lower = stronger correction but higher variance:

```python
for lam in [0.0, 0.25, 0.5, 0.75, 1.0]:
    w = ss.domain_weights(source=source_prob_demo, target=target_prob_demo, reweight="both", shrinkage=lam)
    e = w.effective_sample_size()
    print(f"shrinkage={lam:<4}  source ESS={e.source:7.2f}  target ESS={e.target:7.2f}")
```

```text
shrinkage=0.0   source ESS=   6.53  target ESS= 203.61
shrinkage=0.25  source ESS= 189.14  target ESS= 258.52
shrinkage=0.5   source ESS= 283.68  target ESS= 302.34
shrinkage=0.75  source ESS= 342.03  target ESS= 347.07
shrinkage=1.0   source ESS= 400.00  target ESS= 400.00
```

--8<-- "snippets/ess-rule.txt"

--8<-- "snippets/shrinkage-table.txt"

If ESS stays low even at `0.5`, groups barely overlap — skip weighting or collect more comparable data.

??? example "Full scripts — copy and run"
    ```python
    --8<-- "examples/weighting/_code/source_reweighting_example.py:full"
    ```
    ```python
    --8<-- "examples/weighting/_code/diagnose_weight_concentration_example.py:full"
    ```

See [When weights help](../../explanation/importance-weights-rationale.md) for formulas and [Importance weights API](../../api/weighting.md) for reference.
