# How to weight for common support

Start unweighted. Weight only when poor overlap is a real concern. Weighting reframes the comparison around **common support**, the overlap of source and target. It adds no information where groups do not overlap and changes the population you describe. Not a default correction.

## Prerequisites

- An unweighted shift and an honest domain probability `P(target|x)` out of sample (`cross_val_predict` or `oob_decision_function_`).
- Domain probability kept separate from the score you test.

## Steps

### 1. Estimate the domain probability

Estimate out of sample. This is membership, not outcome quality.

```python
--8<-- "snippets/heloc-split.py:heloc-domain"
source_prob = domain_prob[split.values == 0]
target_prob = domain_prob[split.values == 1]
```

### 2. Build weights

--8<-- "snippets/reweight-table.txt"

--8<-- "snippets/shrinkage-table.txt"

```python
import samesame as ss
weights = ss.domain_weights(source=source_prob, target=target_prob, reweight="both", shrinkage=0.5)
```

Weights preserve nominal size (`Σw = n` per group; `1` if unweighted). They change influence, not classifier quality. See [Core concepts](../explanation/core-concepts.md).

### 3. Test with and without weights

Keep `worse` the same so both tests ask the same question.

```python
--8<-- "snippets/heloc-split.py:heloc-split"
--8<-- "snippets/heloc-split.py:heloc-risk-model"
unweighted = ss.test_harmful_shift(source=train_risk, target=deployment_risk, worse="higher", rng=np.random.default_rng(12345))
w_both = ss.domain_weights(source=source_prob, target=target_prob, reweight="both", shrinkage=0.5)
weighted = ss.test_harmful_shift(source=train_risk, target=deployment_risk, worse="higher", weights=w_both, rng=np.random.default_rng(12345))
print(f"Unweighted p={unweighted.pvalue:.4f} Weighted p={weighted.pvalue:.4f}")
```

- Weaker after weighting → evidence in low-overlap regions.
- Persists → harm present where groups overlap.

### 4. Check effective sample size

```python
ess = weights.effective_sample_size()  # Kish (1965): (sum w)² / sum w²
print(f"source ESS {ess.source:.1f}/{len(source_prob)}  target ESS {ess.target:.1f}/{len(target_prob)}")
```

`ESS/n` well below `0.5` warns that few points drive the result. There is no universal cutoff (Elvira et al., 2022). If low even at `shrinkage=0.5`, keep the comparison unweighted. See [Effective sample size](../api/weighting.md#effective-sample-size).

## Expected outcome

On HELOC the risk alarm survives at `shrinkage=0.5` (`p=0.0001` all modes). ESS 2,491/7,683 (about 0.32) and 1,532/2,188 (about 0.70) shows harm where groups overlap.

## Troubleshooting

- **`ESS/n` <<0.5 at `shrinkage=0.5`** → keep the comparison unweighted; groups lack enough common support.
- **Weighted and unweighted agree** → report unweighted; weighting confirms the signal is not driven by low-overlap regions.

??? details "Diagnose weight concentration"

    ```python
    rng = np.random.default_rng(7)
    s = rng.beta(2, 5, size=400); t = rng.beta(5, 2, size=400); s[:8] = rng.uniform(0.97, 0.999, size=8)
    for lam in [0.0, 0.5, 1.0]:
        w = ss.domain_weights(source=s, target=t, reweight="both", shrinkage=lam)
        e = w.effective_sample_size()
        print(f"shrinkage={lam:<4}  ESS s={e.source:6.1f}  t={e.target:6.1f}")
    ```

See [Core concepts](../explanation/core-concepts.md), [Importance weights](../api/weighting.md), and [How the harm test works](../explanation/harmful-shift-statistic.md). Scripts: `examples/weighting/_code/`.

Report the unweighted result first. Add the weighted result when overlap is poor and the question is whether the shift survives on common support.
