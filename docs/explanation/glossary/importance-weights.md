# Importance weights

`ImportanceWeights` — frozen dataclass holding `.source` and `.target` weight arrays, each normalized to sum to its group size.

```python
import samesame as ss
import numpy as np

# from a domain classifier
weights = ss.domain_weights(source=source_prob, target=target_prob, reweight="both", shrinkage=0.5)

# or custom
weights = ss.ImportanceWeights(source=source_w, target=target_w)

ess = weights.effective_sample_size()  # Kish's ESS per group
print(ess.source, ess.target)
```

Build via `ss.domain_weights` (RIW from domain probabilities) or construct directly from custom weights. Inactive groups get `1` for every observation (uniform after normalization).

Weights are permuted with labels under the null (concatenated vector), so inference accounts for them.

See also: [Reweight](reweight.md), [Shrinkage](shrinkage.md), [ESS](ess.md), [Importance weights API](../../api/weighting.md).
