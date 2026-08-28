"""Full runnable example for Diagnose weight concentration (synthetic)."""

# --8<-- [start:full]
import numpy as np

import samesame as ss

rng = np.random.default_rng(7)
source_prob = rng.beta(a=2, b=5, size=400)
target_prob = rng.beta(a=5, b=2, size=400)
source_prob[:8] = rng.uniform(0.97, 0.999, size=8)

# sweep shrinkage and compare ESS
for lam in [0.0, 0.25, 0.5, 0.75, 1.0]:
    w = ss.domain_weights(source=source_prob, target=target_prob, reweight="both", shrinkage=lam)
    e = w.effective_sample_size()
    print(f"shrinkage={lam:<4}  source ESS={e.source:7.2f}  target ESS={e.target:7.2f}")

# example weighted test (synthetic harm scores)
rng = np.random.default_rng(12345)
source_scores = rng.normal(0, 1, 400)
target_scores = rng.normal(0.3, 1, 400)
w = ss.domain_weights(source=source_prob, target=target_prob, reweight="both", shrinkage=0.5)
unweighted = ss.test_harmful_shift(source=source_scores, target=target_scores, worse="higher", rng=np.random.default_rng(1))
weighted = ss.test_harmful_shift(source=source_scores, target=target_scores, worse="higher", weights=w, rng=np.random.default_rng(1))
print(f"Unweighted p={unweighted.pvalue:.4f}, weighted p={weighted.pvalue:.4f}")
# --8<-- [end:full]
