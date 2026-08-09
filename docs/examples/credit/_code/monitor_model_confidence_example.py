"""Guide-owned helpers for the monitor-model-confidence example."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.special import logit

# --8<-- [start:imports]
import numpy as np
from scipy.special import logit
# --8<-- [end:imports]


# --8<-- [start:logit-gap]
def logit_gap(logits: NDArray[np.float64]) -> NDArray[np.float64]:
    logits = np.asarray(logits, dtype=float)
    max_logits = np.max(logits, axis=1)
    mean_rest = (np.sum(logits, axis=1) - max_logits) / (logits.shape[1] - 1)
    return max_logits - mean_rest


# --8<-- [end:logit-gap]
# --8<-- [start:outlier-scores]
def outlier_scores_from_probabilities(
    probabilities: NDArray[np.float64],
    *,
    clip: float = 1e-6,
) -> NDArray[np.float64]:
    clipped = np.clip(np.asarray(probabilities, dtype=float), clip, 1.0 - clip)
    return logit_gap(logit(clipped))


# --8<-- [end:outlier-scores]

__all__ = ["logit_gap", "outlier_scores_from_probabilities"]
