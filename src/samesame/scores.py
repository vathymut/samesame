"""Public outlier-score seam."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def _validate_logits(logits: NDArray) -> NDArray:
    logits = np.asarray(logits)

    if logits.ndim != 2:
        raise ValueError(
            f"logits must be 2D array of shape (n_samples, n_classes), "
            f"got shape {logits.shape}"
        )

    _, n_classes = logits.shape
    if n_classes < 2:
        raise ValueError(f"logits must have at least 2 classes, got {n_classes}")

    if not np.isfinite(logits).all():
        raise ValueError("logits contain NaN or infinite values")

    if np.issubdtype(logits.dtype, np.floating):
        max_abs = np.max(np.abs(logits), initial=0.0)
        if max_abs <= np.finfo(np.float32).max:
            return logits.astype(np.float32, copy=False)
        return logits.astype(np.float64, copy=False)

    return logits.astype(np.float32, copy=False)


def logit_gap(logits: NDArray) -> NDArray:
    """Compute the LogitGap Outlier score."""
    logits = _validate_logits(logits)
    n_classes = logits.shape[1]
    max_logits = np.max(logits, axis=1)
    mean_rest = (np.sum(logits, axis=1) - max_logits) / (n_classes - 1)
    return max_logits - mean_rest


def max_logit(logits: NDArray) -> NDArray:
    """Compute the MaxLogit Outlier score."""
    logits = _validate_logits(logits)
    return np.max(logits, axis=1)


__all__ = ["logit_gap", "max_logit"]
