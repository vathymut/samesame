# Copyright (c) 2025-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

"""Reference logit-derived score functions for confidence and OOD monitoring.

These post-hoc methods are intended for pre-trained classifiers and return
scores that can be used to rank inputs by in-distribution confidence.

References
----------
Liang, J., Hou, R., Hu, M., Chang, H., Shan, S., & Chen, X. (2025).
Revisiting Logit Distributions for Reliable Out-of-Distribution Detection.
arXiv:2510.20134v1. https://arxiv.org/html/2510.20134v1
"""

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
    """LogitGap OOD detection score.

    Compute the average gap between the largest logit and the remaining logits
    for each sample.

    Intuitively, in-distribution samples tend to have a dominant class logit,
    while out-of-distribution samples often have flatter logit profiles.

    The scoring function is defined as:

    .. math::

        S_{\\text{LogitGap}}(x; f) = \\frac{1}{K-1}
        \\sum_{j=2}^{K} (z'_1 - z'_j)

    where :math:`z'_1` is the maximum logit and :math:`z'_j` are logits
    sorted in descending order. Higher scores indicate higher confidence
    for ID samples.

    Parameters
    ----------
    logits : NDArray
        Array of shape (n_samples, n_classes) containing raw logits from
        a pre-trained classification model.

    Returns
    -------
    NDArray
        Array of shape (n_samples,) containing OOD scores. Higher scores
        indicate higher likelihood of being in-distribution.

    Raises
    ------
    ValueError
        If ``logits`` is not a finite 2D array with at least two classes.

    See Also
    --------
    max_logit : Simple baseline using only the maximum logit value.

    Notes
    -----
    The implementation uses
    :math:`\\max_k z_k - \\frac{1}{K-1}\\sum_{j \\ne k^*} z_j`, where
    :math:`k^*` is the index of the maximum logit. This is equivalent to the
    formulation in [1]_.

    References
    ----------
    .. [1] Liang, Jiachen, et al.
       "Revisiting Logit Distributions for Reliable Out-of-Distribution Detection."
       The Thirty-ninth Annual Conference on Neural Information Processing Systems.
       2025.

    Examples
    --------
    >>> logits = np.array([[5.0, 1.0, 0.5], [2.0, 2.1, 1.9]])
    >>> np.round(logit_gap(logits), 2)
    array([4.25, 0.15], dtype=float32)
    >>> logits = np.array([[5.0, 1.0, 0.5], [2.0, 2.1, 1.9]])
    >>> scores = logit_gap(logits)  # doctest: +SKIP
    >>> print(scores)
    [4.25 0.15]
    """
    logits = _validate_logits(logits)
    n_classes = logits.shape[1]
    max_logits = np.max(logits, axis=1)
    mean_rest = (np.sum(logits, axis=1) - max_logits) / (n_classes - 1)
    return max_logits - mean_rest


def max_logit(logits: NDArray) -> NDArray:
    """MaxLogit OOD detection score (baseline method).

    Compute the maximum logit for each sample.

    This is a simple baseline that uses only the top class logit and ignores
    the rest of the logit vector.

    The scoring function is defined as:

    .. math::

        S_{\\text{MaxLogit}}(x; f) = \\max_k z_k

    where :math:`z_k` is the logit for class k. Higher scores indicate
    higher confidence for the predicted class.

    Parameters
    ----------
    logits : NDArray
        Array of shape (n_samples, n_classes) containing raw logits from
        a pre-trained classification model.

    Returns
    -------
    NDArray
        Array of shape (n_samples,) containing OOD scores (maximum logits).
        Higher scores indicate higher confidence, but with limited
        discriminative power for OOD detection compared to LogitGap.

    Raises
    ------
    ValueError
        If ``logits`` is not a finite 2D array with at least two classes.

    See Also
    --------
    logit_gap : Improved OOD detection using logit gap.

    Notes
    -----
    MaxLogit is primarily a baseline comparator for richer logit-distribution
    methods such as :func:`logit_gap` [1].

    References
    ----------
    .. [1] Liang, Jiachen, et al.
       "Revisiting Logit Distributions for Reliable Out-of-Distribution Detection."
       The Thirty-ninth Annual Conference on Neural Information Processing Systems.
       2025.

    Examples
    --------
    >>> logits = np.array([[5.0, 1.0, 0.5], [2.0, 2.1, 1.9]])
    >>> max_logit(logits)
    array([5. , 2.1], dtype=float32)
    >>> logits = np.array([[5.0, 1.0, 0.5], [2.0, 2.1, 1.9]])
    >>> scores = max_logit(logits)  # doctest: +SKIP
    >>> print(scores)
    [5.0 2.1]
    """
    logits = _validate_logits(logits)
    return np.max(logits, axis=1)


__all__ = ["logit_gap", "max_logit"]
