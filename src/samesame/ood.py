# Copyright (c) 2025-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

"""Functions for Out-of-Distribution (OOD) detection.

These post-hoc OOD detection methods are for pre-trained supervised models,
which leverage the information from the entire logit space to enhance
in-distribution (ID) and out-of-distribution (OOD) separability.

References
----------
Liang, J., Hou, R., Hu, M., Chang, H., Shan, S., & Chen, X. (2025).
Revisiting Logit Distributions for Reliable Out-of-Distribution Detection.
arXiv:2510.20134v1. https://arxiv.org/html/2510.20134v1
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def logit_gap(logits: NDArray) -> NDArray:
    """LogitGap OOD detection score.

    Computes the average gap between the maximum logit and remaining logits
    for each sample. This method leverages the observation that
    in-distribution (ID) samples tend to have higher maximum logits with
    lower non-maximum logits, while out-of-distribution (OOD) samples
    exhibit flatter logit distributions.

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

    See Also
    --------
    max_logit : Simple baseline using only the maximum logit value.

    Notes
    -----
    The LogitGap method is motivated by the observation that ID samples
    exhibit more pronounced logit distributions (higher maximum logit with
    lower non-maximum logits), while OOD samples show flatter distributions
    (smaller gaps between logits).

    The implementation computes the average gap efficiently by calculating
    the difference between the maximum logit and the mean of all other
    logits: max_logit - mean(other_logits). This is mathematically
    equivalent to the definition in Equation (4).

    This method requires no additional training or calibration and can be
    applied as a post-hoc scoring function to any pre-trained classification
    model. It demonstrates superior performance compared to MaxLogit baseline
    across various benchmark datasets.

    The function expects raw logits (pre-softmax values) rather than
    probabilities. Using logits directly preserves more information about
    the model's confidence distribution.

    References
    ----------
    .. [1] Liang, Jiachen, et al.
       "Revisiting Logit Distributions for Reliable Out-of-Distribution Detection."
       The Thirty-ninth Annual Conference on Neural Information Processing Systems.
       2025.

    Examples
    --------
    >>> logits = np.array([[5.0, 1.0, 0.5], [2.0, 2.1, 1.9]])
    >>> scores = logit_gap(logits)  # doctest: +SKIP
    >>> print(scores)
    [4.25 0.15]
    """
    logits = np.asarray(logits, dtype=np.float32)

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

    # Sort logits in descending order for each sample
    sorted_logits = np.sort(logits, axis=1)[:, ::-1]

    # Get maximum logit (z_1')
    max_logit_val = sorted_logits[:, 0]

    # Compute average gap: (1/(K-1)) * sum(z_1' - z_j') for j=2 to K
    # This is equivalent to: z_1' - mean(z_2' to z_K')
    avg_other_logits = np.mean(sorted_logits[:, 1:], axis=1)

    scores = max_logit_val - avg_other_logits

    return scores


def max_logit(logits: NDArray) -> NDArray:
    """MaxLogit OOD detection score (baseline method).

    Computes the maximum logit value for each sample. This is a simple
    baseline that uses only the most confident prediction, disregarding
    information from other classes.

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

    See Also
    --------
    logit_gap : Improved OOD detection using logit gap.

    Notes
    -----
    MaxLogit is included as a baseline for comparison. The paper
    demonstrates that LogitGap achieves significantly better OOD detection
    performance by leveraging information from all logits rather than just
    the maximum.

    This method has several limitations:
    - It only uses information from the top predicted class
    - It ignores the distribution of non-maximum logits
    - It provides limited discriminative power between ID and OOD samples

    MaxLogit is conceptually similar to Maximum Softmax Probability (MSP)
    but operates directly on logits rather than probabilities. Both methods
    are widely used baselines in OOD detection literature.

    Despite its simplicity, MaxLogit serves as a reasonable baseline and
    requires no additional computation beyond extracting the maximum value
    from the logit vector. It can be useful for computational efficiency
    when more sophisticated methods are not necessary.

    References
    ----------
    .. [1] Liang, Jiachen, et al.
       "Revisiting Logit Distributions for Reliable Out-of-Distribution Detection."
       The Thirty-ninth Annual Conference on Neural Information Processing Systems.
       2025.

    Examples
    --------
    >>> logits = np.array([[5.0, 1.0, 0.5], [2.0, 2.1, 1.9]])
    >>> scores = max_logit(logits)  # doctest: +SKIP
    >>> print(scores)
    [5.0 2.1]
    """
    logits = np.asarray(logits, dtype=np.float32)

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

    # Simply return the maximum logit for each sample
    scores = np.max(logits, axis=1)

    return scores
