# Copyright (c) 2025-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from samesame._utils import as_numeric_vector


def build_two_sample_dataset(
    reference: ArrayLike,
    candidate: ArrayLike,
) -> tuple[NDArray[np.int_], NDArray]:
    """Build binary labels and a combined score vector from two samples."""
    reference_scores = as_numeric_vector(reference, name="reference")
    candidate_scores = as_numeric_vector(candidate, name="candidate")
    labels = np.concatenate(
        (
            np.zeros(reference_scores.shape[0], dtype=int),
            np.ones(candidate_scores.shape[0], dtype=int),
        )
    )
    scores = np.concatenate((reference_scores, candidate_scores))
    return labels, scores


__all__ = ["build_two_sample_dataset"]
