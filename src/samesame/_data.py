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
    source: ArrayLike,
    target: ArrayLike,
) -> tuple[NDArray[np.int_], NDArray]:
    """Build binary labels and combined outlier scores from two samples."""
    source_scores = as_numeric_vector(source, name="source")
    target_scores = as_numeric_vector(target, name="target")
    labels = np.concatenate(
        (
            np.zeros(source_scores.shape[0], dtype=int),
            np.ones(target_scores.shape[0], dtype=int),
        )
    )
    scores = np.concatenate((source_scores, target_scores))
    return labels, scores


__all__ = ["build_two_sample_dataset"]
