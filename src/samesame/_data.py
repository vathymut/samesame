# Copyright (c) 2025-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray


def assign_labels(samples: Sequence[NDArray]) -> NDArray:
    assert len(samples) > 1, f"{len(samples)=} must be greater than 1."
    labels = [np.repeat(i, np.array(s).shape[0]) for i, s in enumerate(samples)]
    return np.concatenate(labels, axis=None)


def group_by(data: NDArray, groups: NDArray):
    unique_groups, group_indices = np.unique(groups, return_inverse=True)
    grouped = [data[group_indices == i] for i in range(len(unique_groups))]
    return grouped


def concat_samples(samples: Sequence[NDArray]) -> NDArray:
    first_sample = next(iter(samples))
    if first_sample.ndim < 2:
        return np.concatenate(samples, axis=None)
    return np.concatenate(samples, axis=0)
