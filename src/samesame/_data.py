# Copyright (c) 2025-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray
from sklearn.utils.multiclass import type_of_target


def assign_labels(samples: Sequence[NDArray]) -> NDArray:
    if len(samples) <= 1:
        raise ValueError(f"len(samples) must be greater than 1, got {len(samples)}.")
    labels = [np.repeat(i, np.array(s).shape[0]) for i, s in enumerate(samples)]
    return np.concatenate(labels, axis=None).astype(int)


def group_by(data: NDArray, groups: NDArray):
    unique_groups, group_indices = np.unique(groups, return_inverse=True)
    grouped = [data[group_indices == i] for i in range(len(unique_groups))]
    return grouped


def concat_samples(samples: Sequence[NDArray]) -> NDArray:
    first_sample = np.asarray(samples[0])
    if first_sample.ndim < 2:
        return np.concatenate(samples, axis=None)
    return np.concatenate(samples, axis=0)


def build_two_sample_dataset(
    first_sample: NDArray,
    second_sample: NDArray,
) -> tuple[NDArray, NDArray]:
    if type_of_target(first_sample) != type_of_target(second_sample):
        raise ValueError(
            "first_sample and second_sample must have the same target type."
        )
    samples = (first_sample, second_sample)
    return assign_labels(samples), concat_samples(samples)
