# Copyright (c) 2025-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from typing import Literal

import numpy as np
from numpy.typing import NDArray


def _empirical_pvalue(
    observed: float,
    null_distribution: NDArray,
    alternative: Literal["less", "greater", "two-sided"],
    *,
    adjustment: Literal[0, 1] = 1,
) -> float:
    """Compute empirical p-value from a null distribution.

    Parameters
    ----------
    observed : float
        The observed statistic value.
    null_distribution : NDArray
        The null distribution of the statistic.
    alternative : {'less', 'greater', 'two-sided'}
        Direction of the alternative hypothesis.
    adjustment : {0, 1}, optional
        Laplace-style continuity adjustment, by default 1.

    Returns
    -------
    float
        Empirical p-value in [0, 1].
    """
    dtype = np.array(observed).dtype
    eps = 0.0 if not np.issubdtype(dtype, np.inexact) else np.finfo(dtype).eps * 100
    gamma = np.abs(eps * observed)
    n = null_distribution.shape[0]

    if alternative == "less":
        count = (null_distribution <= observed + gamma).sum(axis=0)
    elif alternative == "greater":
        count = (null_distribution >= observed - gamma).sum(axis=0)
    else:  # two-sided
        count_less = (null_distribution <= observed + gamma).sum(axis=0)
        count_greater = (null_distribution >= observed - gamma).sum(axis=0)
        count = np.minimum(count_less + adjustment, count_greater + adjustment)
        return float(np.clip(count * 2 / (n + adjustment), 0, 1))

    return float(np.clip((count + adjustment) / (n + adjustment), 0, 1))


def _bayes_factor(
    posterior: NDArray,
    threshold: float = 0.0,
    adjustment: Literal[0, 1] = 1,
) -> float:
    denom_bf = _empirical_pvalue(threshold, posterior, "less", adjustment=adjustment)
    num_bf = 1.0 - denom_bf
    return float(np.float64(num_bf) / np.float64(denom_bf))
