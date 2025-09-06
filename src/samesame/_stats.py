# Copyright (c) 2025-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from numpy.typing import NDArray


@dataclass
class EmpiricalPvalue:
    """
    Computes p-value from empirical null distribtuion.

    Attributes
    ----------
    observed : float
        The observed statistic value.
    null_distribution : NDArray
        The null distribution of the statistic.
    adjustment : {0, 1}, optional
        Adjustment factor for p-value computation, by default 1.
    eps : float
        Relative tolerance for numerical comparisons.
    gamma : float
        Adjustment for numerical precision.
    n_resamples : int
        Number of resampling iterations.

    Notes
    -----
    This class is designed to handle numerical precision issues when comparing
    observed statistics to null distributions.

    Examples
    --------
    >>> import numpy as np
    >>> from samesame._stats import EmpiricalPvalue
    >>> observed = 0.5
    >>> null_distribution = np.random.rand(1000)
    >>> test = EmpiricalPvalue(observed, null_distribution)
    >>> pvalue = test.two_sided()
    """

    observed: float
    null_distribution: NDArray
    adjustment: Literal[0, 1] = 1
    eps: float = field(init=False, repr=False)
    gamma: float = field(init=False, repr=False)
    n_resamples: int = field(init=False, repr=False)

    def __post_init__(self):
        # relative tolerance for detecting numerically distinct but
        # theoretically equal values in the null distribution
        dtype = np.array(self.observed).dtype
        self.eps = (
            0.0 if not np.issubdtype(dtype, np.inexact) else np.finfo(dtype).eps * 100
        )  # type: ignore
        self.gamma = np.abs(self.eps * self.observed)
        self.n_resamples = self.null_distribution.shape[0]

    def less(self):
        """Compute one-sided p-values for the "less" hypothesis."""
        cmps = self.null_distribution <= self.observed + self.gamma
        pvalues = (cmps.sum(axis=0) + self.adjustment) / (
            self.n_resamples + self.adjustment
        )
        return np.clip(pvalues, 0, 1)

    def greater(self):
        """Compute one-sided p-values for the "greater" alternative."""
        cmps = self.null_distribution >= self.observed - self.gamma
        pvalues = (cmps.sum(axis=0) + self.adjustment) / (
            self.n_resamples + self.adjustment
        )
        return np.clip(pvalues, 0, 1)

    def two_sided(self):
        """Compute two-sided p-values."""
        pvalues_less = self.less()
        pvalues_greater = self.greater()
        pvalues = np.minimum(pvalues_less, pvalues_greater) * 2
        return np.clip(pvalues, 0, 1)


def _bayes_factor(
    posterior: NDArray,
    threshold: float = 0.0,
    adjustment: Literal[0, 1] = 1,
) -> float:
    pvalue = EmpiricalPvalue(
        observed=threshold, null_distribution=posterior, adjustment=adjustment
    )
    denom_bf = pvalue.less()
    num_bf = 1.0 - denom_bf
    return num_bf / denom_bf
