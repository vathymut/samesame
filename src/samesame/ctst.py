# Copyright (c) 2025-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
"""
Classifier two-sample tests (CTST) from binary classification metrics.

The classifier two-sample test broadly consists of three steps: (1)
training a classifier, (2) scoring the two samples and (3) turning a test
statistic into a p-value from these scores. This test statistic can be the
performance metric of a binary classifier such as the (weighted) area under
the receiver operating characteristic curve, the Matthews correlation
coefficient, and the (balanced) accuracy. This module tackles step (3).

References
----------
.. [1] Lopez-Paz, David, and Maxime Oquab. "Revisiting Classifier Two-Sample
   Tests." International Conference on Learning Representations. 2017.

.. [2] Friedman, Jerome. "On multivariate goodness-of-fit and two-sample
   testing." No. SLAC-PUB-10325. SLAC National Accelerator Laboratory (SLAC),
   Menlo Park, CA (United States), 2004.

.. [3] Kübler, Jonas M., et al. "Automl two-sample test." Advances in Neural
   Information Processing Systems 35 (2022): 15929-15941.

.. [4] Ciémençon, Stéphan, Marine Depecker, and Nicolas Vayatis. "AUC
   optimization and the two-sample problem." Proceedings of the 23rd
   International Conference on Neural Information Processing Systems. 2009.

.. [5] Hediger, Simon, Loris Michel, and Jeffrey Näf. "On the use of random
   forest for two-sample testing." Computational Statistics & Data Analysis
   170 (2022): 107435.

.. [6] Kim, Ilmun, et al. "Classification accuracy as a proxy for two-sample
   testing." Annals of Statistics 49.1 (2021): 411-434.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from functools import cached_property
from inspect import signature
from operator import methodcaller
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from sklearn.utils import (
    check_consistent_length,
    column_or_1d,
)
from sklearn.utils.multiclass import type_of_target

from samesame._data import assign_labels, concat_samples
from samesame._permute import permutation_null
from samesame._stats import EmpiricalPvalue
from samesame._utils import check_metric_function


@dataclass
class CTST:
    """
    Classifier two-sample test (CTST) using a binary classification metric.

    This test compares scores (predictions) from two independent samples.
    Rejecting the null implies that scoring is not random and that the
    classifier is able to distinguish between the two samples.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.metrics import balanced_accuracy_score, matthews_corrcoef, roc_auc_score
    >>> from samesame.ctst import CTST
    >>> actual = np.array([0, 1, 1, 0])
    >>> scores = np.array([0.2, 0.8, 0.6, 0.4])
    >>> ctst_acc = CTST(actual, scores, metric=balanced_accuracy_score)
    >>> ctst_mcc = CTST(actual, scores, metric=matthews_corrcoef)
    >>> ctst_auc = CTST(actual, scores, metric=roc_auc_score)
    >>> print(ctst_acc.pvalue) # doctest: +SKIP
    >>> print(ctst_mcc.pvalue) # doctest: +SKIP
    >>> print(ctst_auc.pvalue) # doctest: +SKIP
    >>> ctst_ = CTST.from_samples(scores, scores, metric=roc_auc_score)
    >>> isinstance(ctst_, CTST)
    True
    """

    actual: NDArray = field(repr=False)
    predicted: NDArray = field(repr=False)
    metric: Callable
    n_resamples: int = 9999
    rng: np.random.Generator = np.random.default_rng()
    n_jobs: int = 1
    batch: int | None = None
    alternative: Literal["less", "greater", "two_sided"] = "two_sided"

    def __post_init__(self):
        """Validate inputs."""
        self.actual = column_or_1d(self.actual)
        self.predicted = column_or_1d(self.predicted)
        check_consistent_length(self.actual, self.predicted)
        assert type_of_target(self.actual, "actual") == "binary"
        type_predicted = type_of_target(self.predicted, "predicted")
        assert type_predicted in (
            "binary",
            "continuous",
            "multiclass",
        ), f"Expected 'predicted' to be binary or continuous, got {type_predicted}."
        assert check_metric_function(self.metric), (
            f"'metric' expects a callable that conforms to scikit-learn metric. "
            f"{signature(self.metric)=} does not."
        )

    @cached_property
    def statistic(self) -> float:
        """
        Compute the observed test statistic.

        Returns
        -------
        float
            The test statistic.

        Notes
        -----
        The result is cached to avoid (expensive) recomputation.
        """
        return self.metric(self.actual, self.predicted)

    @cached_property
    def null(self) -> NDArray:
        """
        Compute the null distribution of the test statistic.

        Notes
        -----
        The result is cached to avoid (expensive) recomputation since the
        null distribution requires permutations.
        """
        return permutation_null(
            data=(self.actual, self.predicted),
            statistic=self.metric,
            n_resamples=self.n_resamples,
            rng=self.rng,
            n_jobs=self.n_jobs,
            batch=self.batch,
        ).astype(float)

    @cached_property
    def pvalue(self):
        """
        Compute the p-value using permutations.

        Notes
        -----
        The result is cached to avoid (expensive) recomputation.
        """
        test_ = EmpiricalPvalue(self.statistic, self.null)
        pvalue_ = methodcaller(self.alternative)(test_)
        return pvalue_

    @classmethod
    def from_samples(
        cls,
        first_sample: NDArray,
        second_sample: NDArray,
        metric: Callable,
        n_resamples: int = 9999,
        rng: np.random.Generator = np.random.default_rng(),
        n_jobs: int = 1,
        batch: int | None = None,
        alternative: Literal["less", "greater", "two_sided"] = "two_sided",
    ):
        """
        Create a CTST instance from two samples.

        Parameters
        ----------
        first_sample : NDArray
            First sample of scores. These can be binary or continuous.
        second_sample : NDArray
            Second sample of scores. These can be binary or continuous.

        Returns
        -------
        CTST
            An instance of the CTST class.
        """
        assert type_of_target(first_sample) == type_of_target(second_sample)
        samples = (first_sample, second_sample)
        actual = assign_labels(samples)
        predicted = concat_samples(samples)
        return cls(
            actual,
            predicted,
            metric,
            n_resamples,
            rng,
            n_jobs,
            batch,
            alternative,
        )


__all__ = [
    "CTST",
]
