# Copyright (c) 2025-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
"""
Classifier two-sample tests (CTST) from binary classification metrics.

The classifier two-sample test broadly consists of three steps: (1) training a
(binary) classifier, (2) predicting out-of-sample and (3) turning a test
statistic into a p-value from these predictions. This test statistic can be the
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
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from scipy.stats import permutation_test
from sklearn.utils.multiclass import type_of_target

from samesame._data import build_two_sample_dataset
from samesame._utils import (
    check_metric_function,
    validate_and_normalise_weights,
    validate_binary_actual_with_predicted,
)


@dataclass
class CTST:
    """
    Classifier two-sample test (CTST) using a binary classification metric.

    This test compares out-of-sample scores from two independent samples.
    Small p-values indicate that the score-label association is unlikely under
    random pairing, suggesting separability between the samples.

    Parameters
    ----------
    actual : NDArray
        Binary indicator for sample membership.
    predicted : NDArray
        Estimated (predicted) scores for corresponding samples in `actual`.
    metric : Callable
        A callable that conforms to scikit-learn metric API. This function
        must accept two positional arguments e.g. `y_true` and `y_pred`.
        Weighted usage requires support for a `sample_weight` keyword or
        generic keyword arguments.
    n_resamples : int, optional
        Number of resampling iterations, by default 9999.
    rng : np.random.Generator, optional
        Random number generator, by default np.random.default_rng().
    batch : int or None, optional
        Batch size for parallel processing, by default None.
    alternative : {'less', 'greater', 'two-sided'}, optional
        Defines the alternative hypothesis. Default is 'two-sided'.
    sample_weight : NDArray or None, optional
        Sample weights for each observation, by default None (equal weights).
        When provided, weights are normalised to sum to n_samples internally.
        Weights are fixed across all permutations: they reflect covariate
        importance, not label assignment.

    Raises
    ------
    ValueError
        If input arrays have incompatible lengths, if ``actual`` is not
        binary, if ``predicted`` has unsupported target type, if ``batch`` is
        invalid, or if sample weights fail
        validation.
    TypeError
        If ``metric`` does not expose a compatible scikit-learn metric
        signature for ``(y_true, y_pred, *, sample_weight=...)``.

    Notes
    -----
    The permutation null distribution is generated with
    :func:`scipy.stats.permutation_test` using ``permutation_type='pairings'``.
    This keeps score values fixed and randomizes pairings with labels.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.metrics import matthews_corrcoef, roc_auc_score
    >>> from samesame.ctst import CTST
    >>> actual = np.array([0, 1, 1, 0])
    >>> scores = np.array([0.2, 0.8, 0.6, 0.4])
    >>> ctst_mcc = CTST(actual, scores, metric=matthews_corrcoef)
    >>> ctst_auc = CTST(actual, scores, metric=roc_auc_score)
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
    rng: np.random.Generator = field(default_factory=np.random.default_rng)
    batch: int | None = None
    alternative: Literal["less", "greater", "two-sided"] = "two-sided"
    sample_weight: NDArray | None = field(default=None, repr=False)

    def __post_init__(self):
        """Validate inputs."""
        self.actual, self.predicted = validate_binary_actual_with_predicted(
            self.actual, self.predicted
        )
        type_predicted = type_of_target(self.predicted, "predicted")
        if type_predicted not in ("binary", "continuous", "multiclass"):
            raise ValueError(
                "Expected 'predicted' to be one of binary, continuous, or "
                f"multiclass, got {type_predicted}."
            )
        if not check_metric_function(self.metric):
            raise TypeError(
                "'metric' expects a callable that conforms to scikit-learn "
                "metric API and accepts ``sample_weight`` as a keyword "
                f"argument; got {signature(self.metric)!s}."
            )
        if self.batch is not None and self.batch < 1:
            raise ValueError("batch must be a positive integer or None.")
        self.sample_weight = validate_and_normalise_weights(
            self.sample_weight, len(self.actual)
        )

    @cached_property
    def _result(self):
        _w = self.sample_weight

        def statistic(*args):
            if _w is None:
                return self.metric(args[0], args[1])
            return self.metric(args[0], args[1], sample_weight=_w)

        return permutation_test(
            data=(self.actual, self.predicted),
            statistic=statistic,
            permutation_type="pairings",
            n_resamples=self.n_resamples,
            batch=self.batch,
            alternative=self.alternative,
            rng=self.rng,
        )

    @cached_property
    def statistic(self) -> float:
        """
        Observed value of the chosen classification metric.

        Returns
        -------
        float
            Observed metric value computed on ``(actual, predicted)``.

        Notes
        -----
        The result is cached to avoid (expensive) recomputation.
        """
        return self._result.statistic

    @cached_property
    def null(self) -> NDArray:
        """
        Permutation null distribution of the test statistic.

        Returns
        -------
        NDArray
            Metric values under random label-score pairings.

        Notes
        -----
        The result is cached to avoid (expensive) recomputation since the
        null distribution requires permutations.
        """
        return self._result.null_distribution

    @cached_property
    def pvalue(self) -> float:
        """
        Permutation p-value associated with the observed statistic.

        Returns
        -------
        float
            P-value under the selected ``alternative`` hypothesis.

        Notes
        -----
        The result is cached to avoid (expensive) recomputation.
        """
        return self._result.pvalue

    @classmethod
    def from_samples(
        cls,
        first_sample: NDArray,
        second_sample: NDArray,
        metric: Callable,
        n_resamples: int = 9999,
        rng: np.random.Generator | None = None,
        batch: int | None = None,
        alternative: Literal["less", "greater", "two-sided"] = "two-sided",
        sample_weight: NDArray | None = None,
    ):
        """
        Create a CTST instance from two samples.

        This constructor is convenient when score arrays are provided by
        sample rather than as a combined vector with binary labels.

        Parameters
        ----------
        first_sample : NDArray
            First sample of scores. These can be binary or continuous.
        second_sample : NDArray
            Second sample of scores. These can be binary or continuous.
        metric : Callable
            Classification metric accepting ``(y_true, y_pred, *, sample_weight)``.
        n_resamples : int, optional
            Number of permutation draws.
        rng : np.random.Generator or None, optional
            Random number generator. If None, a new default generator is used.
        batch : int or None, optional
            Batch size passed to :func:`scipy.stats.permutation_test`.
        alternative : {'less', 'greater', 'two-sided'}, optional
            Alternative hypothesis for p-value computation.
        sample_weight : NDArray or None, optional
            Optional sample weights aligned with the combined samples.

        Returns
        -------
        CTST
            An instance of the CTST class.

        Raises
        ------
        ValueError
            If sample target types differ or downstream CTST validation fails.
        """
        if rng is None:
            rng = np.random.default_rng()
        actual, predicted = build_two_sample_dataset(first_sample, second_sample)
        return cls(
            actual,
            predicted,
            metric,
            n_resamples,
            rng,
            batch,
            alternative,
            sample_weight,
        )


__all__ = [
    "CTST",
]
