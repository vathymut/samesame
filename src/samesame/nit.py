# Copyright (c) 2025-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from dataclasses import dataclass
from functools import cached_property

import numpy as np
from numpy.typing import NDArray
from sklearn.utils.multiclass import type_of_target

from samesame._bayesboot import bayesian_posterior
from samesame._data import assign_labels, concat_samples
from samesame._stats import _bayes_factor
from samesame.ctst import CTST
from samesame.metrics import wauc


@dataclass
class WeightedAUC(CTST):
    """
    Two-sample test for no adverse shift using the weighted AUC (WAUC).

    This test compares scores from two independent samples. We reject the
    null hypothesis of no adverse shift for unusually high values of the WAUC
    i.e. when the second sample is relatively worse than the first one. This
    is a robust nonparametric noninferiority test (NIT) with no pre-specified
    margin. It can be used, amongst other things, to detect dataset shift with
    outlier scores, hence the DSOS acronym.

    Attributes
    ----------
    actual : NDArray
        Binary indicator for sample membership.
    predicted : NDArray
        Estimated (predicted) scores for corresponding samples in `actual`.
    n_resamples : int, optional
        Number of resampling iterations, by default 9999.
    rng : np.random.Generator, optional
        Random number generator, by default np.random.default_rng().
    n_jobs : int, optional
        Number of parallel jobs, by default 1.
    batch : int or None, optional
        Batch size for parallel processing, by default None.

    See Also
    --------
    bayes.as_bf : Convert a one-sided p-value to a Bayes factor.

    bayes.as_pvalue : Convert a Bayes factor to a one-sided p-value.

    Notes
    -----
    The frequentist null distribution of the WAUC is based on permutations
    [1]. The Bayesian posterior distribution of the WAUC is based on the
    Bayesian bootstrap [2]. Because this is a one-tailed test of direction
    (it asks the question, 'are we worse off?'), we can convert a one-sided
    p-value into a Bayes factor and vice versa. We can also use these p-values
    for sequential testing [3].

    The test assumes that `predicted` are outlier scores and/or encode some
    notions of outlyingness; higher value of `predicted` indicates worse
    outcomes.

    References
    ----------
    .. [1] Kamulete, Vathy M. "Test for non-negligible adverse shifts."
       Uncertainty in Artificial Intelligence. PMLR, 2022.

    .. [2] Gu, Jiezhun, Subhashis Ghosal, and Anindya Roy. "Bayesian bootstrap
       estimation of ROC curve." Statistics in medicine 27.26 (2008): 5407-5420.

    .. [3] Kamulete, Vathy M. "Are you OK? A Bayesian Sequential Test for
       Adverse Shift." 2025.

    Examples
    --------
    >>> import numpy as np
    >>> from samesame.nit import WeightedAUC
    >>> # alternatively: from samesame.nit import DSOS
    >>> actual = np.array([0, 1, 1, 0])
    >>> scores = np.array([0.2, 0.8, 0.6, 0.4])
    >>> wauc = WeightedAUC(actual, scores)
    >>> print(wauc.pvalue) # doctest: +SKIP
    >>> print(wauc.bayes_factor) # doctest: +SKIP
    >>> wauc_ = WeightedAUC.from_samples(scores, scores)
    >>> isinstance(wauc_, WeightedAUC)
    True
    """

    def __init__(
        self,
        actual: NDArray,
        predicted: NDArray,
        n_resamples: int = 9999,
        rng: np.random.Generator = np.random.default_rng(),
        n_jobs: int = 1,
        batch: int | None = None,
    ):
        """Initialize WeightedAUC with fixed metric and alternative."""
        super().__init__(
            actual=actual,
            predicted=predicted,
            metric=wauc,
            n_resamples=n_resamples,
            rng=rng,
            n_jobs=n_jobs,
            batch=batch,
            alternative="greater",
        )

    @cached_property
    def posterior(self) -> NDArray:
        """
        Compute the posterior distribution of the WAUC.

        Returns
        -------
        NDArray
            The posterior distribution of the WAUC.

        Notes
        -----
        The result is cached to avoid (expensive) recomputation since the
        posterior distribution uses the Bayesian bootstrap.
        """
        return bayesian_posterior(
            self.actual,
            self.predicted,
            self.metric,
            self.n_resamples,
            self.rng,
        )

    @cached_property
    def bayes_factor(self):
        """
        Compute the Bayes factor using the Bayesian bootstrap.

        Notes
        -----
        The result is cached to avoid (expensive) recomputation.
        """
        bayes_threshold = float(np.mean(self.null))
        bf_ = _bayes_factor(self.posterior, bayes_threshold)
        return bf_

    @classmethod
    def from_samples(
        cls,
        first_sample: NDArray,
        second_sample: NDArray,
        n_resamples: int = 9999,
        rng: np.random.Generator = np.random.default_rng(),
        n_jobs: int = 1,
        batch: int | None = None,
    ):
        """
        Create a WeightedAUC instance from two samples.

        Parameters
        ----------
        first_sample : NDArray
            First sample of scores. These can be binary or continuous.
        second_sample : NDArray
            Second sample of scores. These can be binary or continuous.

        Returns
        -------
        WeightedAUC
            An instance of the WeightedAUC class.
        """
        assert type_of_target(first_sample) == type_of_target(second_sample)
        samples = (first_sample, second_sample)
        actual = assign_labels(samples)
        predicted = concat_samples(samples)
        return cls(
            actual,
            predicted,
            n_resamples,
            rng,
            n_jobs,
            batch,
        )


DSOS = WeightedAUC

__all__ = [
    "WeightedAUC",
    "DSOS",
]
