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

from samesame._bayesboot import bayesian_posterior
from samesame._data import build_two_sample_dataset
from samesame._stats import _bayes_factor
from samesame.ctst import CTST
from samesame.metrics import wauc


@dataclass
class WeightedAUC(CTST):
    """
    Two-sample test for no adverse shift using the weighted AUC (WAUC).

    This test compares scores from two independent samples. Small p-values and
    large Bayes factors indicate evidence of adverse shift, where the second
    sample tends to receive worse (higher) outlier scores than the first.
    This is a robust nonparametric noninferiority test (NIT) with no
    pre-specified margin, also referred to as DSOS.

    Parameters
    ----------
    actual : NDArray
        Binary indicator for sample membership.
    predicted : NDArray
        Estimated (predicted) scores for corresponding samples in `actual`.
    n_resamples : int, optional
        Number of resampling iterations, by default 9999.
    rng : np.random.Generator, optional
        Random number generator, by default np.random.default_rng().
    batch : int or None, optional
        Batch size for parallel processing, by default None.
    sample_weight : NDArray or None, optional
        Sample weights for each observation, by default None (equal weights).
        When provided, weights are normalised to sum to n_samples internally.
        Weights are fixed across all permutations and combined multiplicatively
        with Bayesian bootstrap Dirichlet draws for the posterior computation.

    Raises
    ------
    ValueError
        If input validation inherited from :class:`samesame.ctst.CTST` fails.

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
        rng: np.random.Generator | None = None,
        batch: int | None = None,
        sample_weight: NDArray | None = None,
    ):
        """Initialize WeightedAUC."""
        if rng is None:
            rng = np.random.default_rng()
        super().__init__(
            actual=actual,
            predicted=predicted,
            metric=wauc,
            n_resamples=n_resamples,
            rng=rng,
            batch=batch,
            alternative="greater",
            sample_weight=sample_weight,
        )

    @cached_property
    def posterior(self) -> NDArray:
        """
        Bayesian-bootstrap posterior distribution of WAUC.

        Returns
        -------
        NDArray
            Posterior WAUC draws with length ``n_resamples``.

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
            base_weight=self.sample_weight,
        )

    @cached_property
    def bayes_factor(self) -> float:
        """
        Bayes factor for adverse shift from posterior and permutation threshold.

        Returns
        -------
        float
            Bayes factor in favour of adverse shift.

        Notes
        -----
        The threshold is the mean of the permutation null distribution.
        The result is cached to avoid (expensive) recomputation.
        """
        return _bayes_factor(self.posterior, float(np.mean(self.null)))

    @classmethod
    def from_samples(
        cls,
        first_sample: NDArray,
        second_sample: NDArray,
        n_resamples: int = 9999,
        rng: np.random.Generator | None = None,
        batch: int | None = None,
    ):
        """
        Create a WeightedAUC instance from two samples.

        This constructor is convenient when score arrays are provided by
        sample rather than as a combined vector with binary labels.

        Parameters
        ----------
        first_sample : NDArray
            First sample of scores. These can be binary or continuous.
        second_sample : NDArray
            Second sample of scores. These can be binary or continuous.
        n_resamples : int, optional
            Number of permutation and posterior resamples.
        rng : np.random.Generator or None, optional
            Random number generator. If None, a new default generator is used.
        batch : int or None, optional
            Batch size passed to the permutation engine.

        Returns
        -------
        WeightedAUC
            An instance of the WeightedAUC class.

        Raises
        ------
        ValueError
            If sample target types differ or downstream validation fails.
        """
        if rng is None:
            rng = np.random.default_rng()
        actual, predicted = build_two_sample_dataset(first_sample, second_sample)
        return cls(
            actual=actual,
            predicted=predicted,
            n_resamples=n_resamples,
            rng=rng,
            batch=batch,
        )


DSOS = WeightedAUC

__all__ = [
    "WeightedAUC",
    "DSOS",
]
