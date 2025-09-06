# Copyright (c) 2025-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

"""
Functions for working with p-values and Bayes factors.

This module provides functions for converting between p-values and Bayes
factors, as well as calculating Bayes factors from posterior
distributions. These are useful for hypothesis testing and interpreting
statistical evidence.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from numpy.typing import NDArray
from scipy.special import expit, logit

from samesame._stats import _bayes_factor


def as_bf(
    pvalue: NDArray | float,
) -> NDArray | float:
    """
    Convert a one-sided p-value to a Bayes factor.

    This Bayes factor quantifies evidence in favour of a directional effect
    over the null hypothesis of no such effect.

    Parameters
    ----------
    pvalue : NDArray | float
        The p-value(s) to be converted to Bayes factor(s). Can be a single
        value or an array of values.

    Returns
    -------
    NDArray | float
        The corresponding Bayes factor(s). The return type matches the
        input type.

    See Also
    --------
    as_pvalue : Convert a Bayes factor to a p-value.

    Notes
    -----
    The Bayes factor is derived from the one-sided p-value using a
    Bayesian interpretation, which quantifies evidence in favor of a
    directional effect over the null hypothesis of no such effect.

    See [1]_ for theoretical details and implications.

    References
    ----------
    .. [1] Marsman, Maarten, and Eric-Jan Wagenmakers. "Three Insights from
       a Bayesian Interpretation of the One-Sided P Value." *Educational and
       Psychological Measurement*, vol. 77, no. 3, 2017, pp. 529-539.
       doi:10.1177/0013164416669201.

    Examples
    --------
    >>> import numpy as np
    >>> from samesame.bayes import as_bf
    >>> as_bf(0.5)
    np.float64(1.0)
    >>> as_bf(np.array([0.05, 0.1, 0.5])) # doctest: +NORMALIZE_WHITESPACE
    array([19., 9., 1.])
    """
    if np.any(np.logical_or(pvalue >= 1, pvalue <= 0)):
        raise ValueError("pvalue must be within the open interval (0, 1).")
    pvalue = np.clip(pvalue, 1e-10, 1 - 1e-10)  # Avoid numerical issues at extremes
    return 1.0 / np.exp(logit(pvalue))  # evidence in favor of an effect


def as_pvalue(
    bayes_factor: float | NDArray,
) -> float | NDArray:
    """
    Convert a Bayes factor of a directional effect to a one-sided p-value.

    The Bayes factor quantifies evidence in favour of a directional effect
    over the null hypothesis of no such effect.

    Parameters
    ----------
    bayes_factor : float | NDArray
        The Bayes factor(s) to be converted to p-value(s). Can be a single
        value or an array of values.

    Returns
    -------
    float | NDArray
        The corresponding p-value(s). The return type matches the input type.

    See Also
    --------
    as_pvalue : Convert a Bayes factor to a p-value.

    Notes
    -----
    The equivalence between the Bayes factor in favor of a directional effect
    and the one-sided p-value for the null of no such effect is discussed in
    [1]_.

    References
    ----------
    .. [1] Marsman, Maarten, and Eric-Jan Wagenmakers. "Three Insights from a
       Bayesian Interpretation of the One-Sided P Value." *Educational and
       Psychological Measurement*, vol. 77, no. 3, 2017, pp. 529–539,
       https://doi.org/10.1177/0013164416669201.

    Examples
    --------
    >>> import numpy as np
    >>> from samesame.bayes import as_pvalue
    >>> as_pvalue(1)
    np.float64(0.5)
    >>> as_pvalue(np.array([19.0, 9.0, 1.0])) # doctest: +NORMALIZE_WHITESPACE
    array([0.05, 0.1 , 0.5 ])
    """
    if np.any(bayes_factor <= 0):
        raise ValueError("bayes_factor must be strictly positive.")
    bf_ = np.clip(bayes_factor, 1e-10, 1e10)  # Ensure numerical stability
    pvalue = expit(-np.log(bf_))
    return pvalue


def bayes_factor(
    posterior: NDArray,
    threshold: float = 0.0,
    adjustment: Literal[0, 1] = 0,
) -> float:
    """
    Compute the Bayes factor for a test of direction given a threshold.

    The Bayes factor quantifies the evidence in favor of the hypothesis that
    the posterior distribution exceeds the given threshold compared to the
    hypothesis that it does not.

    Parameters
    ----------
    posterior : NDArray
        An array of posterior samples.
    threshold : float, optional
        The threshold value to test against. Default is 0.0.
    adjustment : {0, 1}, optional
        Adjustment to apply to the Bayes factor calculation. Default is 0.

    Returns
    -------
    float
        The computed Bayes factor.

    See Also
    --------
    as_pvalue : Convert a Bayes factor to a p-value.

    as_bf : Convert a p-value to a Bayes factor.

    Notes
    -----
    The Bayes factor is a measure of evidence which compares the likelihood of
    the data under two competing hypotheses. In this function, the numerator
    represents the proportion of posterior samples exceeding the threshold,
    while the denominator represents the proportion of samples not exceeding
    the threshold. If all samples exceed the threshold, the Bayes factor is
    set to infinity, indicating overwhelming evidence in favor of the
    hypothesis.

    The adjustment parameter allows for slight modifications to the Bayes
    factor calculation, which can be useful in specific contexts such as
    sensitivity analyses.

    See [1]_ for further theoretical details and practical implications of
    this approach.

    References
    ----------
    .. [1] Marsman, Maarten, and Eric-Jan Wagenmakers. "Three Insights from a
       Bayesian Interpretation of the One-Sided P Value." *Educational and
       Psychological Measurement*, vol. 77, no. 3, 2017, pp. 529-539.
       doi:10.1177/0013164416669201.

    Examples
    --------
    >>> import numpy as np
    >>> from samesame.bayes import bayes_factor
    >>> posterior_samples = np.array([0.2, 0.5, 0.8, 0.9])
    >>> bayes_factor(posterior_samples, threshold=0.5)
    np.float64(1.0)
    """
    return _bayes_factor(posterior, threshold, adjustment)


__all__ = ["as_bf", "as_pvalue", "bayes_factor"]
