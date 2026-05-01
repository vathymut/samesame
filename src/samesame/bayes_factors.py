# Copyright (c) 2025-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

"""Utilities for one-sided p-values and Bayes factors.

The functions in this module provide numerically stable conversions between
one-sided p-values and Bayes factors for directional hypotheses, and direct
Bayes factor estimation from posterior draws.
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

    This is useful when a directional p-value is available and evidence is
    needed on a Bayes-factor scale. Smaller p-values map to larger Bayes
    factors in favour of a directional effect.

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

    Raises
    ------
    ValueError
        If any p-value is not strictly within the open interval (0, 1).

    See Also
    --------
    as_pvalue : Convert a Bayes factor to a p-value.

    Notes
    -----
    The mapping is based on the one-sided p-value interpretation in [1]_.
    Inputs are clipped near 0 and 1 for numerical stability.

    References
    ----------
    .. [1] Marsman, Maarten, and Eric-Jan Wagenmakers. "Three Insights from
       a Bayesian Interpretation of the One-Sided P Value." *Educational and
       Psychological Measurement*, vol. 77, no. 3, 2017, pp. 529-539.
       doi:10.1177/0013164416669201.

    Examples
    --------
    >>> import numpy as np
    >>> from samesame.bayes_factors import as_bf
    >>> as_bf(0.5)
    np.float64(1.0)
    >>> np.round(as_bf(0.05), 1)
    np.float64(19.0)
    >>> as_bf(np.array([0.05, 0.1, 0.5])) # doctest: +NORMALIZE_WHITESPACE
    array([19., 9., 1.])
    """
    if np.any(np.logical_or(pvalue >= 1, pvalue <= 0)):
        raise ValueError("pvalue must be within the open interval (0, 1).")
    pvalue = np.clip(pvalue, 1e-10, 1 - 1e-10)
    return 1.0 / np.exp(logit(pvalue))


def as_pvalue(
    bayes_factor: float | NDArray,
) -> float | NDArray:
    """
    Convert a Bayes factor of a directional effect to a one-sided p-value.

    This is useful when evidence is summarized as a Bayes factor but
    reporting requires one-sided p-values under the directional null.

    Parameters
    ----------
    bayes_factor : float | NDArray
        The Bayes factor(s) to be converted to p-value(s). Can be a single
        value or an array of values.

    Returns
    -------
    float | NDArray
        The corresponding p-value(s). The return type matches the input type.

    Raises
    ------
    ValueError
        If any Bayes factor is not strictly positive.

    See Also
    --------
    as_bf : Convert a one-sided p-value to a Bayes factor.

    Notes
    -----
    This is the inverse mapping of :func:`as_bf` under the same directional
    interpretation [1]_. Inputs are clipped to improve numerical stability.

    References
    ----------
    .. [1] Marsman, Maarten, and Eric-Jan Wagenmakers. "Three Insights from a
       Bayesian Interpretation of the One-Sided P Value." *Educational and
       Psychological Measurement*, vol. 77, no. 3, 2017, pp. 529–539,
       https://doi.org/10.1177/0013164416669201.

    Examples
    --------
    >>> import numpy as np
    >>> from samesame.bayes_factors import as_pvalue
    >>> as_pvalue(1)
    np.float64(0.5)
    >>> np.round(as_pvalue(19), 2)
    np.float64(0.05)
    >>> as_pvalue(np.array([19.0, 9.0, 1.0])) # doctest: +NORMALIZE_WHITESPACE
    array([0.05, 0.1 , 0.5 ])
    """
    if np.any(bayes_factor <= 0):
        raise ValueError("bayes_factor must be strictly positive.")
    bf_ = np.clip(bayes_factor, 1e-10, 1e10)
    pvalue = expit(-np.log(bf_))
    return pvalue


def bayes_factor(
    posterior: NDArray,
    threshold: float = 0.0,
    adjustment: Literal[0, 1] = 0,
) -> float:
    """
    Compute a directional Bayes factor from posterior samples.

    The Bayes factor compares posterior support for values above a threshold
    against support for values at or below that threshold.

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
        Bayes factor in favour of the posterior mass being above
        ``threshold``.

    See Also
    --------
    as_pvalue : Convert a Bayes factor to a p-value.

    as_bf : Convert a p-value to a Bayes factor.

    Notes
    -----
    If all posterior draws exceed ``threshold``, the denominator is zero and
    the returned Bayes factor can become infinite. ``adjustment`` can be used
    to regularize this edge case in finite samples.

    References
    ----------
    .. [1] Marsman, Maarten, and Eric-Jan Wagenmakers. "Three Insights from a
       Bayesian Interpretation of the One-Sided P Value." *Educational and
       Psychological Measurement*, vol. 77, no. 3, 2017, pp. 529-539.
       doi:10.1177/0013164416669201.

    Examples
    --------
    >>> import numpy as np
    >>> from samesame.bayes_factors import bayes_factor
    >>> posterior_samples = np.array([0.2, 0.5, 0.8, 0.9])
    >>> bayes_factor(posterior_samples, threshold=0.5)
    np.float64(1.0)
    >>> np.isinf(bayes_factor(posterior_samples, threshold=0.0))
    np.True_
    """
    return _bayes_factor(posterior, threshold, adjustment)


__all__ = ["as_bf", "as_pvalue", "bayes_factor"]
