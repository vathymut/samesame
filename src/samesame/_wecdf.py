# Copyright (c) 2025-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
"""
Empirical weighted CDF Functions

Copied/Adapted from:
https://www.statsmodels.org/dev/_modules/statsmodels/distributions/empirical_distribution.html#ECDFDiscrete
"""

import numpy as np


class StepFunction:
    """
    A basic step function.

    Values at the ends are handled in the simplest way possible:
    everything to the left of x[0] is set to ival; everything
    to the right of x[-1] is set to y[-1].

    Parameters
    ----------
    x : array_like
    y : array_like
    ival : float
        ival is the value given to the values to the left of x[0]. Default
        is 0.
    sorted : bool
        Default is False.
    side : {'left', 'right'}, optional
        Default is 'left'. Defines the shape of the intervals constituting the
        steps. 'right' correspond to [a, b) intervals and 'left' to (a, b].

    """

    def __init__(self, x, y, ival=0.0, sorted=False, side="left"):  # noqa
        if side.lower() not in ["right", "left"]:
            msg = "side can take the values 'right' or 'left'"
            raise ValueError(msg)
        self.side = side

        _x = np.asarray(x)
        _y = np.asarray(y)

        if _x.shape != _y.shape:
            msg = "x and y do not have the same shape"
            raise ValueError(msg)
        if len(_x.shape) != 1:
            msg = "x and y must be 1-dimensional"
            raise ValueError(msg)

        self.x = np.r_[-np.inf, _x]
        self.y = np.r_[ival, _y]

        if not sorted:
            asort = np.argsort(self.x)
            self.x = np.take(self.x, asort, 0)
            self.y = np.take(self.y, asort, 0)
        self.n = self.x.shape[0]

    def __call__(self, time):
        tind = np.searchsorted(self.x, time, self.side) - 1 # type: ignore
        return self.y[tind]


class ECDFDiscrete(StepFunction):
    """
    Return the Empirical Weighted CDF of an array as a step function.

    Parameters
    ----------
    x : array_like
        Data values. If freq_weights is None, then x is treated as observations
        and the ecdf is computed from the frequency counts of unique values
        using nunpy.unique.
        If freq_weights is not None, then x will be taken as the support of the
        mass point distribution with freq_weights as counts for x values.
        The x values can be arbitrary sortable values and need not be integers.
    freq_weights : array_like
        Weights of the observations.  sum(freq_weights) is interpreted as nobs
        for confint.
        If freq_weights is None, then the frequency counts for unique values
        will be computed from the data x.
    side : {'left', 'right'}, optional
        Default is 'right'. Defines the shape of the intervals constituting the
        steps. 'right' correspond to [a, b) intervals and 'left' to (a, b].

    Returns
    -------
    Weighted ECDF as a step function.
    """

    def __init__(self, x, freq_weights=None, side="right"):
        if freq_weights is None:
            x, freq_weights = np.unique(x, return_counts=True)
        else:
            x = np.asarray(x)
        assert len(freq_weights) == len(x)
        w = np.asarray(freq_weights)
        sw = np.sum(w)
        assert sw > 0
        ax = x.argsort()
        x = x[ax]
        y = np.cumsum(w[ax])
        y = y / sw
        super().__init__(x, y, side=side, sorted=True)
