from __future__ import annotations

import numpy as np


class ECDFDiscrete:
    def __init__(self, x, freq_weights=None, side="right"):
        if side.lower() not in ["right", "left"]:
            raise ValueError(f"side can take the values 'right' or 'left', got {side}")
        x = np.asarray(x)
        if x.ndim != 1:
            raise ValueError("x must be one-dimensional.")
        if freq_weights is not None:
            freq_weights = np.asarray(freq_weights)
            if freq_weights.ndim != 1:
                raise ValueError("freq_weights must be one-dimensional.")
            if len(freq_weights) != len(x):
                raise ValueError("freq_weights must have the same length as x.")
            if np.any(freq_weights < 0):
                raise ValueError("freq_weights must be non-negative.")
            order = np.argsort(x)
            x_sorted = x[order]
            w_sorted = freq_weights[order]
            x_unique, first = np.unique(x_sorted, return_index=True)
            w_sum = np.add.reduceat(w_sorted, first)
            y = np.cumsum(w_sum) / np.sum(w_sum)
            x = x_unique
        else:
            x = np.sort(x)
            y = np.linspace(1.0 / len(x), 1.0, len(x))
        self.side = side
        self.x = np.r_[-np.inf, x]
        self.y = np.r_[0.0, y]

    def __call__(self, time):
        tind = np.searchsorted(self.x, time, self.side) - 1
        return self.y[tind]


__all__ = ["ECDFDiscrete"]
