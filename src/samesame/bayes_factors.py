# Copyright (c) 2025-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

"""Re-export shim for Bayesian utilities.

Implementations live in :mod:`samesame._bayesboot`.  This module exists so
that existing import paths (``from samesame.bayes_factors import as_bf``)
keep working.
"""

from samesame._bayesboot import as_bf, as_pvalue, bayes_factor

__all__ = ["as_bf", "as_pvalue", "bayes_factor"]
