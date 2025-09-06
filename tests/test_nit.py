# Copyright (c) 2025-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import numpy as np
import pytest

from samesame.nit import WeightedAUC


def test_input_shape_and_length(
    unequal_length_predictions,
    wrong_shape_predictions,
):
    with pytest.raises(ValueError):
        WeightedAUC(**wrong_shape_predictions)
    with pytest.raises(ValueError):
        WeightedAUC(**unequal_length_predictions)


def test_bayes_attributes(decent_predictions, n_resamples=60):
    ctst = WeightedAUC(**decent_predictions, n_resamples=n_resamples)
    for attr in ["posterior", "bayes_factor"]:
        assert hasattr(ctst, attr)
    assert isinstance(ctst.posterior, np.ndarray)
    assert ctst.posterior.shape[0] == int(n_resamples)
    assert ctst.bayes_factor > 0.0
