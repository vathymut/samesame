from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

from samesame import shift


def _load_example_module():
    module_path = (
        Path(__file__).resolve().parents[1]
        / "docs/examples/credit/_code/monitor_model_confidence_example.py"
    )
    spec = importlib.util.spec_from_file_location(
        "monitor_model_confidence_example",
        module_path,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load example module from {module_path}.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


example = _load_example_module()


def test_logit_gap_recipe_matches_known_values() -> None:
    logits = np.array([[5.0, 1.0, 0.5], [2.0, 2.1, 1.9]])
    np.testing.assert_allclose(example.logit_gap(logits), np.array([4.25, 0.15]))


def test_confidence_recipe_feeds_harm_detection_in_expected_direction() -> None:
    source_probabilities = np.array(
        [
            [0.01, 0.99],
            [0.03, 0.97],
            [0.02, 0.98],
            [0.04, 0.96],
        ]
    )
    target_probabilities = np.array(
        [
            [0.42, 0.58],
            [0.45, 0.55],
            [0.40, 0.60],
            [0.48, 0.52],
        ]
    )

    source_scores = example.outlier_scores_from_probabilities(source_probabilities)
    target_scores = example.outlier_scores_from_probabilities(target_probabilities)

    harmful = shift.detect_harmful_shift(
        source_scores,
        target_scores,
        higher_is_worse=False,
        n_resamples=99,
        rng=np.random.default_rng(42),
    )
    reverse = shift.detect_harmful_shift(
        target_scores,
        source_scores,
        higher_is_worse=False,
        n_resamples=99,
        rng=np.random.default_rng(42),
    )

    assert source_scores.mean() > target_scores.mean()
    assert harmful.higher_is_worse is False
    assert harmful.statistic > reverse.statistic
