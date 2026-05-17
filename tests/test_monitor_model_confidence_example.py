from __future__ import annotations

import numpy as np

from samesame import shift


def _logit_gap(logits: np.ndarray) -> np.ndarray:
    logits = np.asarray(logits, dtype=float)
    max_logits = np.max(logits, axis=1)
    mean_rest = (np.sum(logits, axis=1) - max_logits) / (logits.shape[1] - 1)
    return max_logits - mean_rest


def test_logit_gap_recipe_matches_known_values() -> None:
    logits = np.array([[5.0, 1.0, 0.5], [2.0, 2.1, 1.9]])
    np.testing.assert_allclose(_logit_gap(logits), np.array([4.25, 0.15]))


def test_confidence_recipe_feeds_harm_detection_in_expected_direction() -> None:
    source_logits = np.array(
        [
            [5.0, 0.5, 0.2],
            [4.8, 0.4, 0.3],
            [5.2, 0.6, 0.4],
            [4.9, 0.3, 0.2],
        ]
    )
    target_logits = np.array(
        [
            [2.1, 2.0, 1.9],
            [2.0, 1.9, 1.8],
            [2.2, 2.1, 2.0],
            [1.9, 1.8, 1.7],
        ]
    )

    source_scores = _logit_gap(source_logits)
    target_scores = _logit_gap(target_logits)

    harmful = shift.detect_harm(
        source_scores,
        target_scores,
        direction="higher-is-better",
        n_resamples=99,
        random_state=42,
    )
    reverse = shift.detect_harm(
        target_scores,
        source_scores,
        direction="higher-is-better",
        n_resamples=99,
        random_state=42,
    )

    assert source_scores.mean() > target_scores.mean()
    assert harmful.direction == "higher-is-better"
    assert harmful.statistic > reverse.statistic