from __future__ import annotations

import inspect
from dataclasses import FrozenInstanceError

import numpy as np
import pytest

import samesame as ss
from samesame import shift
from samesame.shift import HarmInference, HarmResult, ShiftResult
from samesame.weights import ImportanceWeights, from_domain_probabilities


def test_root_exports() -> None:
    assert hasattr(ss, "shift")
    assert hasattr(ss, "weights")
    assert hasattr(ss, "scores")
    assert hasattr(ss, "stats")
    assert not hasattr(ss, "test_shift")
    assert not hasattr(ss, "test_adverse_shift")
    assert not hasattr(ss, "adverse_shift_posterior")
    assert not hasattr(shift, "as_bf")


def test_signatures_take_positional_source_target() -> None:
    shift_sig = inspect.signature(shift.detect_shift)
    harm_sig = inspect.signature(shift.detect_harm)
    assert (
        shift_sig.parameters["source"].kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
    )
    assert harm_sig.parameters["source"].kind is inspect.Parameter.POSITIONAL_OR_KEYWORD


def test_detect_shift_returns_shift_result(
    shift_samples: dict[str, np.ndarray],
) -> None:
    result = shift.detect_shift(**shift_samples, n_resamples=64)
    assert isinstance(result, ShiftResult)
    assert result.statistic_name == "roc_auc"
    assert isinstance(result.statistic, float)
    assert 0.0 <= result.pvalue <= 1.0
    assert result.null_distribution.shape == (64,)


def test_detect_shift_accepts_positional_source_target(
    shift_samples: dict[str, np.ndarray],
) -> None:
    result = shift.detect_shift(shift_samples["source"], shift_samples["target"])
    assert isinstance(result, ShiftResult)


def test_detect_shift_rejects_unknown_statistic(
    shift_samples: dict[str, np.ndarray],
) -> None:
    with pytest.raises(ValueError, match="statistic must be one of"):
        shift.detect_shift(**shift_samples, statistic="f1")  # type: ignore[arg-type]


def test_binary_only_statistics_require_binary_scores(
    shift_samples: dict[str, np.ndarray],
) -> None:
    with pytest.raises(ValueError, match="requires binary outlier scores"):
        shift.detect_shift(**shift_samples, statistic="balanced_accuracy")


@pytest.mark.parametrize("statistic", ["balanced_accuracy", "matthews_corrcoef"])
def test_binary_only_statistics_accept_binary_scores(
    binary_shift_samples: dict[str, np.ndarray],
    statistic: str,
) -> None:
    result = shift.detect_shift(**binary_shift_samples, statistic=statistic)  # type: ignore[arg-type]
    assert isinstance(result, ShiftResult)
    assert result.statistic_name == statistic


def test_results_are_frozen(shift_samples: dict[str, np.ndarray]) -> None:
    result = shift.detect_shift(**shift_samples, n_resamples=64)
    with pytest.raises(FrozenInstanceError):
        result.pvalue = 0.0  # type: ignore[misc]


def test_detect_harm_requires_direction(
    confidence_samples: dict[str, np.ndarray],
) -> None:
    with pytest.raises(TypeError):
        shift.detect_harm(**confidence_samples)


def test_detect_harm_rejects_unknown_direction(
    confidence_samples: dict[str, np.ndarray],
) -> None:
    with pytest.raises(ValueError, match="direction must be one of"):
        shift.detect_harm(
            **confidence_samples,
            direction="up-is-bad",  # type: ignore[arg-type]
        )


def test_detect_harm_handles_higher_is_better(
    confidence_samples: dict[str, np.ndarray],
) -> None:
    primary = shift.detect_harm(
        **confidence_samples,
        direction="higher-is-better",
        n_resamples=64,
    )
    mirrored = shift.detect_harm(
        source=-confidence_samples["source"],
        target=-confidence_samples["target"],
        direction="higher-is-worse",
        n_resamples=64,
    )
    assert isinstance(primary, HarmResult)
    assert primary.direction == "higher-is-better"
    assert np.isclose(primary.statistic, mirrored.statistic)
    assert np.isclose(primary.pvalue, mirrored.pvalue)


def test_shift_null_distribution_matches_n_resamples(
    shift_samples: dict[str, np.ndarray],
) -> None:
    result = shift.detect_shift(**shift_samples, n_resamples=99)
    assert result.null_distribution.shape == (99,)


def test_shift_supports_explicit_weights(
    shift_samples: dict[str, np.ndarray],
) -> None:
    source, target = shift_samples["source"], shift_samples["target"]
    sample_weight = ImportanceWeights(
        source=np.linspace(1.0, 3.0, len(source)),
        target=np.linspace(1.0, 3.0, len(target)),
    )
    base = shift.detect_shift(**shift_samples, n_resamples=64)
    weighted = shift.detect_shift(
        **shift_samples, n_resamples=64, weights=sample_weight
    )
    assert isinstance(weighted, ShiftResult)
    assert base.statistic != weighted.statistic


def test_shift_supports_contextual_weights(
    shift_samples: dict[str, np.ndarray],
) -> None:
    rng = np.random.default_rng(42)
    source, target = shift_samples["source"], shift_samples["target"]
    source_prob = rng.uniform(0.2, 0.5, size=len(source))
    target_prob = rng.uniform(0.5, 0.8, size=len(target))
    weights = from_domain_probabilities(
        source_prob=source_prob, target_prob=target_prob, mode="source"
    )
    base = shift.detect_shift(**shift_samples, n_resamples=64)
    contextual = shift.detect_shift(**shift_samples, n_resamples=64, weights=weights)
    assert isinstance(contextual, ShiftResult)
    assert base.statistic != contextual.statistic


def test_harm_inference_returns_posterior(
    confidence_samples: dict[str, np.ndarray],
) -> None:
    result = shift.detect_harm(
        **confidence_samples,
        direction="higher-is-better",
        n_resamples=64,
        random_state=0,
    )
    evidence = shift.infer_harm(
        **confidence_samples,
        direction="higher-is-better",
        n_resamples=64,
        random_state=42,
    )
    assert isinstance(result, HarmResult)
    assert isinstance(evidence, HarmInference)
    assert evidence.posterior.shape == (64,)
    assert isinstance(evidence.bayes_factor, float)


def test_harm_detection_supports_importance_weights(
    confidence_samples: dict[str, np.ndarray],
) -> None:
    rng = np.random.default_rng(99)
    source, target = confidence_samples["source"], confidence_samples["target"]
    source_prob = rng.uniform(0.2, 0.5, size=len(source))
    target_prob = rng.uniform(0.5, 0.8, size=len(target))
    weights = from_domain_probabilities(
        source_prob=source_prob, target_prob=target_prob, mode="target"
    )
    base = shift.detect_harm(
        **confidence_samples, direction="higher-is-better", n_resamples=64
    )
    contextual = shift.detect_harm(
        **confidence_samples,
        direction="higher-is-better",
        n_resamples=64,
        weights=weights,
    )
    assert isinstance(contextual, HarmResult)
    assert base.statistic != contextual.statistic
