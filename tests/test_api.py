from __future__ import annotations

import inspect
from dataclasses import FrozenInstanceError
from enum import Enum

import numpy as np
import pytest

import samesame as ss
from samesame import shift
from samesame.shift import BayesianHarmResult, Direction, HarmResult, ShiftResult
from samesame.weights import ImportanceWeights, from_domain_probabilities


def test_direction_is_enum_with_two_members() -> None:
    assert issubclass(Direction, Enum)
    assert {member.name for member in Direction} == {
        "HIGHER_IS_WORSE",
        "HIGHER_IS_BETTER",
    }


def test_root_exports() -> None:
    assert ss.detect_shift is shift.detect_shift
    assert ss.detect_harm is shift.detect_harm
    assert ss.Direction is Direction
    assert ss.from_domain_probabilities is from_domain_probabilities
    assert ss.shift is shift
    assert ss.weights is ss.weights
    assert "detect_shift" in ss.__all__
    assert "detect_harm" in ss.__all__
    assert "Direction" in ss.__all__
    assert "from_domain_probabilities" in ss.__all__
    assert "shift" in ss.__all__
    assert "weights" in ss.__all__
    assert "scores" not in ss.__all__
    assert not hasattr(ss, "stats")
    assert not hasattr(ss, "test_shift")
    assert not hasattr(ss, "test_adverse_shift")
    assert not hasattr(ss, "adverse_shift_posterior")
    assert not hasattr(shift, "as_bf")
    assert not hasattr(shift, "infer_harm")


def test_flat_harm_detection_matches_namespace(
    confidence_samples: dict[str, np.ndarray],
) -> None:
    flat = ss.detect_harm(
        **confidence_samples,
        direction=Direction.HIGHER_IS_BETTER,
        n_resamples=64,
    )
    namespaced = shift.detect_harm(
        **confidence_samples,
        direction=Direction.HIGHER_IS_BETTER,
        n_resamples=64,
    )
    assert flat.statistic == namespaced.statistic
    assert flat.pvalue == namespaced.pvalue


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


def test_detect_shift_rejects_batch_kwarg(
    shift_samples: dict[str, np.ndarray],
) -> None:
    with pytest.raises(TypeError, match="batch"):
        shift.detect_shift(**shift_samples, batch=100)  # type: ignore[call-arg]


def test_detect_harm_requires_direction(
    confidence_samples: dict[str, np.ndarray],
) -> None:
    with pytest.raises(TypeError):
        shift.detect_harm(**confidence_samples)


def test_detect_harm_accepts_direction_enum(
    confidence_samples: dict[str, np.ndarray],
) -> None:
    result = shift.detect_harm(
        **confidence_samples,
        direction=Direction.HIGHER_IS_BETTER,
        n_resamples=64,
    )
    assert isinstance(result, HarmResult)
    assert result.direction is Direction.HIGHER_IS_BETTER


def test_detect_harm_rejects_raw_direction_string(
    confidence_samples: dict[str, np.ndarray],
) -> None:
    with pytest.raises(TypeError, match="must be a samesame.shift.Direction"):
        shift.detect_harm(**confidence_samples, direction="higher-is-worse")  # type: ignore[arg-type]


def test_detect_harm_handles_higher_is_better(
    confidence_samples: dict[str, np.ndarray],
) -> None:
    primary = shift.detect_harm(
        **confidence_samples,
        direction=Direction.HIGHER_IS_BETTER,
        n_resamples=64,
    )
    mirrored = shift.detect_harm(
        source=-confidence_samples["source"],
        target=-confidence_samples["target"],
        direction=Direction.HIGHER_IS_WORSE,
        n_resamples=64,
    )
    assert isinstance(primary, HarmResult)
    assert primary.direction is Direction.HIGHER_IS_BETTER
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


def test_shift_rejects_wrong_length_importance_weights(
    shift_samples: dict[str, np.ndarray],
) -> None:
    source, target = shift_samples["source"], shift_samples["target"]
    sample_weight = ImportanceWeights(
        source=np.ones(len(source) - 1),
        target=np.ones(len(target)),
    )
    with pytest.raises(ValueError, match="weights.source has wrong length"):
        shift.detect_shift(**shift_samples, weights=sample_weight)


def test_shift_rejects_invalid_importance_weight_values(
    shift_samples: dict[str, np.ndarray],
) -> None:
    source, target = shift_samples["source"], shift_samples["target"]
    with pytest.raises(ValueError, match="weights.target must contain only finite"):
        sample_weight = ImportanceWeights(
            source=np.ones(len(source)),
            target=np.full(len(target), np.inf),
        )
        shift.detect_shift(**shift_samples, weights=sample_weight)


def test_shift_rejects_non_finite_scores() -> None:
    with pytest.raises(ValueError, match="source must contain only finite"):
        shift.detect_shift(
            source=np.array([0.1, np.nan, 0.3]),
            target=np.array([0.4, 0.5, 0.6]),
        )


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


def test_detect_harm_has_no_posterior_fields(
    confidence_samples: dict[str, np.ndarray],
) -> None:
    result = shift.detect_harm(
        **confidence_samples,
        direction=Direction.HIGHER_IS_BETTER,
        n_resamples=64,
        random_state=0,
    )
    assert isinstance(result, HarmResult)
    assert not hasattr(result, "posterior")
    assert not hasattr(result, "bayes_factor")


def test_detect_harm_bayesian_flag_returns_bayesian_harm_result(
    confidence_samples: dict[str, np.ndarray],
) -> None:
    result = shift.detect_harm(
        **confidence_samples,
        direction=Direction.HIGHER_IS_BETTER,
        bayesian=True,
        n_resamples=64,
        random_state=42,
    )
    assert isinstance(result, BayesianHarmResult)
    assert isinstance(result, HarmResult)
    assert result.posterior.shape == (64,)
    assert isinstance(result.bayes_factor, float)


def test_detect_harm_bayesian_matches_detect_harm_statistic(
    confidence_samples: dict[str, np.ndarray],
) -> None:
    base = shift.detect_harm(
        **confidence_samples,
        direction=Direction.HIGHER_IS_BETTER,
        n_resamples=64,
        random_state=42,
    )
    with_bayes = shift.detect_harm(
        **confidence_samples,
        direction=Direction.HIGHER_IS_BETTER,
        bayesian=True,
        n_resamples=64,
        random_state=42,
    )
    assert np.isclose(base.statistic, with_bayes.statistic)
    assert np.isclose(base.pvalue, with_bayes.pvalue)
    assert base.direction == with_bayes.direction
    assert np.allclose(base.null_distribution, with_bayes.null_distribution)


def test_detect_harm_rejects_non_finite_threshold(
    confidence_samples: dict[str, np.ndarray],
) -> None:
    with pytest.raises(ValueError, match="threshold must be finite"):
        shift.detect_harm(
            **confidence_samples,
            direction=Direction.HIGHER_IS_BETTER,
            bayesian=True,
            threshold=float("inf"),
        )


def test_detect_harm_rejects_threshold_without_bayesian(
    confidence_samples: dict[str, np.ndarray],
) -> None:
    with pytest.raises(ValueError, match="threshold is only meaningful when bayesian=True"):
        shift.detect_harm(
            **confidence_samples,
            direction=Direction.HIGHER_IS_BETTER,
            threshold=0.1,
        )


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
        **confidence_samples, direction=Direction.HIGHER_IS_BETTER, n_resamples=64
    )
    contextual = shift.detect_harm(
        **confidence_samples,
        direction=Direction.HIGHER_IS_BETTER,
        n_resamples=64,
        weights=weights,
    )
    assert isinstance(contextual, HarmResult)
    assert base.statistic != contextual.statistic


def test_shift_result_repr_omits_null_distribution(
    shift_samples: dict[str, np.ndarray],
) -> None:
    result = shift.detect_shift(**shift_samples, n_resamples=64)
    rendered = repr(result)
    assert "null_distribution" not in rendered
    assert "ShiftResult(" in rendered
    assert "statistic=" in rendered
    assert "pvalue=" in rendered


def test_harm_result_repr_includes_direction(
    confidence_samples: dict[str, np.ndarray],
) -> None:
    result = shift.detect_harm(
        **confidence_samples,
        direction=Direction.HIGHER_IS_WORSE,
        n_resamples=64,
    )
    assert "Direction.HIGHER_IS_WORSE" in repr(result)


def test_significant_uses_pvalue_alpha(shift_samples: dict[str, np.ndarray]) -> None:
    result = shift.detect_shift(**shift_samples, n_resamples=64)
    assert result.significant() == (result.pvalue <= 0.05)
    assert result.significant(alpha=0.5) == (result.pvalue <= 0.5)


def test_significant_rejects_invalid_alpha(shift_samples: dict[str, np.ndarray]) -> None:
    result = shift.detect_shift(**shift_samples, n_resamples=64)
    with pytest.raises(ValueError, match="alpha must be in"):
        result.significant(alpha=0.0)
    with pytest.raises(ValueError, match="alpha must be in"):
        result.significant(alpha=1.0)
    with pytest.raises(ValueError, match="alpha must be in"):
        result.significant(alpha=np.nan)
