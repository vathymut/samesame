from __future__ import annotations

import inspect
from dataclasses import FrozenInstanceError

import numpy as np
import pytest

import samesame as ss
from samesame import shift
from samesame.shift import HarmResult, ShiftResult
from samesame.weights import ImportanceWeights, domain_weights


def test_root_exports() -> None:
    assert ss.detect_shift is shift.detect_shift
    assert ss.detect_harm is shift.detect_harm
    assert ss.domain_weights is domain_weights
    assert ss.ImportanceWeights is ImportanceWeights
    assert ss.shift is shift
    assert ss.weights is ss.weights
    assert "detect_shift" in ss.__all__
    assert "detect_harm" in ss.__all__
    assert "domain_weights" in ss.__all__
    assert not hasattr(ss, "from_domain_probabilities")
    assert "ImportanceWeights" in ss.__all__
    assert "shift" in ss.__all__
    assert "weights" in ss.__all__
    assert "scores" not in ss.__all__
    assert not hasattr(ss, "stats")
    assert not hasattr(ss, "test_shift")
    assert not hasattr(ss, "test_adverse_shift")
    assert not hasattr(ss, "adverse_shift_posterior")
    assert not hasattr(shift, "as_bf")
    assert not hasattr(shift, "infer_harm")
    assert not hasattr(shift, "BayesianHarmResult")
    assert not hasattr(shift, "detect_harm_bayesian")
    assert not hasattr(shift, "ShiftStatistic")


def test_detect_shift_signature_is_minimal() -> None:
    params = set(inspect.signature(shift.detect_shift).parameters)
    assert params == {
        "source", "target", "n_resamples", "batch", "rng", "weights"
    }


def test_detect_harm_signature_is_minimal() -> None:
    params = set(inspect.signature(shift.detect_harm).parameters)
    assert params == {
        "source",
        "target",
        "worse",
        "n_resamples",
        "batch",
        "rng",
        "weights",
    }


def test_batch_is_passed_to_permutation_test(
    shift_samples: dict[str, np.ndarray],
) -> None:
    result = shift.detect_shift(**shift_samples, n_resamples=64, batch=4)
    assert result.null_distribution.shape == (64,)


@pytest.mark.parametrize("batch", [0, -1, 1.5, True])
def test_batch_rejects_invalid_values(
    shift_samples: dict[str, np.ndarray], batch: object
) -> None:
    with pytest.raises(ValueError, match="batch must be None or a positive integer"):
        shift.detect_shift(**shift_samples, batch=batch)  # type: ignore[arg-type]


def test_flat_harm_detection_matches_namespace(
    confidence_samples: dict[str, np.ndarray],
) -> None:
    flat = ss.detect_harm(
        **confidence_samples,
        worse="lower",
        n_resamples=64,
    )
    namespaced = shift.detect_harm(
        **confidence_samples,
        worse="lower",
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
    assert isinstance(result.statistic, float)
    assert 0.0 <= result.pvalue <= 1.0
    assert result.null_distribution.shape == (64,)


def test_detect_shift_accepts_positional_source_target(
    shift_samples: dict[str, np.ndarray],
) -> None:
    result = shift.detect_shift(shift_samples["source"], shift_samples["target"])
    assert isinstance(result, ShiftResult)


def test_results_are_frozen(shift_samples: dict[str, np.ndarray]) -> None:
    result = shift.detect_shift(**shift_samples, n_resamples=64)
    with pytest.raises(FrozenInstanceError):
        result.pvalue = 0.0  # type: ignore[misc]


def test_detect_harm_requires_worse(
    confidence_samples: dict[str, np.ndarray],
) -> None:
    with pytest.raises(TypeError):
        shift.detect_harm(**confidence_samples)


def test_detect_harm_accepts_worse(
    confidence_samples: dict[str, np.ndarray],
) -> None:
    result = shift.detect_harm(
        **confidence_samples,
        worse="lower",
        n_resamples=64,
    )
    assert isinstance(result, HarmResult)
    assert result.worse == "lower"


def test_detect_harm_rejects_invalid_worse(
    confidence_samples: dict[str, np.ndarray],
) -> None:
    with pytest.raises(ValueError, match="worse must be"):
        shift.detect_harm(**confidence_samples, worse="sideways")  # type: ignore[arg-type]


def test_detect_harm_handles_higher_is_better(
    confidence_samples: dict[str, np.ndarray],
) -> None:
    primary = shift.detect_harm(
        **confidence_samples,
        worse="lower",
        n_resamples=64,
    )
    mirrored = shift.detect_harm(
        source=-confidence_samples["source"],
        target=-confidence_samples["target"],
        worse="higher",
        n_resamples=64,
    )
    assert isinstance(primary, HarmResult)
    assert primary.worse == "lower"
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
    weights = domain_weights(
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
        worse="lower",
        n_resamples=64,
        rng=0,
    )
    assert isinstance(result, HarmResult)
    assert not hasattr(result, "posterior")
    assert not hasattr(result, "bayes_factor")


def test_harm_detection_supports_importance_weights(
    confidence_samples: dict[str, np.ndarray],
) -> None:
    rng = np.random.default_rng(99)
    source, target = confidence_samples["source"], confidence_samples["target"]
    source_prob = rng.uniform(0.2, 0.5, size=len(source))
    target_prob = rng.uniform(0.5, 0.8, size=len(target))
    weights = domain_weights(
        source_prob=source_prob, target_prob=target_prob, mode="target"
    )
    base = shift.detect_harm(
        **confidence_samples, worse="lower", n_resamples=64
    )
    contextual = shift.detect_harm(
        **confidence_samples,
        worse="lower",
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


def test_harm_result_repr_includes_worse(
    confidence_samples: dict[str, np.ndarray],
) -> None:
    result = shift.detect_harm(
        **confidence_samples,
        worse="higher",
        n_resamples=64,
    )
    assert "worse='higher'" in repr(result)


def test_significant_uses_pvalue_alpha(shift_samples: dict[str, np.ndarray]) -> None:
    result = shift.detect_shift(**shift_samples, n_resamples=64)
    assert result.significant() == (result.pvalue <= 0.05)
    assert result.significant(alpha=0.5) == (result.pvalue <= 0.5)


def test_significant_rejects_invalid_alpha(
    shift_samples: dict[str, np.ndarray],
) -> None:
    result = shift.detect_shift(**shift_samples, n_resamples=64)
    with pytest.raises(ValueError, match="alpha must be in"):
        result.significant(alpha=0.0)
    with pytest.raises(ValueError, match="alpha must be in"):
        result.significant(alpha=1.0)
    with pytest.raises(ValueError, match="alpha must be in"):
        result.significant(alpha=np.nan)
